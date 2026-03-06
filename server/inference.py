"""
推論パイプライン: 画像パス → 赤色マスクPNG (bytes)
マスク仕様: RGBA PNG, 血管=赤(255,0,0,255), 背景=透明(0,0,0,0)
5-fold アンサンブル推論対応: sigmoid予測を平均して閾値適用
"""
import os
import io
import logging

import torch
import numpy as np
from PIL import Image

from model import create_model
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, N_FOLDS, BEST_MODEL_PATH, get_fold_model_path

logger = logging.getLogger(__name__)

# アンサンブルモデルリスト（再学習完了時にリロード）
_models: list = []
_models_loaded: bool = False

_MEAN = np.array(IMAGENET_MEAN)
_STD = np.array(IMAGENET_STD)

# 内接円マスク（眼底画像の有効領域）をキャッシュ
_circle_mask = None


def _get_circle_mask(size: int) -> np.ndarray:
    """画像に内接する円のブールマスクを返す（円内=True）"""
    global _circle_mask
    if _circle_mask is not None and _circle_mask.shape[0] == size:
        return _circle_mask
    cx, cy, r = size // 2, size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    _circle_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
    return _circle_mask


def _load_single_model(model_path: str):
    """単一モデルをGPUにロードして返す"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().cuda()
    return model


def load_ensemble():
    """5つのfold modelをGPUにロード。fold modelが無い場合はbest.ptにfallback。"""
    global _models, _models_loaded

    # fold modelsを探す
    fold_paths = [get_fold_model_path(i) for i in range(N_FOLDS)]
    existing_folds = [p for p in fold_paths if os.path.exists(p)]

    if existing_folds:
        _models = []
        for p in existing_folds:
            _models.append(_load_single_model(p))
        logger.info(f"Ensemble loaded: {len(_models)} models")
    elif os.path.exists(BEST_MODEL_PATH):
        # fallback: best.pt 単体モデル
        _models = [_load_single_model(BEST_MODEL_PATH)]
        logger.info(f"Single model loaded (fallback): {BEST_MODEL_PATH}")
    else:
        _models = []
        logger.warning("No model files found")

    _models_loaded = True


def run_inference(image_path: str, model_path: str):
    """
    画像に対してアンサンブル推論を実行し、赤色マスクPNGのバイト列を返す。
    fold modelが存在する場合は5モデルのsigmoid予測を平均。
    モデルファイルが存在しない場合は None を返す。
    """
    global _models, _models_loaded

    # モデルが未ロードならロード
    if not _models_loaded:
        load_ensemble()
        if not _models:
            return None

    if not _models:
        return None

    # 1. 画像読み込み・前処理
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # 2. テンソル変換 (ImageNet正規化)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # 3. アンサンブル推論: 全モデルのsigmoid予測を平均
    with torch.no_grad():
        preds = []
        for model in _models:
            logits = model(tensor)
            preds.append(torch.sigmoid(logits))
        avg_pred = torch.stack(preds).mean(dim=0)

    # 4. バイナリ化 + 内接円マスク適用
    mask = (avg_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    circle = _get_circle_mask(IMAGE_SIZE)
    mask[~circle] = 0  # 円外の予測を除去

    # 5. 赤色マスクPNG生成 (RGBA: 血管=赤(255,0,0,255), 背景=透明(0,0,0,0))
    rgba = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    rgba[mask == 1, 0] = 255  # R
    rgba[mask == 1, 3] = 255  # A (不透明)

    mask_img = Image.fromarray(rgba, "RGBA")
    if original_size != (IMAGE_SIZE, IMAGE_SIZE):
        mask_img = mask_img.resize(original_size, Image.NEAREST)

    # 6. PNGバイト列として返す
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


def reload_model():
    """再学習完了後に呼ばれる。次の推論時に新モデルを自動ロード"""
    global _models, _models_loaded
    _models = []
    _models_loaded = False
    logger.info("モデルリロード予約完了（次回推論時にアンサンブルをロードします）")
