"""
推論パイプライン: 画像パス → マルチクラスマスクPNG (bytes)
マスク仕様: RGBA PNG, クラスごとに label_config.json の色で塗り分け, 背景=透明
5-fold アンサンブル推論対応: softmax確率を平均してargmax
前処理: グレースケール + percentile clip(1-99%) + Z-score + 3ch複製
  (TOM NIfTI学習時と同一の正規化方式)
"""
import os
import io
import json
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import create_model
from config import (
    IMAGE_SIZE, MODEL_INPUT_SIZE, N_FOLDS, NUM_CLASSES,
    BEST_MODEL_PATH, get_fold_model_path,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# アンサンブルモデルリスト（再学習完了時にリロード）
_models: list = []
_models_loaded: bool = False


def _load_single_model(model_path: str):
    """単一モデルをロードして返す（CPU/GPU自動選択）"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # チェックポイントキー両対応
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model")
        or checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model


def load_ensemble():
    """fold modelをロード。なければbest.ptにfallback。"""
    global _models, _models_loaded

    fold_paths = [get_fold_model_path(i) for i in range(N_FOLDS)]
    existing_folds = [p for p in fold_paths if os.path.exists(p)]

    if existing_folds:
        _models = [_load_single_model(p) for p in existing_folds]
        logger.info(f"Ensemble loaded: {len(_models)} models ({DEVICE})")
    elif os.path.exists(BEST_MODEL_PATH):
        _models = [_load_single_model(BEST_MODEL_PATH)]
        logger.info(f"Single model loaded (fallback): {BEST_MODEL_PATH} ({DEVICE})")
    else:
        _models = []
        logger.warning("No model files found")

    _models_loaded = True


def _preprocess(image_path: str) -> torch.Tensor:
    """画像をモデル入力テンソルに変換。
    TOM学習時と同一の前処理:
      グレースケール → percentile clip(1-99%) → Z-score → 3ch複製
      → MODEL_INPUT_SIZE×MODEL_INPUT_SIZE にリサイズ
    """
    img = Image.open(image_path).convert("L")  # グレースケール
    arr = np.array(img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))).astype(np.float32)

    # percentile clip (1-99%)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)

    # Z-score正規化
    std = arr.std()
    arr = (arr - arr.mean()) / std if std > 1e-6 else arr - arr.mean()

    # グレースケール → 3ch複製 (H, W) → (3, H, W)
    arr_3ch = np.stack([arr, arr, arr], axis=0).astype(np.float32)
    tensor = torch.from_numpy(arr_3ch).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(DEVICE)


def _load_color_map(image_path: str) -> dict[int, tuple[int, int, int]]:
    """ケースの label_config.json から {class_id: (R, G, B)} を返す。
    見つからない場合はデフォルトカラー（クラス1=赤のみ）を返す。
    """
    config_path = Path(image_path).parent.parent / "label_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        return {c["id"]: tuple(c["color"]) for c in cfg.get("classes", [])}
    # fallback: class 1 のみ赤
    logger.warning(f"label_config.json が見つかりません: {config_path}")
    return {1: (255, 0, 0)}


def _class_mask_to_rgba(
    pred: np.ndarray,
    original_size: tuple,
    class_colors: dict[int, tuple[int, int, int]],
) -> bytes:
    """
    pred (H, W) int → マルチクラスRGBA PNG bytes
    class_colors の色で塗り分け。背景(class 0) → 透明。
    """
    h, w = pred.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, (r, g, b) in class_colors.items():
        px = pred == class_id
        rgba[px, 0] = r
        rgba[px, 1] = g
        rgba[px, 2] = b
        rgba[px, 3] = 255

    mask_img = Image.fromarray(rgba, "RGBA").resize(original_size, Image.NEAREST)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


def run_inference(image_path: str, model_path: str = None, color_map: dict = None) -> bytes | None:
    """
    画像に対してアンサンブル推論を実行し、マルチクラスマスクPNGのバイト列を返す。
    モデルファイルが存在しない場合は None を返す。
    color_map: 後方互換のため残置。カラーマップはlabel_config.jsonから自動ロード。
    """
    global _models, _models_loaded

    if not _models_loaded:
        load_ensemble()
        if not _models:
            return None

    if not _models:
        return None

    original_size = Image.open(image_path).size  # (W, H)
    tensor = _preprocess(image_path)
    class_colors = _load_color_map(image_path)

    with torch.no_grad():
        logits_list = [model(tensor) for model in _models]
        # softmax確率をアンサンブル平均 → argmax
        probs_list = [torch.softmax(l, dim=1) for l in logits_list]
        avg_probs = torch.stack(probs_list).mean(dim=0)      # (1, C, H, W)
        pred = avg_probs.squeeze(0).argmax(dim=0).cpu().numpy()  # (H, W) int

    result = _class_mask_to_rgba(pred, original_size, class_colors)
    logger.info(f"推論完了: {image_path}")
    return result


def reload_model():
    """再学習完了後に呼ばれる。次の推論時に新モデルを自動ロード"""
    global _models, _models_loaded
    _models = []
    _models_loaded = False
    logger.info("モデルリロード予約完了（次回推論時にアンサンブルをロードします）")
