"""
推論パイプライン（マルチクラス版）
画像パス → インデックスカラーPNG (bytes)
マスク仕様: RGBA PNG, 各クラスはlabel_config.jsonの色で着色
10クラスセグメンテーション: argmax → クラスインデックス → カラーPNG
5-fold アンサンブル推論対応: softmax予測を平均してargmax
"""
import os
import io
import json
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model import create_model
from config import (
    IMAGE_SIZE, NUM_CLASSES, SLICES_DIR,
    N_FOLDS, BEST_MODEL_PATH, get_fold_model_path,
)

logger = logging.getLogger(__name__)

# アンサンブルモデルリスト（再学習完了時にリロード）
_models: list = []
_models_loaded: bool = False

# デフォルトのカラーマップ（label_config.jsonが無い場合のfallback）
DEFAULT_CLASS_COLORS = {
    0: (0, 0, 0, 0),        # 背景 → 透明
    1: (255, 0, 0, 255),    # SR
    2: (255, 128, 0, 255),  # LR
    3: (255, 255, 0, 255),  # MR
    4: (0, 255, 0, 255),    # IR
    5: (0, 255, 255, 255),  # ON
    6: (0, 128, 255, 255),  # FAT
    7: (0, 0, 255, 255),    # LG
    8: (128, 0, 255, 255),  # SO
    9: (255, 0, 255, 255),  # EB
}


def _load_class_colors(case_id: str) -> dict[int, tuple]:
    """label_config.json からクラスID→RGBA色 マッピングを構築"""
    config_path = os.path.join(SLICES_DIR, case_id, "label_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        colors = {0: (0, 0, 0, 0)}  # 背景は常に透明
        for cls in config.get("classes", []):
            r, g, b = cls["color"]
            colors[cls["id"]] = (r, g, b, 255)
        return colors
    return DEFAULT_CLASS_COLORS


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


def run_inference(image_path: str, model_path: str = None, case_id: str = None):
    """
    画像に対してマルチクラス推論を実行し、インデックスカラーマスクPNGのバイト列を返す。
    fold modelが存在する場合はアンサンブル（softmax平均→argmax）。

    Args:
        image_path: 入力画像パス
        model_path: 未使用（互換性のため残存）
        case_id: カラーマップ参照用の症例ID（Noneならデフォルト色）

    Returns:
        PNG bytes or None
    """
    global _models, _models_loaded

    # モデルが未ロードならロード
    if not _models_loaded:
        load_ensemble()
        if not _models:
            return None

    if not _models:
        return None

    # 1. 画像読み込み: grayscale 1ch
    img = Image.open(image_path).convert("L")
    original_size = img.size  # (W, H)
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # 2. テンソル変換: [0, 1] 正規化
    arr = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().cuda()  # (1, 1, H, W)

    # 3. アンサンブル推論: 全モデルのsoftmax予測を平均 → argmax
    with torch.no_grad():
        preds = []
        for model in _models:
            logits = model(tensor)  # (1, C, H, W)
            preds.append(F.softmax(logits, dim=1))
        avg_pred = torch.stack(preds).mean(dim=0)  # (1, C, H, W)

    # 4. argmax → クラスインデックスマスク
    class_mask = torch.argmax(avg_pred, dim=1).squeeze().cpu().numpy()  # (H, W)

    # 5. インデックスカラーPNG生成
    class_colors = _load_class_colors(case_id) if case_id else DEFAULT_CLASS_COLORS

    rgba = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    for class_id, color in class_colors.items():
        if class_id == 0:
            continue  # 背景は透明のまま
        mask = class_mask == class_id
        if mask.any():
            rgba[mask] = color

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
