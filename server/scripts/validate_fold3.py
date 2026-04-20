"""
fold3モデル（best.pt）をTOMのval画像セットで検証するスクリプト。

使い方:
  cd server
  python scripts/validate_fold3.py \
    --images  /path/to/tom_fold3_val/images \
    --masks   /path/to/tom_fold3_val/masks \
    --model   data/models/pytorch/current_pt/best.pt \
    --out     data/validate_results

出力:
  data/validate_results/
    ├── {stem}_overlay.png   # 入力画像 + 予測マスク(赤) + GTマスク(緑) の重ね合わせ
    └── summary.csv          # ファイル名, Dice, 前景ピクセル数(pred/gt) の一覧

マスク形式:
  - グレースケール PNG: 白(>128)=前景
  - RGBA PNG: R>128 & G<128 & A>128 を前景とみなす（Annottyエクスポート形式）
  - いずれも自動判別
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

# server/ ディレクトリをパスに追加（config, model をimport）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_INPUT_SIZE, IN_CHANNELS, NUM_CLASSES
from model import create_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ─── ユーティリティ ───────────────────────────────────────

def load_model(model_path: str):
    model = create_model()
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    epoch = ckpt.get("epoch", "?")
    fold  = ckpt.get("fold_idx", "?")
    print(f"[model] loaded: fold={fold}, epoch={epoch}, device={DEVICE}")
    return model


def preprocess(image_path: str) -> torch.Tensor:
    """IMAGE → (1, 3, 256, 256) tensor  (Z-score正規化、inference.pyと同一)"""
    img = Image.open(image_path).convert("L")  # グレースケール
    arr = np.array(img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))).astype(np.float32)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)
    std = arr.std()
    arr = (arr - arr.mean()) / std if std > 1e-6 else arr - arr.mean()
    arr_3ch = np.stack([arr, arr, arr], axis=0).astype(np.float32)  # (3, H, W)
    return torch.from_numpy(arr_3ch).unsqueeze(0).to(DEVICE)


def load_class_mask(mask_path: str, color_to_class: dict) -> np.ndarray:
    """マスクPNG → (H, W) int64 class_id ndarray（元サイズのまま）
    グレースケール: ピクセル値 = class_id（TOM形式）
    RGBA: color_to_class で class_id に変換（Annotty形式）
    """
    m = Image.open(mask_path)
    if m.mode == "L":
        return np.array(m, dtype=np.int64)
    rgba = np.array(m.convert("RGBA"), dtype=np.uint8)
    result = np.zeros(rgba.shape[:2], dtype=np.int64)
    for (r, g, b), class_id in color_to_class.items():
        px = (rgba[:,:,0]==r) & (rgba[:,:,1]==g) & (rgba[:,:,2]==b) & (rgba[:,:,3]>128)
        result[px] = class_id
    return result


def dice_score_multiclass(pred: np.ndarray, gt: np.ndarray,
                          num_classes: int, smooth: float = 1.0) -> float:
    """前景クラス(1〜num_classes-1)のマクロ平均Diceを返す。"""
    dice_list = []
    for c in range(1, num_classes):
        pred_c = pred == c
        gt_c = gt == c
        inter = (pred_c & gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice_list.append(float((2.0 * inter + smooth) / (union + smooth)))
    return float(np.mean(dice_list)) if dice_list else 0.0


def make_overlay(image_path: str, pred: np.ndarray, gt: np.ndarray,
                 out_path: str, class_colors: dict, alpha: int = 140):
    """
    元画像に pred（クラス色）と gt（半透明白枠）を重ねて保存。
    pred / gt は元画像サイズの (H, W) int64 class_id マップ。
    """
    base = Image.open(image_path).convert("RGBA")
    w, h = base.size

    # pred → クラス色で塗り分け
    pred_arr = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, (r, g, b) in class_colors.items():
        px = pred == class_id
        pred_arr[px] = [r, g, b, alpha]
    pred_layer = Image.fromarray(pred_arr, "RGBA")

    # gt → 白の輪郭（前景全体）
    gt_arr = np.zeros((h, w, 4), dtype=np.uint8)
    gt_arr[gt > 0] = [255, 255, 255, alpha // 2]
    gt_layer = Image.fromarray(gt_arr, "RGBA")

    out = Image.alpha_composite(base, gt_layer)
    out = Image.alpha_composite(out, pred_layer)
    out.convert("RGB").save(out_path)


# ─── メイン ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="fold3モデルをTOM valセットで検証")
    parser.add_argument("--images", required=True, help="val画像ディレクトリ")
    parser.add_argument("--masks",  required=True, help="GTマスクディレクトリ")
    parser.add_argument("--model",  default=None,  help="best.pt パス（省略時: config.BEST_MODEL_PATH）")
    parser.add_argument("--out",    default="data/validate_results", help="結果出力ディレクトリ")
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoidしきい値（デフォルト0.5）")
    args = parser.parse_args()

    # モデルパス解決
    if args.model is None:
        from config import BEST_MODEL_PATH
        args.model = BEST_MODEL_PATH

    if not os.path.exists(args.model):
        sys.exit(f"[error] model not found: {args.model}")
    if not os.path.isdir(args.images):
        sys.exit(f"[error] images dir not found: {args.images}")
    if not os.path.isdir(args.masks):
        sys.exit(f"[error] masks dir not found: {args.masks}")

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model)

    # 画像一覧収集
    image_files = sorted([
        f for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    print(f"[info] {len(image_files)} images found in {args.images}")

    results = []
    dice_list = []

    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.images, fname)

        # GTマスクを同名で探す（拡張子違いも試みる）
        mask_path = None
        for ext in [".png", ".jpg", ".bmp"]:
            candidate = os.path.join(args.masks, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f"[skip] no mask for {fname}")
            continue

        # カラーマップ（初回のみロード）
        if "color_to_class" not in dir():
            import json
            cfg_path = os.path.join(args.images, "..", "label_config.json")
            if os.path.exists(cfg_path):
                cfg = json.loads(open(cfg_path, encoding="utf-8").read())
                color_to_class = {tuple(c["color"]): c["id"] for c in cfg.get("classes", [])}
                class_colors   = {c["id"]: tuple(c["color"]) for c in cfg.get("classes", [])}
            else:
                color_to_class = {(255, 0, 0): 1}
                class_colors   = {1: (255, 0, 0)}

        # 推論: softmax → argmax（マルチクラス）
        tensor = preprocess(img_path)
        with torch.no_grad():
            logits = model(tensor)                                      # (1, C, H, W)
        probs = torch.softmax(logits, dim=1).squeeze(0)                # (C, H, W)
        pred_small = probs.argmax(dim=0).cpu().numpy().astype(np.int64) # (H, W)

        # 元サイズにアップスケール
        orig_w, orig_h = Image.open(img_path).size
        pred_full = np.array(
            Image.fromarray(pred_small.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST),
            dtype=np.int64,
        )

        # GT読み込み
        gt_full = load_class_mask(mask_path, color_to_class)

        # Dice計算（マルチクラス）
        d = dice_score_multiclass(pred_full, gt_full, num_classes=NUM_CLASSES)
        dice_list.append(d)
        results.append({
            "file":    fname,
            "dice":    f"{d:.4f}",
            "pred_fg": int((pred_full > 0).sum()),
            "gt_fg":   int((gt_full > 0).sum()),
        })
        print(f"  {fname:40s}  dice={d:.4f}  pred_fg={int((pred_full>0).sum()):6d}px  gt_fg={int((gt_full>0).sum()):6d}px")

        # オーバーレイ画像保存
        overlay_path = os.path.join(args.out, f"{stem}_overlay.png")
        make_overlay(img_path, pred_full, gt_full, overlay_path, class_colors)

    # CSV保存
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "dice", "pred_fg", "gt_fg"])
        writer.writeheader()
        writer.writerows(results)

    # サマリー表示
    print()
    print("=" * 60)
    print(f"  評価枚数: {len(dice_list)}")
    if dice_list:
        print(f"  mean Dice: {np.mean(dice_list):.4f}")
        print(f"  std  Dice: {np.std(dice_list):.4f}")
        print(f"  min  Dice: {np.min(dice_list):.4f}  ({image_files[np.argmin(dice_list)]})")
        print(f"  max  Dice: {np.max(dice_list):.4f}  ({image_files[np.argmax(dice_list)]})")
    print(f"  結果保存先: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
