"""
TOM500 NIfTI → PNG 変換スクリプト

変換方針:
  - 正規化はDataset側（trainer.py / inference.py）で行うため、PNGには焼き込まない
  - 保存形式: percentile clip(1-99%) → 0-255スケール → uint8グレースケールPNG
  - スライス方向: np.rot90(k=-1) を適用（TOM学習時と同じorientation）
  - 出力構造:
      <out_dir>/<case_id>/images/slice_NNN.png   （画像）
      <out_dir>/<case_id>/labels/slice_NNN.png   （ラベル: 0=背景, 1-9=眼窩構造クラス → グレースケール値そのまま）

使い方:
  python scripts/nifti_to_png.py \
      --data-dir D:/TOM500/dataset/train \
      --out-dir  D:/Annotty_MRI/server/data/slices_tom \
      [--skip-empty-slices]  # 前景ピクセルが0枚のスライスをスキップ

引数:
  --data-dir        TOM500のtrain/またはval/ディレクトリ（image/とlabel/を含む）
  --out-dir         出力先ルートディレクトリ
  --skip-empty      前景ピクセルなしのスライスをスキップ（デフォルト: False）
  --lo              percentile clip 下限（デフォルト: 1.0）
  --hi              percentile clip 上限（デフォルト: 99.0）
"""

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def percentile_scale(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """percentile clip → 0-255 uint8スケール。ボリューム全体で計算。"""
    low = float(np.percentile(vol, lo))
    high = float(np.percentile(vol, hi))
    clipped = np.clip(vol, low, high)
    if high > low:
        scaled = (clipped - low) / (high - low) * 255.0
    else:
        scaled = np.zeros_like(clipped)
    return scaled.astype(np.uint8)


def convert_case(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    skip_empty: bool,
    lo: float,
    hi: float,
) -> tuple[int, int]:
    """1症例のNIfTIをPNGスライスに変換。(保存枚数, スキップ枚数) を返す。"""
    case_id = img_path.stem.replace(".nii", "")
    img_out = out_dir / case_id / "images"
    lbl_out = out_dir / case_id / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    img_vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.int64)

    # ボリューム全体でpercentile clip → 0-255スケール
    img_scaled = percentile_scale(img_vol, lo, hi)

    n_slices = img_vol.shape[2]
    saved = skipped = 0

    for si in range(n_slices):
        # TOM orientation: rot90(k=-1)
        img_sl = np.rot90(img_scaled[:, :, si], k=-1).copy()
        lbl_sl = np.rot90(lbl_vol[:, :, si], k=-1).copy().astype(np.uint8)

        if skip_empty and lbl_sl.max() == 0:
            skipped += 1
            continue

        fname = f"slice_{si:03d}.png"
        Image.fromarray(img_sl, mode="L").save(img_out / fname)
        Image.fromarray(lbl_sl, mode="L").save(lbl_out / fname)
        saved += 1

    return saved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="TOM500 NIfTI → PNG変換")
    parser.add_argument("--data-dir", required=True,
                        help="TOM500 train/またはval/ディレクトリ（image/ と label/ を含む）")
    parser.add_argument("--out-dir", required=True,
                        help="出力先ルートディレクトリ")
    parser.add_argument("--skip-empty", action="store_true",
                        help="前景ピクセルなしのスライスをスキップ")
    parser.add_argument("--lo", type=float, default=1.0, help="percentile clip 下限")
    parser.add_argument("--hi", type=float, default=99.0, help="percentile clip 上限")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    img_dir = data_dir / "image"
    lbl_dir = data_dir / "label"

    if not img_dir.is_dir():
        sys.exit(f"[error] image dir not found: {img_dir}")
    if not lbl_dir.is_dir():
        sys.exit(f"[error] label dir not found: {lbl_dir}")

    img_paths = sorted(img_dir.glob("*.nii*"))
    if not img_paths:
        sys.exit(f"[error] No NIfTI files found in {img_dir}")

    print(f"[info] {len(img_paths)} cases found")
    print(f"[info] skip_empty={args.skip_empty}, clip=[{args.lo}, {args.hi}]")
    print(f"[info] out_dir={out_dir}")

    total_saved = total_skipped = 0
    for i, img_path in enumerate(img_paths):
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            print(f"[warn] label not found: {lbl_path}, skipping")
            continue
        saved, skipped = convert_case(img_path, lbl_path, out_dir,
                                      args.skip_empty, args.lo, args.hi)
        total_saved += saved
        total_skipped += skipped
        print(f"  [{i+1:3d}/{len(img_paths)}] {img_path.name}: "
              f"saved={saved}, skipped={skipped}")

    print(f"\n[done] total saved={total_saved}, skipped={total_skipped}")
    print(f"[done] out_dir={out_dir}")


if __name__ == "__main__":
    main()
