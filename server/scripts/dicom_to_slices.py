"""
DICOM → PNG スライス変換スクリプト

nifti/raw/ 配下の .dcm ファイルを症例ごとにグループ化し、
slices/{case_id}/images/ に W/L 適用済み PNG として出力する。

ファイル命名規則:
  {case_id}_{slice_number}.dcm
  例: 100_20150904_T2_Coronal_0.dcm → case_id=100_20150904_T2_Coronal, slice=0

実行方法:
  python server/scripts/dicom_to_slices.py
  python server/scripts/dicom_to_slices.py --case 100_20150904_T2_Coronal  # 特定症例のみ
  python server/scripts/dicom_to_slices.py --overwrite  # 既存スライスを上書き
"""
import os
import re
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pydicom
from PIL import Image

# サーバールートをパスに追加
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVER_DIR)

from config import NIFTI_RAW_DIR, SLICES_DIR

LABEL_CONFIG = {
    "classes": [
        {"id": 1, "name": "SR",  "color": [255, 0, 0]},
        {"id": 2, "name": "LR",  "color": [255, 128, 0]},
        {"id": 3, "name": "MR",  "color": [255, 255, 0]},
        {"id": 4, "name": "IR",  "color": [0, 255, 0]},
        {"id": 5, "name": "ON",  "color": [0, 255, 255]},
        {"id": 6, "name": "FAT", "color": [0, 128, 255]},
        {"id": 7, "name": "LG",  "color": [0, 0, 255]},
        {"id": 8, "name": "SO",  "color": [128, 0, 255]},
        {"id": 9, "name": "EB",  "color": [255, 0, 255]},
    ]
}


def apply_window_level(data: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2
    upper = center + width / 2
    out = np.clip(data.astype(np.float32), lower, upper)
    out = (out - lower) / (upper - lower) * 255.0
    return out.astype(np.uint8)


def auto_window_level(data: np.ndarray) -> tuple[float, float]:
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return float(data.max() / 2), float(data.max())
    p2, p98 = np.percentile(nonzero, [2, 98])
    center = (p2 + p98) / 2
    width = max(p98 - p2, 1.0)
    return float(center), float(width)


def group_dcm_files(raw_dir: str) -> dict[str, list[tuple[int, str]]]:
    """DCMファイルを症例ごとにグループ化。{case_id: [(slice_num, filepath), ...]}"""
    pattern = re.compile(r'^(.+)_(\d+)\.dcm$', re.IGNORECASE)
    groups = defaultdict(list)

    for fname in os.listdir(raw_dir):
        m = pattern.match(fname)
        if not m:
            continue
        case_id = m.group(1)
        slice_num = int(m.group(2))
        groups[case_id].append((slice_num, os.path.join(raw_dir, fname)))

    # スライス番号でソート
    for case_id in groups:
        groups[case_id].sort(key=lambda x: x[0])

    return dict(groups)


def convert_case(case_id: str, slices: list[tuple[int, str]], output_base: str, overwrite: bool) -> bool:
    """1症例分のDICOM→PNG変換"""
    case_dir = os.path.join(output_base, case_id)
    images_dir = os.path.join(case_dir, "images")
    manifest_path = os.path.join(case_dir, "slice_manifest.json")

    if os.path.exists(manifest_path) and not overwrite:
        print(f"  [SKIP] {case_id} (既存。--overwrite で上書き可能)")
        return False

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.join(case_dir, "annotations"), exist_ok=True)

    # 全スライスのピクセル値を読み込んでW/Lを自動推定
    pixel_arrays = []
    for _, fpath in slices:
        ds = pydicom.dcmread(fpath)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        pixel_arrays.append(arr * slope + intercept)

    all_pixels = np.concatenate([a.ravel() for a in pixel_arrays])
    wc, ww = auto_window_level(all_pixels)

    # DICOMヘッダーにW/Lがあれば優先
    ds0 = pydicom.dcmread(slices[0][1])
    if hasattr(ds0, 'WindowCenter') and hasattr(ds0, 'WindowWidth'):
        try:
            wc = float(ds0.WindowCenter) if not hasattr(ds0.WindowCenter, '__iter__') else float(ds0.WindowCenter[0])
            ww = float(ds0.WindowWidth)  if not hasattr(ds0.WindowWidth,  '__iter__') else float(ds0.WindowWidth[0])
        except Exception:
            pass

    slice_info = []
    for i, (slice_num, _) in enumerate(slices):
        arr = pixel_arrays[i]
        sl_8bit = apply_window_level(arr, wc, ww)
        pil_img = Image.fromarray(sl_8bit, mode="L")
        filename = f"slice_{i:03d}.png"
        pil_img.save(os.path.join(images_dir, filename))
        slice_info.append({"filename": filename, "slice_index": slice_num, "original_shape": list(arr.shape)})

    manifest = {
        "volume_id": case_id,
        "source": "dicom",
        "n_slices": len(slices),
        "volume_shape": [len(slices), *pixel_arrays[0].shape],
        "window_center": wc,
        "window_width": ww,
        "slices": slice_info,
        "created_at": datetime.now().isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    config_path = os.path.join(case_dir, "label_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_CONFIG, f, indent=2, ensure_ascii=False)

    return True


def main():
    parser = argparse.ArgumentParser(description="DICOM → PNG スライス変換")
    parser.add_argument("--case", help="特定の症例IDのみ変換")
    parser.add_argument("--overwrite", action="store_true", help="既存スライスを上書き")
    args = parser.parse_args()

    print(f"入力ディレクトリ: {NIFTI_RAW_DIR}")
    print(f"出力ディレクトリ: {SLICES_DIR}")

    groups = group_dcm_files(NIFTI_RAW_DIR)
    if not groups:
        print("DCMファイルが見つかりません。")
        return

    if args.case:
        if args.case not in groups:
            print(f"症例 '{args.case}' が見つかりません。")
            return
        groups = {args.case: groups[args.case]}

    print(f"\n対象症例数: {len(groups)}")
    converted = 0

    for case_id, slices in sorted(groups.items()):
        print(f"\n[{case_id}] {len(slices)} スライス")
        ok = convert_case(case_id, slices, SLICES_DIR, args.overwrite)
        if ok:
            print(f"  → 変換完了 (W/L自動推定)")
            converted += 1

    print(f"\n=== 完了: {converted}/{len(groups)} 症例を変換 ===")


if __name__ == "__main__":
    main()
