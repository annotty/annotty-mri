"""NIfTI → PNG スライサー
NIfTIボリュームをスライス分解し、W/L適用済みRGB PNGとmanifestを生成する。
iPadが普通の画像として扱える形式に変換する。
"""
import os
import json
import glob
from datetime import datetime

import numpy as np
import nibabel as nib
from PIL import Image


def apply_window_level(data: np.ndarray, center: float, width: float) -> np.ndarray:
    """W/L適用して0-255にスケーリング"""
    lower = center - width / 2
    upper = center + width / 2
    out = np.clip(data, lower, upper)
    out = (out - lower) / (upper - lower) * 255.0
    return out.astype(np.uint8)


def auto_window_level(data: np.ndarray) -> tuple[float, float]:
    """ボリューム全体からW/Lを自動推定（2nd-98th percentile）"""
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return float(data.max() / 2), float(data.max())
    p2, p98 = np.percentile(nonzero, [2, 98])
    center = (p2 + p98) / 2
    width = p98 - p2
    return float(center), float(width)


def slice_nifti_to_png(
    nifti_path: str,
    output_dir: str,
    label_config: dict | None = None,
    window_center: float | None = None,
    window_width: float | None = None,
    slice_axis: int = 2,
) -> dict:
    """NIfTIボリュームをスライスPNGに分解する。

    Args:
        nifti_path: 入力NIfTIファイルパス
        output_dir: 出力ディレクトリ（images/, annotations/, manifestを生成）
        label_config: クラス定義dict。Noneの場合デフォルト生成
        window_center: W/Lのcenter。Noneで自動推定
        window_width: W/Lのwidth。Noneで自動推定
        slice_axis: スライス方向の軸 (0=sagittal, 1=axial, 2=coronal)

    Returns:
        生成されたslice_manifest dict
    """
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    header = img.header
    voxel_sizes = [float(v) for v in header.get_zooms()]

    # W/L決定
    if window_center is None or window_width is None:
        wc, ww = auto_window_level(data)
    else:
        wc, ww = window_center, window_width

    # ディレクトリ作成
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    n_slices = data.shape[slice_axis]
    slice_info = []

    for i in range(n_slices):
        # スライス抽出
        if slice_axis == 0:
            sl = data[i, :, :]
        elif slice_axis == 1:
            sl = data[:, i, :]
        else:
            sl = data[:, :, i]

        # W/L適用
        sl_8bit = apply_window_level(sl, wc, ww)

        # 90度右回転（DICOMの向きを正立に補正）
        sl_8bit = np.rot90(sl_8bit, k=-1)

        # グレースケールPNG保存（1ch、学習時もgrayscaleで使用）
        pil_img = Image.fromarray(sl_8bit, mode="L")

        filename = f"slice_{i:02d}.png"
        pil_img.save(os.path.join(images_dir, filename))

        slice_info.append({
            "filename": filename,
            "slice_index": i,
            "original_shape": list(sl.shape),
        })

    # slice_manifest.json 生成
    manifest = {
        "volume_id": os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0],
        "nifti_path": os.path.abspath(nifti_path),
        "n_slices": n_slices,
        "slice_axis": slice_axis,
        "slice_axis_name": ["sagittal", "axial", "coronal"][slice_axis],
        "volume_shape": list(data.shape),
        "voxel_sizes": voxel_sizes,
        "affine": affine.tolist(),
        "window_center": wc,
        "window_width": ww,
        "preprocessing": [{"op": "rotate", "degrees": -90, "description": "90度右回転（DICOM正立補正）"}],
        "slices": slice_info,
        "created_at": datetime.now().isoformat(),
    }

    manifest_path = os.path.join(output_dir, "slice_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # label_config.json
    if label_config is None:
        label_config = {
            "classes": [
                {"id": 1, "name": "SR", "color": [255, 0, 0]},
                {"id": 2, "name": "LR", "color": [255, 128, 0]},
                {"id": 3, "name": "MR", "color": [255, 255, 0]},
                {"id": 4, "name": "IR", "color": [0, 255, 0]},
                {"id": 5, "name": "ON", "color": [0, 255, 255]},
                {"id": 6, "name": "FAT", "color": [0, 128, 255]},
                {"id": 7, "name": "LG", "color": [0, 0, 255]},
                {"id": 8, "name": "SO", "color": [128, 0, 255]},
                {"id": 9, "name": "EB", "color": [255, 0, 255]},
            ]
        }

    config_path = os.path.join(output_dir, "label_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(label_config, f, indent=2, ensure_ascii=False)

    return manifest


def slice_all_volumes(
    nifti_base_dir: str,
    slices_base_dir: str,
    label_config: dict | None = None,
) -> list[dict]:
    """nifti/raw/ 配下の全ボリュームを一括でスライスPNGに変換する。

    Returns:
        各ボリュームのmanifestリスト
    """
    nifti_files = sorted(glob.glob(
        os.path.join(nifti_base_dir, "**", "*.nii.gz"), recursive=True
    ))

    print(f"対象NIfTIファイル: {len(nifti_files)}")
    manifests = []

    for nf in nifti_files:
        # 出力ディレクトリ名: sub-001_date-20150324 のような形式
        basename = os.path.splitext(os.path.splitext(os.path.basename(nf))[0])[0]
        # _T2w を除去してフォルダ名に
        folder_name = basename.replace("_T2w", "")
        output_dir = os.path.join(slices_base_dir, folder_name)

        print(f"\n処理中: {basename}")
        print(f"  出力: {output_dir}")

        manifest = slice_nifti_to_png(
            nifti_path=nf,
            output_dir=output_dir,
            label_config=label_config,
            slice_axis=2,  # coronal
        )

        print(f"  スライス数: {manifest['n_slices']}")
        print(f"  W/L: center={manifest['window_center']:.0f}, width={manifest['window_width']:.0f}")
        manifests.append(manifest)

    print(f"\n=== 完了: {len(manifests)} ボリューム ===")
    return manifests


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nifti_dir = os.path.join(base, "data", "nifti", "raw")
    slices_dir = os.path.join(base, "data", "slices")

    slice_all_volumes(nifti_dir, slices_dir)
