"""マスク再統合モジュール
iPadから返却されたインデックスカラーPNGマスクを
NIfTIラベルマップ (_dseg.nii.gz) に再統合する。

フロー:
  slices/{case_id}/annotations/slice_XX.png
    + slice_manifest.json (affine, volume_shape, slice_axis)
  → nifti/labels/{sub-XXX}/{sub-XXX}_date-YYYYMMDD_T2w_dseg.nii.gz
    + _dseg.json (sidecar metadata)
"""
import os
import json
import logging
from datetime import datetime

import numpy as np
import nibabel as nib
from PIL import Image

from config import SLICES_DIR, NIFTI_LABELS_DIR

logger = logging.getLogger(__name__)


def _parse_index_mask(mask_path: str, label_config: dict | None = None) -> np.ndarray:
    """マスクPNGを読み込み、インデックスマップ（0=背景, 1-N=クラス）に変換する。

    iPadからの返却マスクは色付きRGBA。label_configの色定義に基づいて
    各ピクセルを最も近いクラスIDにマッピングする。
    """
    img = Image.open(mask_path)
    arr = np.array(img)

    # グレースケール or パレットモードならそのままインデックスとして扱う
    if img.mode == "L":
        return arr.astype(np.uint8)
    if img.mode == "P":
        return arr.astype(np.uint8)

    # RGBA/RGB カラーマスクの場合 → 色からクラスIDに変換
    if img.mode == "RGBA":
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
    elif img.mode == "RGB":
        rgb = arr
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
    else:
        rgb = np.array(img.convert("RGB"))
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    index_map = np.zeros(rgb.shape[:2], dtype=np.uint8)

    if label_config is None or "classes" not in label_config:
        logger.warning("label_config なし: カラーマスクをグレースケール変換")
        gray = np.mean(rgb, axis=2)
        return (gray > 128).astype(np.uint8)

    # 各クラスの色をマッチング（透明ピクセルは背景=0）
    for cls in label_config["classes"]:
        cls_id = cls["id"]
        cls_color = np.array(cls["color"], dtype=np.uint8)  # [R, G, B]

        # 色の距離が閾値以内のピクセルをそのクラスに割り当て
        dist = np.sqrt(np.sum((rgb.astype(np.float32) - cls_color.astype(np.float32)) ** 2, axis=2))
        mask = (dist < 80) & (alpha > 128)  # 色距離80以内 & 不透明
        index_map[mask] = cls_id

    return index_map


def reconstruct_label_volume(case_id: str) -> dict:
    """症例のアノテーションPNGをNIfTIラベルマップに再統合する。

    Args:
        case_id: 症例ID（slices/配下のフォルダ名）

    Returns:
        結果情報のdict（output_path, annotated_slices等）
    """
    case_dir = os.path.join(SLICES_DIR, case_id)
    annotations_dir = os.path.join(case_dir, "annotations")
    manifest_path = os.path.join(case_dir, "slice_manifest.json")
    label_config_path = os.path.join(case_dir, "label_config.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"slice_manifest.json not found for case '{case_id}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    label_config = None
    if os.path.exists(label_config_path):
        with open(label_config_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)

    # 元ボリュームの情報
    volume_shape = manifest["volume_shape"]
    affine = np.array(manifest["affine"])
    slice_axis = manifest["slice_axis"]
    n_slices = manifest["n_slices"]

    # 空のラベルボリューム作成
    label_volume = np.zeros(volume_shape, dtype=np.uint8)

    # 各アノテーション済みスライスを再統合
    annotated_slices = []
    for slice_info in manifest["slices"]:
        filename = slice_info["filename"]
        slice_idx = slice_info["slice_index"]
        mask_path = os.path.join(annotations_dir, filename)

        if not os.path.exists(mask_path):
            continue

        index_map = _parse_index_mask(mask_path, label_config)

        # サイズが異なる場合はリサイズ（nearest neighbor で劣化防止）
        if slice_axis == 2:
            target_shape = (volume_shape[0], volume_shape[1])
        elif slice_axis == 1:
            target_shape = (volume_shape[0], volume_shape[2])
        else:
            target_shape = (volume_shape[1], volume_shape[2])

        if index_map.shape != target_shape:
            from PIL import Image as PILImage
            resized = PILImage.fromarray(index_map).resize(
                (target_shape[1], target_shape[0]),
                PILImage.NEAREST,
            )
            index_map = np.array(resized)

        # ボリュームに挿入
        if slice_axis == 0:
            label_volume[slice_idx, :, :] = index_map
        elif slice_axis == 1:
            label_volume[:, slice_idx, :] = index_map
        else:
            label_volume[:, :, slice_idx] = index_map

        annotated_slices.append(slice_idx)

    if not annotated_slices:
        raise ValueError(f"No annotations found for case '{case_id}'")

    # NIfTI保存（元ボリュームと同じaffineを使用）
    volume_id = manifest["volume_id"]
    # sub-001_date-20150324_T2w → sub-001 を抽出
    parts = volume_id.split("_date-")
    subject = parts[0] if parts else case_id
    dseg_name = volume_id.replace("_T2w", "_T2w_dseg")

    out_dir = os.path.join(NIFTI_LABELS_DIR, subject)
    os.makedirs(out_dir, exist_ok=True)

    nifti_path = os.path.join(out_dir, f"{dseg_name}.nii.gz")
    nii = nib.Nifti1Image(label_volume, affine)
    nib.save(nii, nifti_path)

    # JSON sidecar
    sidecar = {
        "labels": {"0": "background"},
        "source_volume": manifest.get("nifti_path", ""),
        "annotation_tool": "Annotty-MRI/iPad",
        "slices_annotated": sorted(annotated_slices),
        "total_slices": n_slices,
        "created_at": datetime.now().isoformat(),
    }
    if label_config and "classes" in label_config:
        for cls in label_config["classes"]:
            sidecar["labels"][str(cls["id"])] = cls["name"]

    sidecar_path = os.path.join(out_dir, f"{dseg_name}.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    logger.info(
        f"NIfTI再統合: {case_id} → {nifti_path} "
        f"(annotated {len(annotated_slices)}/{n_slices} slices)"
    )

    return {
        "status": "reconstructed",
        "case_id": case_id,
        "output_path": nifti_path,
        "sidecar_path": sidecar_path,
        "annotated_slices": len(annotated_slices),
        "total_slices": n_slices,
        "volume_shape": volume_shape,
    }
