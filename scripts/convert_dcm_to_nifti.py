"""DICOM → NIfTI変換スクリプト
Coronal T2 FSE/FRFSE系列のみを対象に、患者ごとにBIDS風ディレクトリへ変換。
SimpleITKのImageSeriesReaderでDICOM→NIfTI変換（spacing/orientation保持）。
"""
import os
import json
import glob
from collections import defaultdict
from datetime import datetime

import pydicom
import SimpleITK as sitk
import numpy as np

DCM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "T2_Coronal")
OUT_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "data", "nifti", "raw")
os.makedirs(OUT_BASE, exist_ok=True)

# --- 全DCMファイル読み込み & シリーズごとに分類 ---
dcm_files = sorted(glob.glob(os.path.join(DCM_DIR, "*.dcm")))
print(f"総DCMファイル数: {len(dcm_files)}")

series_map = defaultdict(list)  # key: SeriesInstanceUID, value: list of file paths
series_meta = {}  # key: SeriesInstanceUID, value: metadata dict

for f in dcm_files:
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    suid = str(ds.SeriesInstanceUID)
    series_map[suid].append(f)
    if suid not in series_meta:
        series_meta[suid] = {
            "patient_id": str(getattr(ds, "PatientID", "unknown")),
            "study_date": str(getattr(ds, "StudyDate", "unknown")),
            "series_description": str(getattr(ds, "SeriesDescription", "unknown")),
            "protocol_name": str(getattr(ds, "ProtocolName", "N/A")),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
            "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [0, 0])],
            "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
            "spacing_between": float(getattr(ds, "SpacingBetweenSlices", 0)),
        }

# --- 3-pl T2* SSFSE を除外 ---
target_series = {}
excluded = []
for suid, meta in series_meta.items():
    desc = meta["series_description"].lower()
    if "ssfse" in desc or "3-pl" in desc:
        excluded.append(f"  EXCLUDED: {meta['patient_id']} / {meta['series_description']} ({len(series_map[suid])} slices)")
        continue
    target_series[suid] = meta

print(f"\n対象シリーズ: {len(target_series)}")
print(f"除外シリーズ: {len(excluded)}")
for e in excluded:
    print(e)

# --- 患者IDから連番への対応表を作成 ---
patient_ids = sorted(set(m["patient_id"] for m in target_series.values()))
pid_to_sub = {pid: f"sub-{i+1:03d}" for i, pid in enumerate(patient_ids)}
print(f"\n患者ID → サブジェクト番号:")
for pid, sub in pid_to_sub.items():
    short_pid = pid.split("^")[-1] if "^" in pid else pid
    print(f"  {short_pid} → {sub}")

# --- SimpleITKでDICOM→NIfTI変換 ---
conversion_log = []

for suid, meta in sorted(target_series.items(), key=lambda x: (x[1]["patient_id"], x[1]["study_date"])):
    pid = meta["patient_id"]
    sub = pid_to_sub[pid]
    study_date = meta["study_date"]
    desc = meta["series_description"].replace(" ", "_").replace("/", "-")
    files = series_map[suid]

    # 同一患者で複数シリーズがある場合、日付で区別
    sub_dir = os.path.join(OUT_BASE, sub)
    os.makedirs(sub_dir, exist_ok=True)

    # ファイル名: sub-XXX_date-YYYYMMDD_T2w.nii.gz
    nifti_name = f"{sub}_date-{study_date}_T2w.nii.gz"
    nifti_path = os.path.join(sub_dir, nifti_name)

    print(f"\n変換中: {sub} / {meta['series_description']} ({study_date}) - {len(files)} slices")

    try:
        # SimpleITK ImageSeriesReader
        reader = sitk.ImageSeriesReader()

        # DICOMファイルをInstanceNumberでソートするためにSimpleITKに任せる
        # ただし同一シリーズのファイルのみを渡す
        # SimpleITKのGetGDCMSeriesFileNamesはディレクトリ単位なので、
        # 手動でソートしてファイルリストを渡す
        dcm_slices = []
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            pos = [float(x) for x in getattr(ds, "ImagePositionPatient", [0, 0, 0])]
            instance_num = int(getattr(ds, "InstanceNumber", 0))
            dcm_slices.append((f, pos, instance_num))

        # ImagePositionPatientのスライス方向でソート
        # Coronalの場合、通常はY軸（AP方向）でソート
        # ImageOrientationPatientからスライス法線を計算
        ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
        iop = [float(x) for x in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        row_dir = np.array(iop[:3])
        col_dir = np.array(iop[3:])
        slice_dir = np.cross(row_dir, col_dir)

        # スライス方向への射影でソート
        dcm_slices.sort(key=lambda x: np.dot(x[1], slice_dir))
        sorted_files = [s[0] for s in dcm_slices]

        reader.SetFileNames(sorted_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        print(f"  Size: {image.GetSize()}")
        print(f"  Spacing: {[round(s, 4) for s in image.GetSpacing()]}")
        print(f"  Origin: {[round(o, 2) for o in image.GetOrigin()]}")

        sitk.WriteImage(image, nifti_path)
        print(f"  保存: {nifti_path}")

        # ログ記録
        conversion_log.append({
            "subject": sub,
            "original_patient_id": pid,
            "study_date": study_date,
            "series_description": meta["series_description"],
            "protocol_name": meta["protocol_name"],
            "n_slices": len(files),
            "nifti_file": nifti_name,
            "image_size": list(image.GetSize()),
            "spacing": [round(s, 4) for s in image.GetSpacing()],
            "origin": [round(o, 2) for o in image.GetOrigin()],
            "pixel_spacing": meta["pixel_spacing"],
            "slice_thickness": meta["slice_thickness"],
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        conversion_log.append({
            "subject": sub,
            "original_patient_id": pid,
            "study_date": study_date,
            "series_description": meta["series_description"],
            "error": str(e),
        })

# --- 変換ログを保存 ---
log_path = os.path.join(OUT_BASE, "conversion_log.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump({
        "created_at": datetime.now().isoformat(),
        "source_dir": DCM_DIR,
        "total_series_converted": len([l for l in conversion_log if "error" not in l]),
        "patient_id_mapping": pid_to_sub,
        "series": conversion_log,
    }, f, indent=2, ensure_ascii=False)

print(f"\n=== 変換完了 ===")
print(f"変換成功: {len([l for l in conversion_log if 'error' not in l])}/{len(conversion_log)}")
print(f"変換ログ: {log_path}")
