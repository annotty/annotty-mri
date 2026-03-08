"""DICOM画像の確認スクリプト: 複数患者/シリーズ対応"""
import os
import glob
from collections import defaultdict
import pydicom
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DCM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "T2_Coronal")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "output")
os.makedirs(OUT_DIR, exist_ok=True)

# --- 全DCMファイル読み込み ---
dcm_files = sorted(glob.glob(os.path.join(DCM_DIR, "*.dcm")))
print(f"DCMファイル数: {len(dcm_files)}")

# --- 患者/シリーズごとに分類 ---
series_map = defaultdict(list)  # key: (PatientID, SeriesInstanceUID)
for f in dcm_files:
    ds = pydicom.dcmread(f)
    pid = getattr(ds, "PatientID", "unknown")
    series_uid = getattr(ds, "SeriesInstanceUID", "unknown")
    series_map[(pid, series_uid)].append(ds)

print(f"\n=== データ構成 ===")
print(f"ユニーク患者数: {len(set(k[0] for k in series_map))}")
print(f"ユニークシリーズ数: {len(series_map)}")

# --- 各シリーズの情報を表示 ---
for i, ((pid, suid), ds_list) in enumerate(sorted(series_map.items())):
    # Instance Numberでソート
    ds_list.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    ds0 = ds_list[0]
    shapes = set()
    for ds in ds_list:
        shapes.add(ds.pixel_array.shape)

    print(f"\n--- Series {i+1}/{len(series_map)} ---")
    print(f"  PatientID:       {pid}")
    print(f"  StudyDate:       {getattr(ds0, 'StudyDate', 'N/A')}")
    print(f"  SeriesDesc:      {getattr(ds0, 'SeriesDescription', 'N/A')}")
    print(f"  Protocol:        {getattr(ds0, 'ProtocolName', 'N/A')}")
    print(f"  Slices:          {len(ds_list)}")
    print(f"  Image shapes:    {shapes}")
    print(f"  PixelSpacing:    {getattr(ds0, 'PixelSpacing', 'N/A')}")
    print(f"  SliceThickness:  {getattr(ds0, 'SliceThickness', 'N/A')}")
    print(f"  SpacingBetween:  {getattr(ds0, 'SpacingBetweenSlices', 'N/A')}")
    print(f"  WindowCenter:    {getattr(ds0, 'WindowCenter', 'N/A')}")
    print(f"  WindowWidth:     {getattr(ds0, 'WindowWidth', 'N/A')}")
    print(f"  BitsStored:      {getattr(ds0, 'BitsStored', 'N/A')}")

# --- 各患者の代表スライス（中央付近）を1枚ずつ並べた一覧画像 ---
def apply_wl(img, center, width):
    lower = center - width / 2
    upper = center + width / 2
    out = np.clip(img, lower, upper)
    out = (out - lower) / (upper - lower) * 255
    return out.astype(np.uint8)

def get_wl(ds):
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return None, None
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    else:
        wc = float(wc)
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    else:
        ww = float(ww)
    return wc, ww

# 各シリーズから中央スライスを3枚選んで表示
n_series = len(series_map)
fig, axes = plt.subplots(n_series, 3, figsize=(9, n_series * 3))
if n_series == 1:
    axes = axes[np.newaxis, :]

for row, ((pid, suid), ds_list) in enumerate(sorted(series_map.items())):
    ds_list.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    n = len(ds_list)
    # 3枚: 前方、中央、後方
    indices = [n // 4, n // 2, 3 * n // 4]

    for col, idx in enumerate(indices):
        idx = min(idx, n - 1)
        ds = ds_list[idx]
        arr = ds.pixel_array.astype(np.float32)
        wc, ww = get_wl(ds)
        if wc is not None:
            img = apply_wl(arr, wc, ww)
        else:
            img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

        axes[row, col].imshow(img, cmap="gray")
        if col == 0:
            short_pid = pid.split("^")[-1] if "^" in pid else pid
            axes[row, col].set_ylabel(f"{short_pid}\n{n}slices", fontsize=7)
        axes[row, col].set_title(f"Slice {idx+1}/{n}", fontsize=7)
        axes[row, col].axis("off")

plt.suptitle(f"T2 Coronal - {n_series} series, {len(dcm_files)} total slices", fontsize=11)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "dicom_overview.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n一覧画像を保存: {out_path}")
