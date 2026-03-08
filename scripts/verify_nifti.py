"""変換後NIfTIの確認: 各ボリュームの中央スライスを一覧表示"""
import os
import glob
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIFTI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "data", "nifti", "raw")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "output")
os.makedirs(OUT_DIR, exist_ok=True)

nifti_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "**", "*.nii.gz"), recursive=True))
print(f"NIfTIファイル数: {len(nifti_files)}")

n = len(nifti_files)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = np.array(axes).flatten()

for i, nf in enumerate(nifti_files):
    img = nib.load(nf)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    voxel_sizes = header.get_zooms()

    basename = os.path.basename(nf).replace(".nii.gz", "")
    mid = data.shape[2] // 2
    sl = data[:, :, mid]

    # Auto W/L (2nd-98th percentile)
    p2, p98 = np.percentile(sl[sl > 0], [2, 98]) if np.any(sl > 0) else (0, 1)
    sl_norm = np.clip(sl, p2, p98)
    sl_norm = (sl_norm - p2) / (p98 - p2 + 1e-8)

    axes[i].imshow(sl_norm.T, cmap="gray", origin="lower")
    axes[i].set_title(f"{basename}\n{data.shape} vox={[round(v,2) for v in voxel_sizes]}", fontsize=7)
    axes[i].axis("off")

    print(f"{basename}: shape={data.shape}, voxel={[round(v,2) for v in voxel_sizes]}, "
          f"range=[{data.min():.0f}, {data.max():.0f}]")

for i in range(n, len(axes)):
    axes[i].axis("off")

plt.suptitle(f"NIfTI Verification - {n} volumes (mid-slice)", fontsize=11)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "nifti_verification.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n検証画像を保存: {out_path}")
