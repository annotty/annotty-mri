"""スライスPNGの確認: 各症例の中央スライスを一覧表示"""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SLICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "data", "slices")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "output")

cases = sorted([d for d in os.listdir(SLICES_DIR) if os.path.isdir(os.path.join(SLICES_DIR, d))])
n = len(cases)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = np.array(axes).flatten()

for i, case in enumerate(cases):
    images_dir = os.path.join(SLICES_DIR, case, "images")
    pngs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mid = len(pngs) // 2
    img = np.array(Image.open(pngs[mid]))
    axes[i].imshow(img)
    axes[i].set_title(f"{case}\nslice {mid}/{len(pngs)}", fontsize=6)
    axes[i].axis("off")

for i in range(n, len(axes)):
    axes[i].axis("off")

plt.suptitle(f"Slice PNG Verification - {n} cases (mid-slice)", fontsize=10)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "slice_png_verification.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"検証画像: {out_path}")
