"""
指定フォルダの画像を data/unannotated/images/ にコピー + 512×512リサイズ
元データが様々なサイズの場合に使用

Usage:
  python scripts/import_images.py <source_folder>
  python scripts/import_images.py <source_folder> --size 1024
  python scripts/import_images.py <source_folder> --mask --target data/unannotated/annotations
  python scripts/import_images.py <source_folder> --target data/completed/images
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from config import UNANNOTATED_IMAGES_DIR, IMAGE_SIZE


def import_images(source_dir, target_dir=None, size=None, is_mask=False):
    if target_dir is None:
        target_dir = UNANNOTATED_IMAGES_DIR
    if size is None:
        size = IMAGE_SIZE

    os.makedirs(target_dir, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(source_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            img = Image.open(os.path.join(source_dir, f))
            if is_mask:
                # マスクは元のモード(L/RGBA等)を保持、NEAREST補間
                img = img.resize((size, size), Image.NEAREST)
            else:
                img = img.convert("RGB")
                img = img.resize((size, size), Image.LANCZOS)
            out_name = os.path.splitext(f)[0] + ".png"
            img.save(os.path.join(target_dir, out_name))
            count += 1
    label = "マスク" if is_mask else "画像"
    print(f"{count}枚の{label}を {target_dir} にインポートしました (サイズ: {size}x{size})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_images.py <source_folder> [--size N]")
        sys.exit(1)

    source = sys.argv[1]
    size = IMAGE_SIZE
    target = None
    is_mask = "--mask" in sys.argv
    if "--size" in sys.argv:
        idx = sys.argv.index("--size")
        size = int(sys.argv[idx + 1])
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        target = sys.argv[idx + 1]

    import_images(source, target_dir=target, size=size, is_mask=is_mask)
