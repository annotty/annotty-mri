"""
テスト用のダミー眼底画像とマスクを生成
サーバーの動作確認に使用
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR


def generate_dummy_fundus(output_dir, n=10):
    """n枚のダミー眼底画像(512x512)を生成"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        # 暗い赤~茶色の背景（眼底画像っぽく）
        arr = np.random.randint(30, 80, (512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + 50, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        # 円形の明るい領域（視神経乳頭っぽく）
        draw = ImageDraw.Draw(img)
        cx = 256 + np.random.randint(-50, 50)
        cy = 256 + np.random.randint(-50, 50)
        draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(200, 180, 100))
        img.save(os.path.join(output_dir, f"img_{i + 1:03d}.png"))
    print(f"Generated {n} dummy images in {output_dir}")


def generate_dummy_labels(output_dir, n=3):
    """n枚のダミー赤色マスクを生成（最初のn枚分）"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        # 赤い線（血管っぽく）
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        img = Image.fromarray(rgba, "RGBA")
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x1, y1 = np.random.randint(0, 512, 2)
            x2, y2 = np.random.randint(0, 512, 2)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0, 255), width=3)
        img.save(os.path.join(output_dir, f"img_{i + 1:03d}.png"))
    print(f"Generated {n} dummy labels in {output_dir}")


if __name__ == "__main__":
    generate_dummy_fundus(UNANNOTATED_IMAGES_DIR, n=10)
    generate_dummy_labels(UNANNOTATED_ANNOTATIONS_DIR, n=3)
    print("Done! Server can now be started for testing.")
