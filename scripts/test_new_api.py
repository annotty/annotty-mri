"""新API動作確認テスト（サーバーなしでDataManager直接テスト + reconstruct検証）"""
import os
import sys
import json
import numpy as np
from PIL import Image

# server/ をパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "server"))

from data_manager import DataManager
from config import SLICES_DIR

dm = DataManager()

print("=== 1. 症例一覧 ===")
cases = dm.list_cases()
print(f"症例数: {len(cases)}")
for c in cases:
    print(f"  {c['case_id']}: {c['total_slices']} slices (labeled={c['labeled_slices']})")

print("\n=== 2. 画像一覧（最初の症例） ===")
if cases:
    case_id = cases[0]["case_id"]
    images = dm.list_images(case_id)
    print(f"  {case_id}: {len(images)} images")
    for img in images[:3]:
        print(f"    {img['id']} (has_label={img['has_label']})")
    if len(images) > 3:
        print(f"    ... ({len(images) - 3} more)")

print("\n=== 3. 画像パス取得 ===")
if cases and images:
    path = dm.get_image_path(case_id, images[0]["id"])
    print(f"  path: {path}")
    print(f"  exists: {path is not None and os.path.exists(path)}")

print("\n=== 4. 次の未ラベルスライス ===")
if cases:
    next_img = dm.get_next_unlabeled(case_id)
    print(f"  next: {next_img}")

print("\n=== 5. manifest / label_config ===")
if cases:
    manifest = dm.get_manifest(case_id)
    print(f"  volume_id: {manifest['volume_id']}")
    print(f"  n_slices: {manifest['n_slices']}")
    print(f"  volume_shape: {manifest['volume_shape']}")

    config = dm.get_label_config(case_id)
    print(f"  classes: {[c['name'] for c in config['classes']]}")

print("\n=== 6. 統計 ===")
stats = dm.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n=== 7. ダミーアノテーション → reconstruct テスト ===")
if cases:
    # 最初の症例の中央3スライスにダミーマスクを作成
    test_case = cases[0]["case_id"]
    manifest = dm.get_manifest(test_case)
    label_config = dm.get_label_config(test_case)
    n = manifest["n_slices"]
    mid = n // 2
    test_slices = [mid - 1, mid, mid + 1]

    annotations_dir = os.path.join(SLICES_DIR, test_case, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # ダミーマスク: 中央に赤い円（class 1 = lateral_rectus）
    for si in test_slices:
        shape = manifest["slices"][si]["original_shape"]
        h, w = shape
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        # 中央に赤い円を描画
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        circle = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        mask[circle] = [255, 0, 0, 255]  # 赤=class 1 (lateral_rectus)
        # 少しずらして緑の円（class 4 = inferior_rectus）
        circle2 = (x - cx - r * 2) ** 2 + (y - cy) ** 2 <= r ** 2
        mask[circle2] = [0, 255, 0, 255]  # 緑=class 4

        fname = manifest["slices"][si]["filename"]
        Image.fromarray(mask, "RGBA").save(os.path.join(annotations_dir, fname))
        print(f"  ダミーマスク作成: {fname}")

    # reconstruct実行
    from medical_adapter.reconstructor import reconstruct_label_volume
    result = reconstruct_label_volume(test_case)
    print(f"\n  再統合結果:")
    print(f"    output: {result['output_path']}")
    print(f"    annotated: {result['annotated_slices']}/{result['total_slices']} slices")

    # NIfTI検証
    import nibabel as nib
    nii = nib.load(result["output_path"])
    label_data = nii.get_fdata()
    unique_labels = np.unique(label_data)
    print(f"    shape: {label_data.shape}")
    print(f"    unique labels: {unique_labels}")
    print(f"    affine matches: {np.allclose(nii.affine, np.array(manifest['affine']))}")

    # sidecar確認
    with open(result["sidecar_path"], "r") as f:
        sidecar = json.load(f)
    print(f"    sidecar labels: {sidecar['labels']}")

    # ダミーマスク削除
    for si in test_slices:
        fname = manifest["slices"][si]["filename"]
        os.remove(os.path.join(annotations_dir, fname))
    print("\n  ダミーマスク削除完了")

print("\n=== 全テスト完了 ===")
