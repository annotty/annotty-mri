"""
データ移行スクリプト: フラット構成 → images_completed / images_unannotated 分離構成

実行すると:
1. data/images_images_completed/images/, data/images_images_completed/annotations/ を作成
2. data/images_images_unannotated/images/, data/images_images_unannotated/annotations/ を作成
3. data/images/ → data/images_images_completed/images/ にファイル移動
4. data/annotations/ → data/images_images_completed/annotations/ にファイル移動
5. 空になった data/images/, data/annotations/ を削除
6. 検証: images_completed の images と annotations のファイル名が一致することを確認

Usage:
    python scripts/migrate_data.py
    python scripts/migrate_data.py --dry-run   # 実際には移動せず確認のみ
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    COMPLETED_IMAGES_DIR, COMPLETED_ANNOTATIONS_DIR,
    UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR,
)

# 旧パス
OLD_IMAGES_DIR = os.path.join(DATA_DIR, "images")
OLD_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")


def migrate(dry_run: bool = False):
    # 1. 旧ディレクトリの存在チェック
    old_images_exist = os.path.isdir(OLD_IMAGES_DIR)
    old_annotations_exist = os.path.isdir(OLD_ANNOTATIONS_DIR)

    if not old_images_exist and not old_annotations_exist:
        print("旧ディレクトリ (data/images/, data/annotations/) が見つかりません。")
        print("既に移行済みか、ディレクトリが存在しません。")
        return

    # 旧ファイル一覧
    old_images = sorted(os.listdir(OLD_IMAGES_DIR)) if old_images_exist else []
    old_annotations = sorted(os.listdir(OLD_ANNOTATIONS_DIR)) if old_annotations_exist else []

    print(f"旧 data/images/: {len(old_images)} ファイル")
    print(f"旧 data/annotations/: {len(old_annotations)} ファイル")

    if dry_run:
        print("\n[DRY RUN] 以下の操作を実行予定:")
        print(f"  data/images/ → data/images_images_completed/images/ ({len(old_images)} ファイル)")
        print(f"  data/annotations/ → data/images_images_completed/annotations/ ({len(old_annotations)} ファイル)")
        print(f"  data/images_images_unannotated/images/ (空ディレクトリ作成)")
        print(f"  data/images_images_unannotated/annotations/ (空ディレクトリ作成)")
        return

    # 2. 新ディレクトリ作成
    for d in [COMPLETED_IMAGES_DIR, COMPLETED_ANNOTATIONS_DIR,
              UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"ディレクトリ作成: {d}")

    # 3. ファイル移動: images
    if old_images_exist:
        moved = 0
        for f in old_images:
            src = os.path.join(OLD_IMAGES_DIR, f)
            dst = os.path.join(COMPLETED_IMAGES_DIR, f)
            if os.path.isfile(src):
                shutil.move(src, dst)
                moved += 1
        print(f"data/images/ → data/images_images_completed/images/: {moved} ファイル移動完了")

    # 4. ファイル移動: annotations
    if old_annotations_exist:
        moved = 0
        for f in old_annotations:
            src = os.path.join(OLD_ANNOTATIONS_DIR, f)
            dst = os.path.join(COMPLETED_ANNOTATIONS_DIR, f)
            if os.path.isfile(src):
                shutil.move(src, dst)
                moved += 1
        print(f"data/annotations/ → data/images_images_completed/annotations/: {moved} ファイル移動完了")

    # 5. 旧ディレクトリ削除（空の場合のみ）
    for d in [OLD_IMAGES_DIR, OLD_ANNOTATIONS_DIR]:
        if os.path.isdir(d):
            remaining = os.listdir(d)
            if not remaining:
                os.rmdir(d)
                print(f"空ディレクトリ削除: {d}")
            else:
                print(f"警告: {d} にまだ {len(remaining)} ファイルが残っています。手動で確認してください。")

    # 6. 検証
    print("\n=== 検証 ===")
    completed_images = set(os.listdir(COMPLETED_IMAGES_DIR))
    completed_annotations = set(os.listdir(COMPLETED_ANNOTATIONS_DIR))

    print(f"images_completed/images/: {len(completed_images)} ファイル")
    print(f"images_completed/annotations/: {len(completed_annotations)} ファイル")

    # 画像にアノテーションがないもの
    missing_annotations = completed_images - completed_annotations
    if missing_annotations:
        print(f"警告: アノテーションがない画像: {len(missing_annotations)} 枚")
        for f in sorted(missing_annotations)[:5]:
            print(f"  - {f}")

    # アノテーションに画像がないもの
    orphan_annotations = completed_annotations - completed_images
    if orphan_annotations:
        print(f"警告: 画像がないアノテーション: {len(orphan_annotations)} 枚")
        for f in sorted(orphan_annotations)[:5]:
            print(f"  - {f}")

    if not missing_annotations and not orphan_annotations:
        print("OK: images と annotations のファイル名が完全一致")

    unannotated_images = os.listdir(UNANNOTATED_IMAGES_DIR)
    unannotated_annotations = os.listdir(UNANNOTATED_ANNOTATIONS_DIR)
    print(f"images_unannotated/images/: {len(unannotated_images)} ファイル")
    print(f"images_unannotated/annotations/: {len(unannotated_annotations)} ファイル")

    print("\n移行完了!")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)
