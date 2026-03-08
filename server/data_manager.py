"""
データアクセス層（MRI症例ベース）
slices/ 配下の症例フォルダを統合管理し、
サーバー・トレーナーに統一インターフェースを提供する。

ディレクトリ構造:
  slices/
    {case_id}/
      images/        ← W/L適用済みPNG（サーバーが生成、read-only）
      annotations/   ← iPadから返却されたマスクPNG
      slice_manifest.json
      label_config.json
"""
import os
import json
import random
import logging

from config import SLICES_DIR

logger = logging.getLogger(__name__)


def _list_png(directory: str) -> list[str]:
    """指定ディレクトリ内の .png ファイル名一覧"""
    if not os.path.exists(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".png"))


class DataManager:
    """slices/ 配下の症例フォルダを統合管理するデータアクセス層"""

    # ----- 症例一覧 -----

    def list_cases(self) -> list[dict]:
        """全症例の一覧を返す"""
        if not os.path.exists(SLICES_DIR):
            return []

        cases = []
        for name in sorted(os.listdir(SLICES_DIR)):
            case_dir = os.path.join(SLICES_DIR, name)
            if not os.path.isdir(case_dir):
                continue

            images_dir = os.path.join(case_dir, "images")
            annotations_dir = os.path.join(case_dir, "annotations")
            images = _list_png(images_dir)
            annotations = set(_list_png(annotations_dir))

            labeled = sum(1 for img in images if img in annotations)
            cases.append({
                "case_id": name,
                "total_slices": len(images),
                "labeled_slices": labeled,
                "unlabeled_slices": len(images) - labeled,
            })
        return cases

    # ----- 症例内の画像操作 -----

    def list_images(self, case_id: str) -> list[dict]:
        """指定症例内の画像一覧 + has_label判定"""
        images_dir = os.path.join(SLICES_DIR, case_id, "images")
        annotations_dir = os.path.join(SLICES_DIR, case_id, "annotations")

        images = _list_png(images_dir)
        label_set = set(_list_png(annotations_dir))
        return [{"id": img, "has_label": img in label_set} for img in images]

    def get_image_path(self, case_id: str, image_id: str) -> str | None:
        """症例内の画像パス（存在チェック付き）"""
        path = os.path.join(SLICES_DIR, case_id, "images", image_id)
        return path if os.path.exists(path) else None

    def get_annotation_path(self, case_id: str, image_id: str) -> str | None:
        """症例内のアノテーションパス（存在チェック付き）"""
        path = os.path.join(SLICES_DIR, case_id, "annotations", image_id)
        return path if os.path.exists(path) else None

    def save_annotation(self, case_id: str, image_id: str, data: bytes) -> None:
        """annotations/ にアノテーションを保存。

        安全チェック: image_id が images/ に存在しなければ ValueError。
        """
        image_path = os.path.join(SLICES_DIR, case_id, "images", image_id)
        if not os.path.exists(image_path):
            raise ValueError(
                f"Image '{image_id}' not found in case '{case_id}'. "
                "Cannot save annotation for non-existent images."
            )
        annotations_dir = os.path.join(SLICES_DIR, case_id, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        save_path = os.path.join(annotations_dir, image_id)
        with open(save_path, "wb") as f:
            f.write(data)
        logger.info(f"アノテーション保存: {case_id}/{image_id} ({len(data)} bytes)")

    def get_next_unlabeled(self, case_id: str, strategy: str = "sequential") -> str | None:
        """症例内の未ラベルスライスを返す"""
        images_dir = os.path.join(SLICES_DIR, case_id, "images")
        annotations_dir = os.path.join(SLICES_DIR, case_id, "annotations")

        images = set(_list_png(images_dir))
        labels = set(_list_png(annotations_dir))
        unlabeled = sorted(images - labels)

        if not unlabeled:
            return None
        if strategy == "random":
            return random.choice(unlabeled)
        return unlabeled[0]

    # ----- メタデータ -----

    def get_manifest(self, case_id: str) -> dict | None:
        """症例のslice_manifest.jsonを読み込む"""
        path = os.path.join(SLICES_DIR, case_id, "slice_manifest.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_label_config(self, case_id: str) -> dict | None:
        """症例のlabel_config.jsonを読み込む"""
        path = os.path.join(SLICES_DIR, case_id, "label_config.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def case_exists(self, case_id: str) -> bool:
        """症例ディレクトリが存在するか"""
        return os.path.isdir(os.path.join(SLICES_DIR, case_id))

    # ----- 学習向け -----

    def get_all_training_pairs(self) -> list[tuple[str, str]]:
        """全症例からアノテーション済みの (image_path, annotation_path) ペアを返す"""
        pairs = []
        for case_info in self.list_cases():
            case_id = case_info["case_id"]
            images_dir = os.path.join(SLICES_DIR, case_id, "images")
            annotations_dir = os.path.join(SLICES_DIR, case_id, "annotations")

            annotations = set(_list_png(annotations_dir))
            for fname in _list_png(images_dir):
                if fname in annotations:
                    pairs.append((
                        os.path.join(images_dir, fname),
                        os.path.join(annotations_dir, fname),
                    ))

        logger.info(f"Training pairs: total={len(pairs)}")
        return pairs

    # ----- 統計 -----

    def get_stats(self) -> dict:
        """全症例の統計情報"""
        cases = self.list_cases()
        total_slices = sum(c["total_slices"] for c in cases)
        total_labeled = sum(c["labeled_slices"] for c in cases)
        return {
            "total_cases": len(cases),
            "total_slices": total_slices,
            "labeled_slices": total_labeled,
            "unlabeled_slices": total_slices - total_labeled,
            "total_training_pairs": total_labeled,
        }
