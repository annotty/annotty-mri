"""
データアクセス層
completed / unannotated の物理分離を隠蔽し、
サーバー・トレーナーに統一インターフェースを提供する。

安全設計:
- サーバーコードは completed/ に一切書き込まない
- save_annotation() は unannotated/images/ に存在するIDのみ受け付ける
"""
import os
import random
import logging

from config import (
    COMPLETED_IMAGES_DIR, COMPLETED_ANNOTATIONS_DIR,
    UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR,
)

logger = logging.getLogger(__name__)


def _list_png(directory: str) -> list[str]:
    """指定ディレクトリ内の .png ファイル名一覧"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(".png")]


class DataManager:
    """completed / unannotated を統合管理するデータアクセス層"""

    # ----- iPad向け（unannotated のみ） -----

    def list_unannotated_images(self) -> list[dict]:
        """unannotated/images/ の画像一覧 + has_label判定"""
        images = sorted(_list_png(UNANNOTATED_IMAGES_DIR))
        label_set = set(_list_png(UNANNOTATED_ANNOTATIONS_DIR))
        return [{"id": img, "has_label": img in label_set} for img in images]

    def get_next_unlabeled(self, strategy: str = "random") -> str | None:
        """unannotated のうちアノテーション未完了の画像IDを返す"""
        images = set(_list_png(UNANNOTATED_IMAGES_DIR))
        labels = set(_list_png(UNANNOTATED_ANNOTATIONS_DIR))
        unlabeled = list(images - labels)

        if not unlabeled:
            return None

        if strategy == "sequential":
            return sorted(unlabeled)[0]
        return random.choice(unlabeled)

    def get_unannotated_image_path(self, image_id: str) -> str | None:
        """unannotated/images/ 内の画像パス（存在チェック付き）"""
        path = os.path.join(UNANNOTATED_IMAGES_DIR, image_id)
        return path if os.path.exists(path) else None

    def get_annotation_path(self, image_id: str) -> str | None:
        """unannotated → completed の順にアノテーションを検索"""
        for d in [UNANNOTATED_ANNOTATIONS_DIR, COMPLETED_ANNOTATIONS_DIR]:
            path = os.path.join(d, image_id)
            if os.path.exists(path):
                return path
        return None

    def save_annotation(self, image_id: str, data: bytes) -> None:
        """unannotated/annotations/ にアノテーションを保存。

        安全チェック: image_id が unannotated/images/ に存在しなければ ValueError。
        これにより completed のデータが上書きされることを防止する。
        """
        image_path = os.path.join(UNANNOTATED_IMAGES_DIR, image_id)
        if not os.path.exists(image_path):
            raise ValueError(
                f"Image '{image_id}' not found in unannotated/images/. "
                "Cannot save annotation for completed or non-existent images."
            )
        save_path = os.path.join(UNANNOTATED_ANNOTATIONS_DIR, image_id)
        with open(save_path, "wb") as f:
            f.write(data)
        logger.info(f"アノテーション保存: {image_id} ({len(data)} bytes)")

    # ----- 推論向け（completed + unannotated 両方検索） -----

    def get_image_path(self, image_id: str) -> str | None:
        """unannotated → completed の順に画像を検索"""
        for d in [UNANNOTATED_IMAGES_DIR, COMPLETED_IMAGES_DIR]:
            path = os.path.join(d, image_id)
            if os.path.exists(path):
                return path
        return None

    # ----- 学習向け -----

    def get_all_training_pairs(self) -> list[tuple[str, str]]:
        """学習用の (image_path, annotation_path) ペアを全て返す。

        - completed: 全ペア（images/ と annotations/ が1:1対応前提）
        - unannotated: annotations/ が存在するもののみ
        """
        pairs = []

        # completed 全ペア
        completed_annotations = set(_list_png(COMPLETED_ANNOTATIONS_DIR))
        for fname in _list_png(COMPLETED_IMAGES_DIR):
            if fname in completed_annotations:
                pairs.append((
                    os.path.join(COMPLETED_IMAGES_DIR, fname),
                    os.path.join(COMPLETED_ANNOTATIONS_DIR, fname),
                ))

        completed_count = len(pairs)

        # unannotated のうちアノテーション済みのみ
        unannotated_annotations = set(_list_png(UNANNOTATED_ANNOTATIONS_DIR))
        for fname in _list_png(UNANNOTATED_IMAGES_DIR):
            if fname in unannotated_annotations:
                pairs.append((
                    os.path.join(UNANNOTATED_IMAGES_DIR, fname),
                    os.path.join(UNANNOTATED_ANNOTATIONS_DIR, fname),
                ))

        unannotated_count = len(pairs) - completed_count
        logger.info(
            f"Training pairs: completed={completed_count}, "
            f"unannotated={unannotated_count}, total={len(pairs)}"
        )
        return pairs

    # ----- 統計 -----

    def get_stats(self) -> dict:
        """completed / unannotated の画像・アノテーション数を返す"""
        completed_images = _list_png(COMPLETED_IMAGES_DIR)
        completed_annotations = _list_png(COMPLETED_ANNOTATIONS_DIR)
        unannotated_images = _list_png(UNANNOTATED_IMAGES_DIR)
        unannotated_annotations = _list_png(UNANNOTATED_ANNOTATIONS_DIR)

        unannotated_set = set(unannotated_images)
        unannotated_labeled = [
            f for f in unannotated_annotations if f in unannotated_set
        ]

        return {
            "completed_images": len(completed_images),
            "completed_annotations": len(completed_annotations),
            "unannotated_images": len(unannotated_images),
            "unannotated_labeled": len(unannotated_labeled),
            "unannotated_unlabeled": len(unannotated_images) - len(unannotated_labeled),
            "total_training_pairs": len(completed_annotations) + len(unannotated_labeled),
        }
