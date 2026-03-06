"""
モデルバージョン管理
学習実行ごとにバージョンディレクトリを作成し、
foldモデル・best.pt・training_log.json を保全する。
"""
import os
import re
import json
import shutil
import logging
from datetime import datetime

from config import (
    VERSIONS_DIR, CURRENT_PT_DIR, BEST_MODEL_PATH,
    ENCODER_NAME, IMAGE_SIZE, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, N_FOLDS,
    get_fold_model_path,
)

logger = logging.getLogger(__name__)


def get_next_version() -> str:
    """versions/ 内の既存 v{NNN} をスキャンし次番号を返す"""
    existing = []
    if os.path.isdir(VERSIONS_DIR):
        for name in os.listdir(VERSIONS_DIR):
            m = re.match(r"^v(\d{3,})$", name)
            if m:
                existing.append(int(m.group(1)))
    next_num = max(existing, default=0) + 1
    return f"v{next_num:03d}"


def create_version_log(
    version: str,
    training_pairs: list[tuple[str, str]],
    max_epochs: int,
    total_image_count: int = 0,
    unannotated_count: int = 0,
) -> dict:
    """メタデータ初期化（ハイパラ・データセット統計）"""
    return {
        "version": version,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "status": "running",
        "hyperparameters": {
            "encoder_name": ENCODER_NAME,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs_per_fold": max_epochs,
            "n_folds": N_FOLDS,
        },
        "dataset": {
            "total_pairs": total_image_count,
            "completed_pairs": len(training_pairs),
            "unannotated_pairs": unannotated_count,
        },
        "results": None,
        "folds": [
            {
                "fold_idx": i,
                "train_count": 0,
                "val_count": 0,
                "best_dice": 0.0,
                "best_epoch": 0,
                "epochs": [],
            }
            for i in range(N_FOLDS)
        ],
    }


def set_fold_split_info(log: dict, fold_idx: int, train_count: int, val_count: int) -> None:
    """fold の train/val 分割数をセット"""
    log["folds"][fold_idx]["train_count"] = train_count
    log["folds"][fold_idx]["val_count"] = val_count


def record_epoch(
    log: dict,
    fold_idx: int,
    epoch: int,
    train_loss: float,
    val_dice: float,
) -> None:
    """per-epoch メトリクスを log dict に追記"""
    fold = log["folds"][fold_idx]
    fold["epochs"].append({
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_dice": round(val_dice, 6),
    })
    if val_dice > fold["best_dice"]:
        fold["best_dice"] = round(val_dice, 6)
        fold["best_epoch"] = epoch


def finalize_version(
    version: str,
    log: dict,
    fold_dices: list[float],
    best_fold_idx: int,
    status: str = "completed",
) -> str:
    """バージョンディレクトリ作成、foldモデル+best.ptコピー、JSON書き出し。
    Returns: バージョンディレクトリのパス。
    """
    version_dir = os.path.join(VERSIONS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    # fold モデルをコピー
    for i in range(N_FOLDS):
        src = get_fold_model_path(i)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(version_dir, f"fold_{i}.pt"))

    # best.pt をコピー
    if os.path.exists(BEST_MODEL_PATH):
        shutil.copy2(BEST_MODEL_PATH, os.path.join(version_dir, "best.pt"))

    # results セクション更新
    mean_dice = sum(fold_dices) / len(fold_dices) if fold_dices else 0.0
    log["results"] = {
        "mean_dice": round(mean_dice, 6),
        "fold_dices": [round(d, 6) for d in fold_dices],
        "best_fold_idx": best_fold_idx,
        "best_fold_dice": round(fold_dices[best_fold_idx], 6) if fold_dices else 0.0,
    }
    log["status"] = status
    log["completed_at"] = datetime.now().isoformat()

    # JSON 書き出し
    log_path = os.path.join(version_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    logger.info(f"バージョン {version} 保存完了: {version_dir}")
    return version_dir


def restore_version(version: str) -> None:
    """指定バージョンのモデルを current_pt/ に復元する。

    Raises:
        FileNotFoundError: バージョンディレクトリが存在しない場合
    """
    version_dir = os.path.join(VERSIONS_DIR, version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"バージョン {version} が見つかりません: {version_dir}")

    # fold モデルを復元
    for i in range(N_FOLDS):
        src = os.path.join(version_dir, f"fold_{i}.pt")
        if os.path.exists(src):
            shutil.copy2(src, get_fold_model_path(i))

    # best.pt を復元
    src_best = os.path.join(version_dir, "best.pt")
    if os.path.exists(src_best):
        shutil.copy2(src_best, BEST_MODEL_PATH)

    # 推論モデルをリロード
    from inference import reload_model
    reload_model()

    logger.info(f"バージョン {version} を復元しました")


def list_all_versions() -> list[dict]:
    """全バージョンのサマリーリスト（APIエンドポイント用）"""
    versions = []
    if not os.path.isdir(VERSIONS_DIR):
        return versions

    for name in sorted(os.listdir(VERSIONS_DIR)):
        m = re.match(r"^v(\d{3,})$", name)
        if not m:
            continue
        version_dir = os.path.join(VERSIONS_DIR, name)
        log_path = os.path.join(version_dir, "training_log.json")
        if not os.path.isfile(log_path):
            versions.append({"version": name, "status": "unknown"})
            continue
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f)
            summary = {
                "version": name,
                "status": log.get("status", "unknown"),
                "started_at": log.get("started_at"),
                "completed_at": log.get("completed_at"),
            }
            results = log.get("results")
            if results:
                summary["mean_dice"] = results.get("mean_dice")
                summary["best_fold_dice"] = results.get("best_fold_dice")
                summary["best_fold_idx"] = results.get("best_fold_idx")
            dataset = log.get("dataset")
            if dataset:
                summary["completed_pairs"] = dataset.get("completed_pairs")
            versions.append(summary)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"バージョン {name} のログ読み込みエラー: {e}")
            versions.append({"version": name, "status": "error"})

    return versions
