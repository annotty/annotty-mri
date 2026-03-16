"""
再学習ワーカー（マルチクラス版）
眼窩MRI 10クラスセグメンテーション用 HITL fine-tuning
入力: grayscale 1ch MRI → インデックスカラーPNGマスク → 5-fold CV
"""
import os
import json
import random
import shutil
import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from model import create_model
from config import (
    IMAGE_SIZE, NUM_CLASSES, IN_CHANNELS, SLICES_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MIN_IMAGES_FOR_TRAINING, N_FOLDS,
    PRETRAINED_PATH, BEST_MODEL_PATH, get_fold_model_path,
)

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    """学習キャンセル時に送出される例外"""
    pass


# =====================================================
# 損失関数: CrossEntropy + Dice Loss（マルチクラス版）
# =====================================================
class MultiClassDiceCELoss(nn.Module):
    """CrossEntropy + Dice Loss の複合損失（マルチクラスセグメンテーション用）

    logits: (B, C, H, W) — モデル出力（raw logits）
    targets: (B, H, W) — クラスインデックス (long, 0..NUM_CLASSES-1)
    """

    def __init__(self, num_classes: int, dice_weight=1.0, ce_weight=1.0, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # CrossEntropy Loss
        ce_loss = F.cross_entropy(logits, targets)

        # Dice Loss (per-class, excluding background=0)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_onehot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dice_sum = 0.0
        count = 0
        for c in range(1, self.num_classes):  # skip background
            pred_c = probs[:, c]
            gt_c = targets_onehot[:, c]
            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice_sum += (2.0 * intersection + self.smooth) / (union + self.smooth)
            count += 1

        dice_loss = 1.0 - dice_sum / max(count, 1)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# =====================================================
# 評価指標: Multi-class Dice Score
# =====================================================
def multiclass_dice(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int, smooth=1.0) -> float:
    """マルチクラスDice係数を計算（背景除外の平均）

    pred: (H, W) — argmax後のクラスインデックス
    target: (H, W) — GTクラスインデックス
    """
    dice_sum = 0.0
    count = 0
    for c in range(1, num_classes):  # skip background
        pred_c = (pred == c).float()
        gt_c = (target == c).float()
        if gt_c.sum() == 0 and pred_c.sum() == 0:
            continue  # クラスが存在しないスライスはスキップ
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice_sum += (2.0 * intersection + smooth) / (union + smooth)
        count += 1
    return dice_sum / max(count, 1)


# =====================================================
# カラーマスク → インデックスマスク変換
# =====================================================
def _load_color_to_index_map(case_id: str) -> dict[tuple[int, int, int], int]:
    """label_config.json からカラー→クラスID マッピングを構築"""
    config_path = os.path.join(SLICES_DIR, case_id, "label_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    mapping = {}
    for cls in config.get("classes", []):
        color = tuple(cls["color"])
        mapping[color] = cls["id"]
    return mapping


def _color_mask_to_index(mask_img: Image.Image, color_map: dict) -> np.ndarray:
    """カラーマスクPNGをクラスインデックス配列に変換

    mask_img: RGBA or RGB カラーマスク画像
    color_map: {(R,G,B): class_id}
    Returns: (H, W) ndarray of int64
    """
    if mask_img.mode == "RGBA":
        rgba = np.array(mask_img)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
    else:
        rgb = np.array(mask_img.convert("RGB"))
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    h, w = rgb.shape[:2]
    index_mask = np.zeros((h, w), dtype=np.int64)

    for color, class_id in color_map.items():
        match = (
            (rgb[:, :, 0] == color[0]) &
            (rgb[:, :, 1] == color[1]) &
            (rgb[:, :, 2] == color[2]) &
            (alpha > 128)
        )
        index_mask[match] = class_id

    return index_mask


# =====================================================
# データセット（マルチクラス版）
# =====================================================
class MRIDataset(Dataset):
    """眼窩MRI + インデックスカラーマスクのデータセット

    pairs: list[tuple[str, str]] — (image_path, annotation_path) のフルパスペア
    入力: grayscale 1ch, 256×256, [0,1]正規化
    マスク: long tensor (H, W), 値 = クラスインデックス 0..NUM_CLASSES-1
    """

    def __init__(self, pairs: list[tuple[str, str]], augment: bool = False):
        self.pairs = pairs
        self.augment = augment
        # カラーマップをキャッシュ（case_id → color_map）
        self._color_maps: dict[str, dict] = {}

    def _get_color_map(self, annotation_path: str) -> dict:
        """アノテーションパスから case_id を推定してカラーマップを取得"""
        # annotation_path: .../slices/{case_id}/annotations/slice_XX.png
        parts = annotation_path.replace("\\", "/").split("/")
        try:
            ann_idx = parts.index("annotations")
            case_id = parts[ann_idx - 1]
        except (ValueError, IndexError):
            case_id = "__unknown__"

        if case_id not in self._color_maps:
            self._color_maps[case_id] = _load_color_to_index_map(case_id)
        return self._color_maps[case_id]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, annotation_path = self.pairs[idx]

        # 画像読み込み: grayscale 1ch, [0,1] 正規化
        img = Image.open(image_path).convert("L")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()  # (1, H, W)

        # マスク読み込み: インデックスカラーPNG → クラスインデックス
        mask_raw = Image.open(annotation_path)
        mask_resized = mask_raw.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        color_map = self._get_color_map(annotation_path)
        if color_map:
            index_mask = _color_mask_to_index(mask_resized, color_map)
        else:
            # fallback: グレースケールマスク（レガシー互換）
            gray = np.array(mask_resized.convert("L"))
            index_mask = (gray > 128).astype(np.int64)

        mask_tensor = torch.from_numpy(index_mask).long()  # (H, W)

        # Data Augmentation (学習時のみ)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [1])
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [1])
                mask_tensor = torch.flip(mask_tensor, [0])

        return img_tensor, mask_tensor


# =====================================================
# Fold分割ヘルパー
# =====================================================
def _make_folds(all_pairs: list[tuple[str, str]], n_folds: int) -> list[list[tuple[str, str]]]:
    """Round-robin分配で均等にn_folds個に分割"""
    folds = [[] for _ in range(n_folds)]
    for i, pair in enumerate(all_pairs):
        folds[i % n_folds].append(pair)
    return folds


# =====================================================
# 単一Fold学習
# =====================================================
def _train_single_fold(
    fold_idx: int,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    fold_save_path: str,
    max_epochs: int,
    global_epoch_offset: int,
    status_callback=None,
    cancel_event: threading.Event | None = None,
    epoch_log_callback=None,
) -> float:
    """1つのfoldを学習し、best_diceを返す。
    各foldはPRETRAINED_PATH or ランダム重みから独立に学習開始（アンサンブル多様性確保）。
    """
    device = torch.device("cuda")

    logger.info(
        f"[Fold {fold_idx}] train={len(train_pairs)}, val={len(val_pairs)}"
    )

    train_ds = MRIDataset(train_pairs, augment=True)
    val_ds = MRIDataset(val_pairs, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # モデル: PRETRAINED_PATH（MRI_TOM best model）から初期化
    model = create_model().to(device)
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[Fold {fold_idx}] 事前学習モデルをロード: {PRETRAINED_PATH}")
    elif os.path.exists(BEST_MODEL_PATH):
        # HITL学習済みモデルがある場合はそこから継続
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[Fold {fold_idx}] 既存best.pthをロード: {BEST_MODEL_PATH}")

    # 損失関数・Optimizer
    loss_fn = MultiClassDiceCELoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_dice = 0.0

    for epoch in range(1, max_epochs + 1):
        # --- キャンセルチェック ---
        if cancel_event is not None and cancel_event.is_set():
            logger.info(f"[Fold {fold_idx}] キャンセル検出 (epoch {epoch})")
            raise TrainingCancelled()

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # --- Validate ---
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = torch.argmax(model(images), dim=1)  # (B, H, W)
                for i in range(preds.shape[0]):
                    d = multiclass_dice(preds[i], masks[i], NUM_CLASSES)
                    dice_scores.append(d.item() if isinstance(d, torch.Tensor) else float(d))

        mean_dice = np.mean(dice_scores) if dice_scores else 0.0

        global_epoch = global_epoch_offset + epoch
        logger.info(
            f"[Fold {fold_idx}] Epoch {epoch}/{max_epochs} (global {global_epoch}) - "
            f"loss: {avg_loss:.4f}, val_dice: {mean_dice:.4f}, best_dice: {best_dice:.4f}"
        )

        # per-epoch メトリクス記録
        if epoch_log_callback:
            epoch_log_callback(fold_idx, epoch, avg_loss, mean_dice)

        # コールバックでステータス更新（global epochで報告）
        if status_callback:
            status_callback(global_epoch, mean_dice, fold_idx)

        # ベストモデル保存
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "fold_idx": fold_idx,
            }, fold_save_path)
            logger.info(f"[Fold {fold_idx}] ベストモデル保存: dice={best_dice:.4f}")

    logger.info(f"[Fold {fold_idx}] 完了: best_dice={best_dice:.4f}")
    return best_dice


# =====================================================
# 学習メインループ（5-fold CV）
# =====================================================
def train_model(
    training_pairs: list[tuple[str, str]],
    model_save_path: str,
    max_epochs: int = 50,
    status_callback=None,
    cancel_event: threading.Event | None = None,
) -> tuple[float, str]:
    """
    5-fold CVで学習を実行し、(5-fold平均Dice, バージョン文字列) を返す。
    各foldモデルは folds/fold_{i}.pt に保存。
    best foldを best.pt にコピー（CoreML変換/互換性）。
    学習完了後にバージョンディレクトリにコピーを保全する。

    training_pairs: DataManagerから受け取った (image_path, annotation_path) のリスト
    """
    from version_manager import (
        get_next_version, create_version_log, set_fold_split_info,
        record_epoch, finalize_version,
    )

    all_pairs = list(training_pairs)

    if len(all_pairs) < MIN_IMAGES_FOR_TRAINING:
        raise ValueError(
            f"ラベル付き画像が{len(all_pairs)}枚しかありません。"
            f"最低{MIN_IMAGES_FOR_TRAINING}枚必要です。"
        )

    # バージョン管理: 初期化
    version = get_next_version()
    version_log = create_version_log(version, all_pairs, max_epochs)
    logger.info(f"バージョン {version} で学習開始")

    # シャッフルしてfold分割
    random.shuffle(all_pairs)
    folds = _make_folds(all_pairs, N_FOLDS)

    logger.info(f"5-Fold CV開始: 合計{len(all_pairs)}枚, fold sizes={[len(f) for f in folds]}")

    fold_dices = []
    best_fold_idx = -1
    best_fold_dice = 0.0

    def epoch_log_cb(fold_idx, epoch, train_loss, val_dice):
        record_epoch(version_log, fold_idx, epoch, train_loss, val_dice)

    try:
        for fold_idx in range(N_FOLDS):
            # val = fold_idx番目, train = 残り全部
            val_pairs = folds[fold_idx]
            train_pairs_fold = []
            for j in range(N_FOLDS):
                if j != fold_idx:
                    train_pairs_fold.extend(folds[j])

            set_fold_split_info(version_log, fold_idx, len(train_pairs_fold), len(val_pairs))

            fold_save_path = get_fold_model_path(fold_idx)
            global_epoch_offset = fold_idx * max_epochs

            fold_dice = _train_single_fold(
                fold_idx=fold_idx,
                train_pairs=train_pairs_fold,
                val_pairs=val_pairs,
                fold_save_path=fold_save_path,
                max_epochs=max_epochs,
                global_epoch_offset=global_epoch_offset,
                status_callback=status_callback,
                cancel_event=cancel_event,
                epoch_log_callback=epoch_log_cb,
            )

            fold_dices.append(fold_dice)

            if fold_dice > best_fold_dice:
                best_fold_dice = fold_dice
                best_fold_idx = fold_idx

        # best foldを best.pt にコピー（CoreML変換/info互換性）
        best_fold_path = get_fold_model_path(best_fold_idx)
        if os.path.exists(best_fold_path):
            shutil.copy2(best_fold_path, BEST_MODEL_PATH)
            logger.info(f"Best fold {best_fold_idx} を best.pt にコピー (dice={best_fold_dice:.4f})")

        # 推論モデルをリロード
        from inference import reload_model
        reload_model()

        mean_dice = float(np.mean(fold_dices))
        logger.info(
            f"5-Fold CV完了: fold_dices={[f'{d:.4f}' for d in fold_dices]}, "
            f"mean_dice={mean_dice:.4f}, best_fold={best_fold_idx} (dice={best_fold_dice:.4f})"
        )

        # バージョン保全
        finalize_version(version, version_log, fold_dices, best_fold_idx, status="completed")

        return mean_dice, version

    except TrainingCancelled:
        # キャンセル時も部分バージョンを保存
        finalize_version(
            version, version_log, fold_dices,
            max(best_fold_idx, 0), status="cancelled",
        )
        raise

    except Exception:
        # エラー時も部分バージョンを保存
        finalize_version(
            version, version_log, fold_dices,
            max(best_fold_idx, 0), status="error",
        )
        raise
