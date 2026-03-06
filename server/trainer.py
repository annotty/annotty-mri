"""
再学習ワーカー
純PyTorch + smp。MONAI不要。
annotations/ にある赤色マスクデータで U-Net を 5-fold CV で fine-tuning
"""
import os
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
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MIN_IMAGES_FOR_TRAINING, N_FOLDS,
    PRETRAINED_PATH, BEST_MODEL_PATH, get_fold_model_path,
)

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    """学習キャンセル時に送出される例外"""
    pass


# =====================================================
# 損失関数: DiceBCELoss（純PyTorch実装）
# =====================================================
class DiceBCELoss(nn.Module):
    """Dice Loss + Binary Cross Entropy の複合損失
    血管セグメンテーションのclass imbalance対策
    circle_mask で内接円内のみloss計算（眼底画像の有効領域限定）"""

    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1.0, circle_mask=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        # circle_mask: (H, W) bool tensor、円内=True
        self.register_buffer("circle_mask", circle_mask)

    def forward(self, logits, targets):
        # 内接円マスク適用: 円外のピクセルをloss計算から除外
        if self.circle_mask is not None:
            mask = self.circle_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            logits = logits * mask
            targets = targets * mask

        probs = torch.sigmoid(logits)

        # Dice Loss
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # BCE Loss (pos_weight で血管ピクセルを重み付け)
        if self.circle_mask is not None:
            # 円内のみでBCE計算
            mask_bool = self.circle_mask.bool().unsqueeze(0).unsqueeze(0).expand_as(logits)
            logits_masked = logits[mask_bool]
            targets_masked = targets[mask_bool]
            pos_weight = torch.tensor([5.0], device=logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits_masked, targets_masked, pos_weight=pos_weight
            )
        else:
            pos_weight = torch.tensor([5.0], device=logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# =====================================================
# 評価指標: Dice Score（純PyTorch実装）
# =====================================================
def dice_score(pred_mask, gt_mask, smooth=1.0):
    """Dice係数を計算 (入力はバイナリマスク 0/1)"""
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


# =====================================================
# データセット
# =====================================================
class RetinalDataset(Dataset):
    """眼底画像 + 赤色マスクのデータセット

    pairs: list[tuple[str, str]] — (image_path, annotation_path) のフルパスペア
    """
    MEAN = np.array(IMAGENET_MEAN)
    STD = np.array(IMAGENET_STD)

    def __init__(self, pairs: list[tuple[str, str]], augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, annotation_path = self.pairs[idx]

        # 画像読み込み + 正規化
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = (img_arr - self.MEAN) / self.STD
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

        # マスク読み込み: 複数形式に対応
        #   - グレースケール(L): 白(255)=血管, 黒(0)=背景
        #   - RGBA赤色マスク: 赤(255,0,0,255)=血管, 白(255,255,255,255)or透明(0,0,0,0)=背景
        mask_raw = Image.open(annotation_path)
        mask_resized = mask_raw.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        if mask_raw.mode == "L":
            binary = (np.array(mask_resized) > 128).astype(np.float32)
        else:
            mask_arr = np.array(mask_resized.convert("RGBA"))
            binary = (
                (mask_arr[:, :, 0] > 128) &
                (mask_arr[:, :, 1] < 128) &
                (mask_arr[:, :, 3] > 128)
            ).astype(np.float32)
        mask_tensor = torch.from_numpy(binary).unsqueeze(0)  # (1, 512, 512)

        # Data Augmentation (学習時のみ)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [2])
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [1])
                mask_tensor = torch.flip(mask_tensor, [1])

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
    各foldはPRETRAINED_PATH or ImageNet重みから独立に学習開始（アンサンブル多様性確保）。
    """
    device = torch.device("cuda")

    logger.info(
        f"[Fold {fold_idx}] train={len(train_pairs)}, val={len(val_pairs)}"
    )

    train_ds = RetinalDataset(train_pairs, augment=True)
    val_ds = RetinalDataset(val_pairs, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # モデル: PRETRAINED_PATH or ImageNet重みから独立に初期化
    model = create_model().to(device)
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[Fold {fold_idx}] 事前学習モデルをロード: {PRETRAINED_PATH}")

    # 損失関数・Optimizer
    loss_fn = DiceBCELoss()
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
                preds = (torch.sigmoid(model(images)) > 0.5).float()
                for i in range(preds.shape[0]):
                    d = dice_score(preds[i], masks[i])
                    dice_scores.append(d.item())

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
