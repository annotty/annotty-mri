"""
再学習ワーカー
純PyTorch。MONAI不要。
annotations/ にあるマルチクラスカラーマスクで U-Net を 5-fold CV で fine-tuning
対応マスク形式:
  - Annotty RGBA PNG: RGBカラー → label_config.json でclass_id(0-9)に変換
  - TOM グレースケール PNG: ピクセル値がそのままclass_id(0-9)
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
    IMAGE_SIZE, MODEL_INPUT_SIZE, IN_CHANNELS, NUM_CLASSES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MIN_IMAGES_FOR_TRAINING, N_FOLDS,
    PRETRAINED_PATH, BEST_MODEL_PATH, get_fold_model_path,
    TOM_PNG_DIR,
)

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    """学習キャンセル時に送出される例外"""
    pass


# =====================================================
# 損失関数: MultiClassDiceCELoss
# =====================================================
class MultiClassDiceCELoss(nn.Module):
    """CrossEntropy + マルチクラスDice Loss
    logits: (B, C, H, W)
    targets: (B, H, W) int64  class_id 0〜C-1
    前景クラス(1〜C-1)のマクロ平均Diceを使用。背景(0)は除外。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        n_classes = logits.shape[1]
        targets_onehot = (
            F.one_hot(targets, num_classes=n_classes)
            .permute(0, 3, 1, 2)
            .float()
        )  # (B, C, H, W)

        inter = (probs * targets_onehot).sum(dim=(2, 3))   # (B, C)
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_per_class = (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_class[:, 1:].mean()  # 前景クラスのみ

        return ce_loss + dice_loss


# =====================================================
# 評価指標: マルチクラス Dice Score
# =====================================================
def dice_score(preds: torch.Tensor, targets: torch.Tensor,
               num_classes: int, smooth: float = 1.0) -> float:
    """前景クラス(1〜num_classes-1)のマクロ平均Diceを返す。
    preds:   (B, H, W) int64 — argmax済み予測クラス
    targets: (B, H, W) int64 — GTクラス
    """
    dice_list = []
    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        gt_c = (targets == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice_list.append(((2.0 * inter + smooth) / (union + smooth)).item())
    return float(np.mean(dice_list)) if dice_list else 0.0


# =====================================================
# カラーマップ読み込みヘルパー
# =====================================================
def load_color_map(pairs: list[tuple[str, str]]) -> dict[tuple[int, int, int], int]:
    """ペアのいずれかのディレクトリにある label_config.json から
    {(R, G, B): class_id} のマッピングを構築して返す。
    見つからない場合は空dict。
    """
    import json
    from pathlib import Path

    for image_path, _ in pairs:
        config_path = Path(image_path).parent.parent / "label_config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                tuple(c["color"]): c["id"]
                for c in cfg.get("classes", [])
            }
    logger.warning("label_config.json が見つかりません。カラーマップなしで進めます。")
    return {}


# =====================================================
# データセット: MRISegDataset
# =====================================================
class MRISegDataset(Dataset):
    """MRI画像 + マルチクラスマスクのデータセット。

    対応マスク形式:
      - Annotty RGBA PNG: RGBカラー → color_to_class でclass_id(0-9)に変換
      - TOM グレースケール PNG: ピクセル値 = class_id(0-9)

    画像前処理: グレースケール → percentile_clip(1-99%) → Z-score → 3ch複製
    マスク出力: (H, W) int64  class_id 0〜NUM_CLASSES-1
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        augment: bool = False,
        color_to_class: dict[tuple[int, int, int], int] | None = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.color_to_class = color_to_class or {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        # --- 画像: Z-score正規化（inference.py と同一） ---
        img = Image.open(image_path).convert("L")
        arr = np.array(img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))).astype(np.float32)
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip(arr, p1, p99)
        std = arr.std()
        arr = (arr - arr.mean()) / std if std > 1e-6 else arr - arr.mean()
        img_tensor = torch.from_numpy(np.stack([arr, arr, arr], axis=0)).float()  # (3,H,W)

        # --- マスク: class_id マップに変換 ---
        mask_raw = Image.open(mask_path)
        mask_resized = mask_raw.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.NEAREST)

        if mask_raw.mode == "L":
            # TOM グレースケール: ピクセル値 = class_id
            class_mask = np.array(mask_resized, dtype=np.int64)
        else:
            # Annotty RGBA: RGBカラー → class_id
            rgba = np.array(mask_resized.convert("RGBA"), dtype=np.uint8)
            class_mask = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.int64)
            for (r, g, b), class_id in self.color_to_class.items():
                match = (
                    (rgba[:, :, 0] == r) &
                    (rgba[:, :, 1] == g) &
                    (rgba[:, :, 2] == b) &
                    (rgba[:, :, 3] > 128)
                )
                class_mask[match] = class_id

        mask_tensor = torch.from_numpy(class_mask).long()  # (H, W) int64

        # --- Augmentation (学習時のみ) ---
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [1])
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [1])
                mask_tensor = torch.flip(mask_tensor, [0])

        return img_tensor, mask_tensor


# =====================================================
# TOM500 ペア収集
# =====================================================
def collect_tom_pairs(tom_png_dir: str) -> list[tuple[str, str]]:
    """TOM500 PNG変換済みディレクトリから (image_path, label_path) ペアを収集する。
    構造: <tom_png_dir>/<case_id>/images/slice_NNN.png
          <tom_png_dir>/<case_id>/labels/slice_NNN.png
    """
    from pathlib import Path
    pairs = []
    root = Path(tom_png_dir)
    if not root.is_dir():
        logger.warning(f"TOM PNG ディレクトリが見つかりません: {tom_png_dir}")
        return pairs

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        img_dir = case_dir / "images"
        lbl_dir = case_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("slice_*.png")):
            lbl_path = lbl_dir / img_path.name
            if lbl_path.exists():
                pairs.append((str(img_path), str(lbl_path)))

    logger.info(f"TOM ペア収集: {len(pairs)} スライス ({tom_png_dir})")
    return pairs


# =====================================================
# Fold分割ヘルパー
# =====================================================
def _make_folds(
    all_pairs: list[tuple[str, str]], n_folds: int
) -> tuple[list[list[tuple[str, str]]], int]:
    """ケース単位 Group k-fold。同一症例のスライスがtrain/valに混在しない。
    パス構造: slices/<case_id>/images/slice_NNN.png からcase_idを取得。

    ケース数が n_folds より少ない場合は effective_folds = ケース数に削減する。
    ケース数が1の場合は 1-fold（train=val=全件）で過学習確認用として返す。

    Returns: (folds, effective_folds)
    """
    from pathlib import Path

    # case_id ごとにペアをグループ化
    case_map: dict[str, list] = {}
    for pair in all_pairs:
        case_id = Path(pair[0]).parent.parent.name
        case_map.setdefault(case_id, []).append(pair)

    case_ids = sorted(case_map.keys())
    n_cases = len(case_ids)
    effective_folds = min(n_folds, n_cases)

    if effective_folds < n_folds:
        logger.warning(
            f"ケース数({n_cases}) < N_FOLDS({n_folds})のため、"
            f"{effective_folds}-fold CVに削減します"
        )

    if effective_folds == 1:
        # 1症例: train=val=全件（過学習確認用）
        return [list(all_pairs)], 1

    # round-robinでケースをfoldバケットに割り当て
    buckets: list[list[str]] = [[] for _ in range(effective_folds)]
    for i, cid in enumerate(case_ids):
        buckets[i % effective_folds].append(cid)

    folds = []
    for bucket in buckets:
        bucket_set = set(bucket)
        folds.append([p for cid in case_ids if cid in bucket_set for p in case_map[cid]])
    return folds, effective_folds


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

    color_to_class = load_color_map(train_pairs)
    train_ds = MRISegDataset(train_pairs, augment=True, color_to_class=color_to_class)
    val_ds = MRISegDataset(val_pairs, augment=False, color_to_class=color_to_class)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # モデル: PRETRAINED_PATH から独立に初期化
    model = create_model().to(device)
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
        model.load_state_dict(state_dict)
        logger.info(f"[Fold {fold_idx}] 事前学習モデルをロード: {PRETRAINED_PATH}")

    # 損失関数・Optimizer
    loss_fn = MultiClassDiceCELoss()
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
                preds = model(images).argmax(dim=1)  # (B, H, W) int64
                for i in range(preds.shape[0]):
                    d = dice_score(preds[i], masks[i], num_classes=NUM_CLASSES)
                    dice_scores.append(d)

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

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
) -> tuple[float, str, int]:
    """
    5-fold CVで学習を実行し、(平均Dice, バージョン文字列, effective_folds) を返す。
    各foldモデルは folds/fold_{i}.pt に保存。
    best foldを best.pt にコピー（CoreML変換/互換性）。
    学習完了後にバージョンディレクトリにコピーを保全する。

    training_pairs: DataManagerから受け取った (image_path, annotation_path) のリスト
    TOM混合: TOM_PNG_DIR が存在する場合、各trainフォールドにTOMペアを追加する。
             valフォールドはAnnottyのみ（HITL品質の公正な評価のため）。
    """
    from version_manager import (
        get_next_version, create_version_log, set_fold_split_info,
        record_epoch, finalize_version,
    )

    annotty_pairs = list(training_pairs)

    if len(annotty_pairs) < MIN_IMAGES_FOR_TRAINING:
        raise ValueError(
            f"ラベル付き画像が{len(annotty_pairs)}枚しかありません。"
            f"最低{MIN_IMAGES_FOR_TRAINING}枚必要です。"
        )

    # TOM500 ペア収集（TOM_PNG_DIRが存在する場合のみ）
    tom_pairs = collect_tom_pairs(TOM_PNG_DIR) if os.path.isdir(TOM_PNG_DIR) else []
    if tom_pairs:
        logger.info(f"TOM混合学習: {len(tom_pairs)} スライスをtrain foldに追加")
    else:
        logger.info("TOMデータなし。Annottyのみで学習します。")

    # バージョン管理: 初期化
    version = get_next_version()
    version_log = create_version_log(version, annotty_pairs, max_epochs)
    logger.info(f"バージョン {version} で学習開始")

    # Annottyのみでシャッフル・fold分割（valはAnnottyのみ）
    random.shuffle(annotty_pairs)
    folds, effective_folds = _make_folds(annotty_pairs, N_FOLDS)

    logger.info(
        f"{effective_folds}-Fold CV開始: Annotty={len(annotty_pairs)}枚, "
        f"TOM={len(tom_pairs)}枚, fold sizes={[len(f) for f in folds]}"
    )

    fold_dices = []
    best_fold_idx = -1
    best_fold_dice = 0.0

    def epoch_log_cb(fold_idx, epoch, train_loss, val_dice):
        record_epoch(version_log, fold_idx, epoch, train_loss, val_dice)

    try:
        for fold_idx in range(effective_folds):
            # val = Annotty fold_idx番目のみ（HITL品質の公正評価）
            val_pairs = folds[fold_idx]

            # train = Annotty残り + TOM全件
            train_pairs_fold = []
            for j in range(N_FOLDS):
                if j != fold_idx:
                    train_pairs_fold.extend(folds[j])
            train_pairs_fold.extend(tom_pairs)

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

        return mean_dice, version, effective_folds

    except TrainingCancelled:
        finalize_version(
            version, version_log, fold_dices,
            max(best_fold_idx, 0), status="cancelled",
        )
        raise

    except Exception:
        finalize_version(
            version, version_log, fold_dices,
            max(best_fold_idx, 0), status="error",
        )
        raise
