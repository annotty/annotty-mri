"""
全体設定を一箇所に集約
パス・ハイパーパラメータの変更はここだけ
"""
import os

# === パス ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# images_completed: アノテーション完了済み学習データ（read-only）
COMPLETED_IMAGES_DIR = os.path.join(DATA_DIR, "images_completed", "images")
COMPLETED_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "images_completed", "annotations")

# images_unannotated: アノテーション対象データ（iPadから書き込み）
UNANNOTATED_IMAGES_DIR = os.path.join(DATA_DIR, "images_unannotated", "images")
UNANNOTATED_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "images_unannotated", "annotations")

MODELS_DIR = os.path.join(DATA_DIR, "models")
PYTORCH_DIR = os.path.join(MODELS_DIR, "pytorch")
COREML_DIR = os.path.join(MODELS_DIR, "coreml")
PRETRAINED_PATH = os.path.join(PYTORCH_DIR, "pretrained.pt")
CURRENT_PT_DIR = os.path.join(PYTORCH_DIR, "current_pt")
VERSIONS_DIR = os.path.join(PYTORCH_DIR, "versions")
BEST_MODEL_PATH = os.path.join(CURRENT_PT_DIR, "best.pt")  # ベストfold（CoreML変換・フォールバック用）
COREML_PATH = os.path.join(COREML_DIR, "SegmentationModel.mlpackage")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# === モデル ===
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
NUM_CLASSES = 1
IMAGE_SIZE = 512

# === 学習 ===
BATCH_SIZE = 4
DEFAULT_MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
N_FOLDS = 5
MIN_IMAGES_FOR_TRAINING = 2

# === ImageNet正規化 ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# === サーバー ===
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# === ヘルパー関数 ===
def get_fold_model_path(fold_idx: int) -> str:
    """fold_idx番目のfoldモデルのパスを返す（current_pt/ 内）"""
    return os.path.join(CURRENT_PT_DIR, f"fold_{fold_idx}.pt")

# === ディレクトリ自動作成 ===
for d in [
    COMPLETED_IMAGES_DIR, COMPLETED_ANNOTATIONS_DIR,
    UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR,
    PYTORCH_DIR, CURRENT_PT_DIR, VERSIONS_DIR, COREML_DIR, STATIC_DIR, LOG_DIR,
]:
    os.makedirs(d, exist_ok=True)
