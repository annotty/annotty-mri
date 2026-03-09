"""
全体設定を一箇所に集約
パス・ハイパーパラメータの変更はここだけ
"""
import os

# === パス ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- MRI データ ---
NIFTI_RAW_DIR = os.path.join(DATA_DIR, "nifti", "raw")
NIFTI_LABELS_DIR = os.path.join(DATA_DIR, "nifti", "labels")
SLICES_DIR = os.path.join(DATA_DIR, "slices")

# --- レガシー（眼底画像用、後方互換） ---
COMPLETED_IMAGES_DIR = os.path.join(DATA_DIR, "images_completed", "images")
COMPLETED_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "images_completed", "annotations")
UNANNOTATED_IMAGES_DIR = os.path.join(DATA_DIR, "images_unannotated", "images")
UNANNOTATED_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "images_unannotated", "annotations")

# --- モデル ---
MODELS_DIR = os.path.join(DATA_DIR, "models")
PYTORCH_DIR = os.path.join(MODELS_DIR, "pytorch")
COREML_DIR = os.path.join(MODELS_DIR, "coreml")
PRETRAINED_PATH = os.path.join(PYTORCH_DIR, "pretrained.pt")
CURRENT_PT_DIR = os.path.join(PYTORCH_DIR, "current_pt")
VERSIONS_DIR = os.path.join(PYTORCH_DIR, "versions")
BEST_MODEL_PATH = os.path.join(CURRENT_PT_DIR, "best.pt")
COREML_PATH = os.path.join(COREML_DIR, "SegmentationModel.mlpackage")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# === モデル ===
MODEL_TYPE = "vanilla"  # "vanilla" (MRI_TOM互換) or "smp" (resnet34 encoder)
ENCODER_NAME = "resnet34"      # smpモード用
ENCODER_WEIGHTS = "imagenet"   # smpモード用
IN_CHANNELS = 1   # 1=grayscale MRI, 3=RGB
NUM_CLASSES = 10  # 0=背景 + 9クラス（SR,LR,MR,IR,ON,FAT,LG,SO,EB）
IMAGE_SIZE = 256  # vanilla U-Netのトレーニング解像度

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
    NIFTI_RAW_DIR, NIFTI_LABELS_DIR, SLICES_DIR,
    COMPLETED_IMAGES_DIR, COMPLETED_ANNOTATIONS_DIR,
    UNANNOTATED_IMAGES_DIR, UNANNOTATED_ANNOTATIONS_DIR,
    PYTORCH_DIR, CURRENT_PT_DIR, VERSIONS_DIR, COREML_DIR, STATIC_DIR, LOG_DIR,
]:
    os.makedirs(d, exist_ok=True)
