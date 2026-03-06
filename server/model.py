"""
U-Net モデル定義
smp (segmentation_models_pytorch) を使用
"""
import segmentation_models_pytorch as smp
from config import ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, NUM_CLASSES


def create_model():
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,  # sigmoid はloss/推論側で適用
    )
