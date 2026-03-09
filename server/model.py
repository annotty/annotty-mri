"""
U-Net モデル定義
vanilla_unet: MRI_TOM互換のオリジナルU-Net（1ch grayscale入力）
smp_unet: smp (segmentation_models_pytorch) ベース（3ch RGB入力）
"""
from config import IN_CHANNELS, NUM_CLASSES, MODEL_TYPE


def create_model(model_type: str = None):
    """モデルを生成する。

    Args:
        model_type: "vanilla" or "smp"。Noneの場合config.MODEL_TYPEを使用。
    """
    model_type = model_type or MODEL_TYPE

    if model_type == "vanilla":
        from vanilla_unet import VanillaUNet
        return VanillaUNet(
            in_channels=IN_CHANNELS,
            n_classes=NUM_CLASSES,
            bilinear=True,
        )
    elif model_type == "smp":
        import segmentation_models_pytorch as smp
        from config import ENCODER_NAME, ENCODER_WEIGHTS
        return smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
