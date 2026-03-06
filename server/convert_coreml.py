"""
PyTorch → CoreML 変換スクリプト
iPadでのローカル推論用に .mlpackage を生成

Usage:
  python convert_coreml.py
  python convert_coreml.py --input data/models/pytorch/current_pt/best.pt --output data/models/coreml/SegmentationModel.mlpackage
"""
import os
import logging

import torch
import coremltools as ct

from model import create_model
from config import BEST_MODEL_PATH, COREML_PATH, IMAGE_SIZE

logger = logging.getLogger(__name__)


def convert_to_coreml(pytorch_path: str = None, output_path: str = None) -> str:
    """
    PyTorchチェックポイントをCoreML .mlpackage に変換する。

    Args:
        pytorch_path: 入力 .pt ファイルパス (デフォルト: config.BEST_MODEL_PATH)
        output_path: 出力 .mlpackage パス (デフォルト: config.COREML_PATH)

    Returns:
        output_path: 保存先パス
    """
    pytorch_path = pytorch_path or BEST_MODEL_PATH
    output_path = output_path or COREML_PATH

    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(f"PyTorchモデルが見つかりません: {pytorch_path}")

    # 1. PyTorchモデルロード
    model = create_model()
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"PyTorchモデルロード完了: {pytorch_path}")

    # 2. TorchScriptに変換
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, dummy_input)
    logger.info("TorchScript変換完了")

    # 3. CoreMLに変換
    #    TensorType: MLMultiArray入力を受け付ける（ImageTypeはCVPixelBuffer必須で不便）
    #    出力名 "logits" はiPadアプリ側と一致させる
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE))],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
    )
    logger.info("CoreML変換完了")

    # 4. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    logger.info(f"CoreMLモデル保存: {output_path}")

    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pytorch_path = BEST_MODEL_PATH
    output_path = COREML_PATH

    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        pytorch_path = sys.argv[idx + 1]
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_path = sys.argv[idx + 1]

    result = convert_to_coreml(pytorch_path, output_path)
    print(f"CoreML model saved to {result}")
