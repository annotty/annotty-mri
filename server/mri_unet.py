"""
MRI U-Net モデル定義
best.pt チェックポイントのアーキテクチャに対応:
  encoders / bottleneck / upconvs / decoders / final
"""
import torch
import torch.nn as nn


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MRIUNet(nn.Module):
    """
    channels = [32, 64, 128, 256], bottleneck = 512, classes = 10
    """
    def __init__(self, in_channels=1, n_classes=10, base_channels=32):
        super().__init__()
        ch = [base_channels * (2 ** i) for i in range(4)]  # [32,64,128,256]
        bot = ch[-1] * 2  # 512

        self.encoders = nn.ModuleList([
            _double_conv(in_channels, ch[0]),
            _double_conv(ch[0], ch[1]),
            _double_conv(ch[1], ch[2]),
            _double_conv(ch[2], ch[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(ch[3], bot)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(bot,   ch[3], 2, stride=2),
            nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2),
            nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2),
            nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2),
        ])
        self.decoders = nn.ModuleList([
            _double_conv(bot,   ch[3]),
            _double_conv(ch[3], ch[2]),
            _double_conv(ch[2], ch[1]),
            _double_conv(ch[1], ch[0]),
        ])
        self.final = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)
