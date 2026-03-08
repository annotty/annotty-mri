# MRIシーケンス選別の知見

## 眼窩MRI T2系シーケンスの扱い

| シーケンス | 混在可否 | 理由 |
|-----------|---------|------|
| T2 FSE (Fast Spin Echo) | OK | 標準的なT2 |
| T2 FRFSE (Fast Recovery FSE) | OK | FSE + T1回復パルス。コントラストほぼ同等 |
| T2 FLAIR | NG（除外推奨） | 水信号抑制あり。信号特性がFSEと異なる |
| 3-pl T2* SSFSE | NG（除外） | スカウト/ロケーター。256x256, 粗い解像度 |

## 判別方法
- SeriesDescriptionに `SSFSE` or `3-pl` → スカウト、除外
- SeriesDescriptionに `FLAIR` → FLAIR、学習データ均質性のため除外推奨
- FSE / FRFSE / FSE T2 はGEの名称差のみ、混在OK

## メーカー間名称対応
| GE | Siemens | Philips | 実質 |
|----|---------|---------|------|
| FSE | TSE | TSE | Fast/Turbo Spin Echo（同じ原理） |
| FRFSE | RESTORE | DRIVE | Fast Recovery付き |
