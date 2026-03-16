# サーバー推論デバッグ知見

## 問題: inference.py で NameError が発生し推論が動かない

### 原因
- `server/inference.py` で `F.softmax()` を使用しているが、`import torch.nn.functional as F` が記述されていなかった
- `trainer.py` には同じ import があったため、トレーニングは正常動作していた

### 解決策
```python
import torch
import torch.nn.functional as F  # ← これが欠落していた
```

### 教訓
- **新しいモジュールで torch の関数を使う際は import を確認する**。特に `F.softmax`, `F.cross_entropy` など `torch.nn.functional` 経由の関数は忘れやすい
- エラーが iPad 側に見えても、まずサーバーログ（`/infer` エンドポイントの例外）を確認する
- trainer.py と inference.py は似た処理をするが別ファイルなので、片方に入れた import がもう片方にあるとは限らない

### iPad ↔ サーバー間のデータ形式（正常時）

| 項目 | iPad | サーバー | 一致 |
|------|------|----------|------|
| 推論入力 | グレースケール [0,1] 正規化 | グレースケール [0,1] 正規化 | OK |
| モデル入力サイズ | 256x256 (サーバーモデル) | 256x256 (IMAGE_SIZE) | OK |
| 推論出力 | UNetMaskResult (class ID配列) | RGBA カラーPNG (class色) | OK |
| マスク提出 | 512x512 RGBA カラーPNG | save_annotation で保存 | OK |
| 学習時変換 | - | _color_mask_to_index() で色→class ID | OK |
