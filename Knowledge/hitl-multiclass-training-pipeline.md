# HITLマルチクラストレーニングパイプラインの知見

## バイナリ→マルチクラス移行時の注意点

### 損失関数
- バイナリ: `sigmoid` + `BCEWithLogitsLoss` + Binary Dice
- マルチクラス: `softmax` + `CrossEntropyLoss` + Per-class Dice
- **重要**: Dice Lossで背景(class=0)を除外すること。背景は面積が大きく、含めるとDiceが常に高くなり学習が進まない

```python
# マルチクラスDice（背景除外）
for c in range(1, num_classes):  # skip class 0
    pred_c = probs[:, c]
    gt_c = targets_onehot[:, c]
    dice_sum += (2 * intersection + smooth) / (union + smooth)
```

### 評価指標
- バイナリ: 単純なDice係数
- マルチクラス: per-class Dice（背景除外平均）
  - スライスにそのクラスが存在しない場合はスキップ（`gt.sum()==0 and pred.sum()==0`）
  - 全クラスが存在しないスライスも多い（例：視神経は限られたスライスにのみ出現）

### 推論パイプライン
- バイナリ: `sigmoid > 0.5` → 0/1マスク
- マルチクラス: `softmax → argmax` → クラスインデックス
- アンサンブル: `sigmoid平均` → `softmax平均`に変更（logitsを直接平均するとスケールが合わない）

## インデックスカラーマスクの扱い

### iPad↔サーバー間のマスク形式
- **RGBA PNG**: 各クラスに固有色（`label_config.json`で定義）
- 背景は透明 `(0,0,0,0)`、各クラスは不透明 `(R,G,B,255)`

### カラー→インデックス変換のポイント
- `alpha > 128` チェック必須（背景=透明をclass 0にマッピング）
- NEAREST補間でリサイズ（バイリニアだとクラス境界で不正値が出る）
- カラーマップはcase_idごとにキャッシュしてJSON読み込みを最小化

## Grayscale 1ch入力の正規化

### MRI_TOMとの互換性
- MRI_TOM学習時の正規化: `pixel / 255.0`（ImageNet正規化は使わない）
- RGB 3ch + ImageNet正規化は眼底画像用で、MRIには不適
- iPad側の`createGrayscaleInputArray()`でも同じ`pixel/255.0`を使用 → 推論結果一致

### 注意
- 画像配信時にグレースケール→RGB変換している（iPad MTKTextureLoaderの都合）
- iPad側でRGBに戻ってきた画像を再度grayscaleに変換してからモデル入力にしている

## モデルチェックポイントの互換性

### MRI_TOMの保存形式
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "metrics": dict,
    "best_val_dice": float,
    "history": dict,
    "model_name": str,
    # ... その他
}
```

### Annotty-MRI HILの保存形式
```python
{
    "model_state_dict": OrderedDict,
    "epoch": int,
    "best_dice": float,
    "fold_idx": int,
}
```

- 共通キーは `model_state_dict` → `model.load_state_dict(ckpt["model_state_dict"])` で統一的にロード可能
- PRETRAINED_PATH → best.pth の順でフォールバック

## CoreML変換のタイミング

### 自動変換の実装
- `POST /train` 完了後に `convert_to_coreml()` を自動実行
- 失敗してもwarningのみ（手動で`POST /models/convert`も可能）
- CoreML変換にはcoremltools必要（`pip install coremltools`）
- 変換先: `data/models/coreml/SegmentationModel.mlpackage/`
- iPadダウンロード: `GET /models/latest` → ZIP圧縮して配信
