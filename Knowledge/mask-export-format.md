# マスクエクスポート形式（サーバー送信用）

## 正しい形式（サーバー期待値）
- **サイズ**: 512x512
- **形式**: RGBA カラー PNG
- **エンコード**: classID → `classRGBColors` テーブルで RGB に変換、Alpha=255。背景は (0,0,0,0)
- **色空間**: `CGColorSpaceCreateDeviceRGB()`
- **bitmapInfo**: `CGImageAlphaInfo.premultipliedLast`

## 過去に試したが不採用の形式
- 256x256 グレースケール index PNG（pixel値=classID）
  - サイズ小・処理速いが、サーバー側の期待と不一致

## ラベル設定の受け渡し
- サーバーからの取得: `downloadLabelConfig()` → `LabelConfigResponse` (classes: `[LabelClassInfo]`)
- iPad側への適用: `ProjectFileService.saveLabelConfig()` → `loadLabelConfigFromProject()`
- メモリ直接適用（`applyServerLabelConfig`）方式は不採用。ファイルI/O経由の方が永続性あり

## 関連ファイル
- `CanvasViewModel.swift` の `exportMaskForServer()`
- `HILServerClient.swift` の `LabelClassInfo`, `downloadLabelConfig()`
- `HILViewModel.swift` の `connect()`, `importImages()`, `loadImageIntoCanvas()`
