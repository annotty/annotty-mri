# TestFlight アプリアイコン設定の注意点

## 必須要件
1. **全サイズの個別PNGを用意する** — universal 1024x1024の単一エントリだけではApp Store Connectのバリデーションで弾かれる場合がある
2. **Contents.jsonに全エントリを記載** — iPhone(20,29,40,60pt × 2x,3x)、iPad(20,29,40,76pt × 1x,2x + 83.5pt × 2x)、ios-marketing(1024pt × 1x)
3. **アルファチャンネルなし(RGB)のPNG** — RGBA(透過あり)はリジェクトされる
4. **角丸は不要** — iOSが自動で角丸マスクをかける

## `GENERATE_INFOPLIST_FILE = YES` の場合
- Info.plistに直接キーを書いても反映されないことがある
- **`INFOPLIST_KEY_CFBundleIconName = AppIcon`** を `project.pbxproj` のビルド設定(Debug/Release両方)に追加する
- **`ASSETCATALOG_COMPILER_INCLUDE_ALL_APPICON_ASSETS = YES`** も追加推奨

## Pillowで全サイズ自動生成
1024x1024のソースPNGから `Image.resize()` で各サイズを生成可能。
