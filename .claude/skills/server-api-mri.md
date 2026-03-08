# MRI HILサーバー API 操作

## いつ使う
サーバーAPIの変更・デバッグ・テスト時

## アーキテクチャ

```
server/
├── config.py              # パス・ハイパーパラメータ一括管理
├── data_manager.py        # データアクセス層（slices/{case_id}/ ベース）
├── main.py                # FastAPI エンドポイント
├── medical_adapter/
│   ├── slicer.py          # NIfTI → スライスPNG + manifest
│   └── reconstructor.py   # マスクPNG → NIfTI _dseg.nii.gz 再統合
├── model.py               # U-Net定義（smp）
├── trainer.py             # 5-fold CV学習ワーカー
├── inference.py           # アンサンブル推論
└── data/
    ├── nifti/raw/         # 元NIfTIボリューム（正本）
    ├── nifti/labels/      # 再統合されたラベルマップ
    ├── slices/            # iPad通信用PNG（症例フォルダ単位）
    └── models/            # PyTorch / CoreML モデル
```

## データフロー

```
DICOM → nifti/raw/ → slices/{case_id}/images/ → iPad → slices/{case_id}/annotations/ → nifti/labels/
```

## 主要API

| Method | Path | 説明 |
|--------|------|------|
| GET | `/info` | サーバー情報 |
| GET | `/cases` | 症例一覧 |
| GET | `/cases/{id}/images` | スライス一覧 |
| GET | `/cases/{id}/images/{img}/download` | PNG取得 |
| PUT | `/cases/{id}/submit/{img}` | マスク送信 |
| GET | `/cases/{id}/labels/{img}/download` | マスク取得 |
| GET | `/cases/{id}/next` | 次の未ラベル |
| GET | `/cases/{id}/manifest` | NIfTI復元メタデータ |
| GET | `/cases/{id}/label_config` | クラス定義 |
| POST | `/cases/{id}/reconstruct` | NIfTI再統合 |
| POST | `/train` | 学習開始 |
| GET | `/status` | 学習ステータス |

## テスト方法
```bash
# DataManager + reconstructor の単体テスト
.venv/Scripts/python scripts/test_new_api.py

# サーバー起動
cd server && ../.venv/Scripts/python main.py

# API呼び出し例
curl http://localhost:8000/cases
curl http://localhost:8000/cases/sub-001_date-20150324/images
```

## セキュリティ
- `validate_case_id()`: `[\w\-\.]+` のみ許可、`..` / `/` / `\` 拒否
- `validate_image_id()`: `[\w\-\.]+\.png$` のみ許可
- `save_annotation()`: images/ に存在するIDのみ受付（書き込み先制限）

## マスク再統合（reconstructor.py）の仕組み
1. `slice_manifest.json` からaffine, volume_shape, slice_axisを取得
2. `label_config.json` の色定義でRGBAマスク → クラスIDに変換（色距離80以内）
3. 空のラベルボリューム（uint8）にスライス単位で挿入
4. 元ボリュームと同一affineで `_dseg.nii.gz` として保存
5. JSON sidecar（クラス名, アノテータ, 修正スライス一覧）を生成
