# DICOM → NIfTI → スライスPNG パイプライン

## いつ使う
新しいDICOMデータセットをAnnotty-MRIに取り込む時

## 前提
- `uv venv .venv --python 3.11` で環境構築済み
- pydicom, SimpleITK, nibabel, matplotlib, numpy, pillow がインストール済み

## 手順

### 1. DICOMデータ確認
```bash
.venv/Scripts/python scripts/view_dicom.py
```
- DCMファイルを全読み込み → PatientID / SeriesInstanceUID で分類
- 各シリーズのメタデータ（解像度, PixelSpacing, SliceThickness, SeriesDescription）を表示
- 出力: `scripts/output/dicom_overview.png`（中央スライス一覧）

### 2. シリーズ選別の判断基準
- **採用**: COR T2 FSE / Cor T2 FRFSE（コントラストほぼ同等、混在OK）
- **除外**: 3-pl T2* SSFSE（スカウト/ロケーター）
- **除外**: T2 FLAIR（水信号抑制あり、T2 FSEと信号特性が異なる）
- **注意**: FSE = Fast Spin Echo, FRFSE = Fast Recovery FSE（GE名称差のみ）

### 3. DICOM → NIfTI 変換
```bash
.venv/Scripts/python scripts/convert_dcm_to_nifti.py
```
- SimpleITK ImageSeriesReader使用（spacing/orientation保持）
- スライスソート: ImagePositionPatientからスライス法線方向を計算して射影ソート
- 出力: `server/data/nifti/raw/sub-XXX/sub-XXX_date-YYYYMMDD_T2w.nii.gz`
- 変換ログ: `server/data/nifti/raw/conversion_log.json`（患者ID対応表含む）

### 4. NIfTI 検証
```bash
.venv/Scripts/python scripts/verify_nifti.py
```
- 出力: `scripts/output/nifti_verification.png`

### 5. NIfTI → スライスPNG（iPad配信用）
```bash
.venv/Scripts/python -m server.medical_adapter.slicer
```
- W/L自動推定（2nd-98th percentile）→ 8bit RGB PNG
- 出力: `server/data/slices/{case_id}/images/`, `slice_manifest.json`, `label_config.json`

### 6. スライスPNG検証
```bash
.venv/Scripts/python scripts/verify_slices.py
```

## よくある問題
- **複数患者が1ディレクトリに混在**: pydicomでPatientID/SeriesInstanceUIDごとに分類
- **画像サイズ不統一**: シリーズ単位で処理（np.stackしない）
- **SimpleITK Non uniform sampling警告**: スライス間隔の微小な揺らぎ。実用上無視可能
