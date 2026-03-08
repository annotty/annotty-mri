# DICOM → NIfTI 変換の知見

## 複数患者/シリーズ混在時の対処
- 1ディレクトリに全DCMが混在するケースがある
- `pydicom.dcmread(f, stop_before_pixels=True)` で高速にメタデータだけ読む
- `PatientID` + `SeriesInstanceUID` でグルーピングしてからシリーズ単位で処理

## SimpleITK ImageSeriesReader のスライスソート
- SimpleITKの `GetGDCMSeriesFileNames()` はディレクトリ単位でしか動かない
- 複数シリーズ混在時は手動でファイルリストを渡す必要がある
- ソート方法: `ImageOrientationPatient` からスライス法線ベクトルを計算し、`ImagePositionPatient` を射影

```python
iop = ds.ImageOrientationPatient
row_dir = np.array(iop[:3])
col_dir = np.array(iop[3:])
slice_dir = np.cross(row_dir, col_dir)
# slice_dir への射影でソート
dcm_slices.sort(key=lambda x: np.dot(x.position, slice_dir))
```

## Non uniform sampling 警告
- SimpleITKが「Non uniform sampling or missing slices detected」と出ることがある
- スライス間隔の微小な揺らぎ（小数点以下の差）が原因で、実用上は無視可能

## W/L（Window/Level）自動推定
- DICOM TagのWindowCenter/WindowWidthがある場合はそれを使う
- 複数シリーズ混在でW/Lが異なる場合はボリューム単位で自動推定が安全
- 自動推定: 非ゼロピクセルの2nd-98th percentileが安定
