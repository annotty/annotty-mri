dicom形式の眼窩MRIを読み込んでセグメンテーションを学習するフローにiPadを用いたhitlを組み込みます。その場合のやり取りする画像の形式はどのようにするのが良いでしょうか？！
monaiなど既存のhitlシステムを参考にして考えて下さい

3月6日

DICOMからiPad間の画像形式パイプラインを設計した。
DICOMからiPad間の画像形式パイプラインを設計した。
これは HIL-Next/Annotty の眼窩MRIセグメンテーションへの応用ですね。MONAI Label や他の既存システムのアーキテクチャを踏まえて整理します。

パイプライン全体の画像形式設計

1. 保存・学習基盤：NIfTI (.nii.gz)
   MONAI Label が内部的に採用している形式で、医用画像MLのデファクトスタンダードです。DICOM を受け取ったらサーバー側で早い段階で NIfTI に変換し、以降の学習パイプラインはすべて NIfTI ベースで統一するのが合理的です。

SimpleITK や dcm2niix で DICOM → NIfTI 変換
メタデータ（spacing, orientation, origin）が保持される
3Dボリュームとして1ファイルに格納でき、スライス管理が不要
2. サーバー → iPad（表示用）：Window/Level 適用済み 8-bit PNG
ここが最も重要な設計判断です。MONAI Label が OHIF Viewer と通信する際も、REST API 経由でスライス単位の画像を送っています。

8-bit PNG を推奨する理由：

iPad の UIImage / CoreGraphics が最もネイティブに扱える
Apple Pencil でのアノテーション（ストローク描画）は表示画像座標系で行うため、表示品質が確保されていれば十分
眼窩MRI（T2, STIR等）では、適切な Window/Level をサーバー側で事前適用して 8-bit に変換すれば、臨床的に必要な情報は損なわれない
転送サイズが小さく、モバイル回線でも実用的
具体的なフロー：

サーバー側:
NIfTI volume → スライス抽出 → W/L正規化 → 0-255にスケール → PNG encode

API response:
{
  "image": "base64エンコードPNG or URL",
  "slice_index": 15,
  "total_slices": 30,
  "window_center": 300,
  "window_width": 600,
  "original_shape": [512, 512],
  "spacing": [0.5, 0.5, 3.0]
}
16-bit PNG にしない理由： iPad 側で W/L 調整をリアルタイムに行いたい場合は 16-bit が有利ですが、実装の複雑さが増します。サーバー側で複数の W/L プリセット（T2用、STIR用、脂肪抑制用など）を用意して切り替え可能にする方が、iPad 側の実装がシンプルになります。ただし、将来的にiPad上でインタラクティブなW/L調整を入れたくなる可能性があるなら、Float32の生データをバイナリで送り、iPad側でMetal shaderでレンダリングするオプションも設計に含めておくと拡張性が上がります。

3. マスク（iPad → サーバー）：インデックスカラー PNG
   MONAI Label でもラベルマップは整数値の numpy array（NIfTI形式）でやり取りされますが、iPad → サーバー間では：

推奨：単一チャネル PNG（パレットモード）

ピクセル値でクラスをエンコード:
  0 = 背景
  1 = 外直筋
  2 = 内直筋
  3 = 上直筋
  4 = 下直筋
  5 = 視神経
  6 = 眼球
  ...
ロスレスで、クラス境界が劣化しない（JPEGは絶対NG）
サーバー側で np.array(Image.open(png)) → そのまま NIfTI ラベルマップの該当スライスに挿入可能
ファイルサイズが非常に小さい（セグメンテーションマスクは高圧縮率）
Apple Pencil のストロークデータも併送する設計：

json
{
  "mask_png": "base64...",
  "strokes": [
    {
      "class_id": 1,
      "points": [[x,y,pressure,timestamp], ...],
      "tool": "brush|eraser"
    }
  ],
  "slice_index": 15,
  "time_spent_seconds": 45
}
ストロークの生データを保存しておくと、後からアノテーション品質の分析や、アノテーション行動のモデル学習に使えます。

4. CoreML 推論結果（iPad上）→ 表示
   iPad 上で CoreML によるオンデバイス推論を行う場合：

CoreML の出力は MLMultiArray → これをクラスごとの確率マップまたは argmax のインデックスマップとして取得
表示用にはクラスごとに色分けした半透明オーバーレイ RGBA 画像を生成
アノテーター修正後のマスクをサーバーに返すときは上記の index PNG 形式
5. MONAI Label との比較まとめ
要素	MONAI Label (3D Slicer/OHIF)	HIL-Next (iPad) 推奨
内部形式	NIfTI	NIfTI（同じ）
Viewer送信	NIfTI or DICOMweb	W/L適用済み 8-bit PNG（スライス単位）
ラベル形式	NIfTI (int)	Index PNG → サーバーでNIfTI化
通信	REST API	REST API（FastAPI、同じ）
推論	サーバー側GPU	CoreML（オンデバイス）+ サーバー側
3D操作	フル3D対応	2Dスライス単位（必要に応じMPR）
実装上のポイント
一点注意として、眼窩MRIはスライス厚が比較的厚い（2-3mm）ことが多いので、スライス間の連続性をどう担保するかが課題になります。MONAI Label の SlidingWindowInferer のように、3Dで推論してスライス単位で修正を受ける設計が望ましいです。iPad側では隣接スライスのマスクを半透明で重ねて表示すると、アノテーターが3D的な一貫性を意識しやすくなります。

何か特定のフェーズ（例えば CoreML モデルの入出力仕様や、FastAPI のエンドポイント設計など）をさらに詳しく詰めたい部分はありますか？

monaiのトレーニングは3Dでも行えるということ？！

3月6日

MONAIの3D学習機能について技術的に検証した。
MONAIの3D学習機能について技術的に検証した。
はい、むしろ3Dが MONAI の主戦場です。

MONAI の 3D 学習
MONAI は元々、CT・MRI などの3Dボリュームデータを扱うために設計されたフレームワークです。主要なコンポーネントがすべて3D対応になっています：

ネットワーク例：

UNet / AttentionUnet → 2D/3D を spatial_dims=3 の引数一つで切り替え可能
SwinUNETR → 3D Vision Transformer ベースで、BTCV など腹部CTベンチマークで高精度
SegResNet → BraTS（脳腫瘍MRI）チャレンジ向けに設計された3Dアーキテクチャ
3D特有の仕組み：

RandCropByPosNegLabel → 3Dボリュームからパッチをランダム切り出し（GPU メモリに載るサイズに）
SlidingWindowInferer → 推論時にパッチ単位で3Dボリューム全体をスキャン、オーバーラップ部分を重み付け平均
Spacing, Orientation などの Transform → 3D の spacing や向きを自動で統一
眼窩MRIでの現実的な判断としては：

眼窩MRIはスライス数が比較的少ない（20〜30枚程度）ので、ボリューム全体を一括で入力できる場合も多く、パッチ分割が不要なケースもあります。その一方で、スライス内の in-plane 解像度とスライス厚の異方性（例：0.5×0.5×3.0 mm）が大きいので、Spacing Transform で等方的にリサンプルするか、異方性を考慮したカーネル設計にするかは検討が必要です。

TOM500 データセットで外眼筋セグメンテーションをやるなら、SegResNet か SwinUNETR を spatial_dims=3 で使い、異方性 spacing に対応した augmentation を入れるのが一つの定石になります。

iPad の HIL-Next との連携では、学習・推論は3D、アノテーション修正は2Dスライス単位という分離がそのまま MONAI Label のアーキテクチャと同じ考え方です。修正されたスライスマスクをサーバー側で3Dボリュームに再統合してから再学習に回す、という流れになります。

画像が疎であり現時点ではcoronal限定です。3dでは精度を出すのにかなりの数のデータが必要では？？

3月6日

ユーザーの懸念を検証し、3D学習の実現可能性を評価した。
ユーザーの懸念を検証し、3D学習の実現可能性を評価した。
おっしゃる通りで、その条件だと2Dの方が現実的です。

なぜ3Dが不利か（眼窩MRI coronal の場合）
データの物理的な問題：

coronal 限定でスライス厚が厚い（2-3mm）、in-plane が 0.5mm 程度 → 異方性比が 4〜6倍
スライス間の情報量が極めて少なく、3D convolution の through-plane 方向のカーネルがほぼ意味をなさない
3mm 厚で 20 スライス → Z 方向はたった 60mm 分、解像度としては実質 20 ピクセル
データ量の問題：

3D モデルはパラメータ数が 2D の数倍〜十数倍に膨れるので、同じ汎化性能を出すのにデータが桁違いに必要
TOM500 でも 500 例程度で、3D で十分な精度を出すにはぎりぎりか足りない水準
KAKENHI の few-shot adaptation テーマとも方向性が合わない
推奨アプローチ
基本は 2D スライス単位の学習で、スライス間整合性は後処理で担保するのが最も効率的です。

具体的には：

2D UNet / SegFormer 等でスライス単位セグメンテーション — 各 coronal スライスを独立した2D画像として学習。データ数が「症例数 × スライス数」に増えるのも大きなメリット
隣接スライス入力（2.5D） — 前後1〜2枚を追加チャネルとして入力する手法。3D conv を使わずにスライス間文脈を取り込める。実装も簡単で、MONAI の Transform で対応可能
python

# 2.5D: 前後スライスをチャネル方向にスタック

# 入力: [3, H, W]  (slice-1, slice, slice+1)

# spatial_dims=2 のまま

後処理で3D整合性を確保 — スライス単位の予測結果に対して、Z方向に connected component analysis や morphological smoothing を適用
iPad HIL-Next との相性
2D の方がむしろ iPad との親和性が高いです：

CoreML モデルも 2D の方が軽量でオンデバイス推論が快適
アノテーターが修正したスライスと学習単位が一致するので、active learning のフィードバックループが直感的
修正 → 再学習 → 再推論のサイクルが速い
2.5D + iPad HITL + few-shot LoRA adaptation という組み合わせが、今の条件（coronal 限定、データ疎、KAKENHI テーマ）に最もフィットすると思います。3D は将来的にデータが十分蓄積され、かつ multi-planar 撮像が入ってきた段階で検討すれば良いのではないでしょうか。

corneaiは角膜スリットランプ画像なので…

3月6日

プロジェクトを混同していたことを認識し、整理し直した。
プロジェクトを混同していたことを認識し、整理し直した。
すみません、混同しました。プロジェクトを整理すると：

CorneAI → スマホ撮影のスリットランプ／角膜画像（2D、完全に別プロジェクト）
眼窩MRI セグメンテーション → 今回の話題（coronal, 外眼筋等のセグメンテーション）
これらは別のパイプラインですね。

HIL-Next / Annotty をプラットフォームとして両方に使う場合、画像形式の設計で変わってくる点は：

CorneAI 側： そもそもスマホ撮影の RGB カラー画像なので、JPEG/PNG でそのまま iPad に送れる。W/L 変換は不要。

眼窩MRI 側： 先ほどの議論通り、DICOM → NIfTI → W/L 適用済み grayscale PNG → iPad、という変換パイプラインが必要。

Annotty のアーキテクチャとしては、画像ソースに応じた前処理モジュール（adapter）を差し替える設計にしておけば、角膜画像でも MRI でも同じアノテーション UI・同じマスク形式（index PNG + ストロークデータ）でやり取りできるので、プラットフォームとしての汎用性が活きてきます。

MRI 側の 2D/2.5D アプローチの話に戻りますか？それとも Annotty の adapter 設計の方を詰めますか？

pngでアノーテーションした後に、nii.gzとどのように整合性を取りますか？！

3月6日

PNG注釈とNIfTIボリュームの空間的整合性を検討した。
PNG注釈とNIfTIボリュームの空間的整合性を検討した。
これはパイプライン設計の最も重要なポイントですね。ずれると学習データが壊れます。

整合性の問題が起きる箇所
NIfTI → PNG（サーバー側で抽出）：

リサイズ、パディング、W/L正規化で座標系が変わりうる
affine 行列の情報が PNG には存在しない
上下左右の flip が起きやすい（NIfTI の RAS/LPS 座標系と画像座標系の不一致）
PNG マスク → NIfTI に再統合：

どのスライスの、どの座標系で描いたマスクなのかを正確に復元する必要がある
設計方針：「変換メタデータを往復させる」
核心は、PNG を切り出した時の変換パラメータをすべて記録して、マスク返却時に逆変換することです。

── 送信時（サーバー → iPad）──────────────

NIfTI volume (H×W×D)
  │
  ├─ slice_index = 15 を抽出
  ├─ original_shape: [512, 480]
  ├─ orientation: "LPS"
  ├─ flip_applied: ["horizontal"]  ← RAS→画像座標変換で必要だった反転
  ├─ padding: [0, 16, 0, 16]      ← 正方形にするために足したパディング
  ├─ resize: 512 → 384            ← iPad表示用にリサイズした場合
  │
  ▼
PNG (384×384) + slice_manifest.json
json
{
  "volume_id": "patient_001_T2",
  "nifti_path": "data/patient_001_T2.nii.gz",
  "slice_index": 15,
  "slice_axis": "coronal",
  "original_shape": [512, 480],
  "affine": [[...4x4 matrix...]],
  "preprocessing": [
    {"op": "flip", "axis": "horizontal"},
    {"op": "pad", "padding": [0, 16, 0, 16]},
    {"op": "resize", "from": [512, 512], "to": [384, 384]}
  ]
}

```

```

── 返却時（iPad → サーバー）──────────────

mask PNG (384×384) + slice_manifest.json（そのまま返す）
  │
  ▼ サーバー側で逆変換
  ├─ resize: 384 → 512
  ├─ unpad: [0,16,0,16] を除去 → 512×480
  ├─ flip 戻し: horizontal
  │
  ▼
NIfTI volume の slice_index=15 にマスクを挿入
（affine はオリジナルをそのまま使用）
実装のポイント

1. 前処理を可逆に限定する

これが最も大事です。iPad に送る画像に対して不可逆な変換をしない：

リサイズは整数比（÷2 等）か、nearest neighbor で戻せるサイズに限定
W/L 正規化は表示用 PNG にだけ適用、マスクには影響しない
augmentation（回転、弾性変形等）は学習時にのみ適用し、アノテーション画像には絶対にかけない
2. Round-trip テストを必ず書く

python
def test_roundtrip():
    vol = nib.load("test.nii.gz")
    data = vol.get_fdata()

    # 送信側
    slice_img, manifest = extract_slice(vol, index=15)
    png_bytes = encode_png(slice_img)

    # iPad側を模擬（マスクをそのまま返す）
    mask_png = create_dummy_mask(png_bytes.shape)

    # 復元側
    mask_restored = restore_to_volume(mask_png, manifest)

    assert mask_restored.shape == data[:,:,15].shape
    assert_affine_unchanged(vol, restored_vol)
3. NIfTI の affine は絶対に触らない

マスクの NIfTI を作る時は、元ボリュームの affine と header をコピーして使う：

python
mask_vol = np.zeros_like(volume_data, dtype=np.uint8)
mask_vol[:, :, slice_index] = restored_mask

mask_nii = nib.Nifti1Image(mask_vol, affine=original_affine)
こうすれば物理座標系が完全に一致します。

4. iPad 側ではピクセル座標以外を意識させない

iPad のアノテーターは、表示された画像のピクセル座標でストロークを描くだけ。座標変換の責務はすべてサーバー側に持たせることで、iPad アプリの実装がシンプルに保てますし、バグの原因を局所化できます。

MONAI Label はどうしているか
MONAI Label も本質的に同じことをしていて、3D Slicer / OHIF との間で：

サーバーが NIfTI からスライスを配信
Viewer 側でアノテーション
ラベルマップを NIfTI 形式で返却し、サーバーが元の座標系に揃えて保存
違いは、3D Slicer は NIfTI を直接読めるので PNG 変換が不要な点です。iPad を使う以上、manifest による可逆変換の管理が HIL-Next 固有の設計課題になります。

FastAPI のエンドポイント設計まで具体的に詰めますか？

nii.gzのアノーテーションを格納する標準形式はありますか？！

3月6日

NIfTIアノーテーション形式を複数検討し整理した。
NIfTIアノーテーション形式を複数検討し整理した。
良い質問です。実は完全に統一された「標準」は存在しないのが現状ですが、コミュニティごとにデファクトがあります。

主要な形式

1. NIfTI ラベルマップ（最も広く使われる）
   nnU-Net、MONAI、Medical Segmentation Decathlon など主要ベンチマーク・フレームワークがすべてこれを採用しています。

patient_001/
  imaging.nii.gz        # 元画像
  segmentation.nii.gz   # 同じ shape, affine の整数ラベルマップ
各ボクセルに 0, 1, 2, ... のクラスIDが入る
元画像と同一の affine / header を共有するので整合性が自明
multi-label（重なりなし）が前提。外眼筋セグメンテーションのように各構造が排他的な場合はこれで十分
ただし「標準」というよりは「みんなが使っているからそうなった」という状態で、ラベルIDとクラス名の対応を記述する仕様は統一されていません。

2. BIDS Derivatives（最も体系的）
   Brain Imaging Data Structure の派生規格で、神経画像分野ではこれが最も標準化に近い存在です。

derivatives/
  segmentation/
    sub-001/
      sub-001_space-T2w_dseg.nii.gz     # discrete segmentation
      sub-001_space-T2w_dseg.tsv         # ラベル定義
tsv の中身：

index  name           abbreviation  color
1      lateral_rectus  LR           #FF0000
2      medial_rectus   MR           #00FF00
3      optic_nerve     ON           #0000FF
_dseg（discrete segmentation）や _probseg（確率マップ）のようにサフィックスでセグメンテーションの種類を区別
ラベルIDと名称の対応が tsv で明示的に定義される
ファイル命名規則が厳密なので、データ管理が構造化される
眼窩MRI 専用の BIDS extension はまだ無いですが、命名規則を借用するだけでも十分有用です。

3. DICOM SEG
   DICOM の正式なセグメンテーション格納規格です。

PACS やクリニカルワークフローとの互換性が最高
pydicom + highdicom で作成可能
ただし作成が煩雑で、研究用途ではオーバースペック
PMDA 申請など将来的に医療機器として出す場合には意識する必要がある
4. NRRD（3D Slicer ネイティブ）
patient_001.seg.nrrd
3D Slicer でアノテーションすると自然にこの形式になる
セグメンテーションのメタデータ（色、名前、レイヤー情報）がヘッダに含まれる
MONAI Label + 3D Slicer の組み合わせではこれが使われる
ただしMLパイプラインに入れる前に NIfTI 変換が必要になることが多い
眼窩MRI × HIL-Next での推奨
保存は NIfTI ラベルマップ、管理は BIDS 風の命名 + JSON sidecar が最もバランスが良いと思います。

dataset/
  raw/
    sub-001/
      sub-001_T2w.nii.gz

  labels/
    sub-001/
      sub-001_T2w_dseg.nii.gz       # ラベルマップ
      sub-001_T2w_dseg.json          # メタデータ
JSON sidecar：

json
{
  "labels": {
    "0": "background",
    "1": "lateral_rectus",
    "2": "medial_rectus",
    "3": "superior_rectus",
    "4": "inferior_rectus",
    "5": "optic_nerve",
    "6": "globe"
  },
  "annotator": "annotator_A",
  "tool": "HIL-Next/Annotty",
  "model_version": "v0.3_2.5D_unet",
  "annotation_type": "model_assisted_corrected",
  "slices_modified": [8, 9, 10, 14, 15],
  "time_spent_seconds": 320,
  "created_at": "2026-03-06T10:30:00+09:00"
}
こうしておくと：

nnU-Net / MONAI にそのまま食わせられる（NIfTI ラベルマップなので）
HITL のプロベナンスが追跡できる（誰が、どのモデルベースで、どのスライスを修正したか）
将来 BIDS 準拠にしたくなったら命名規則だけ調整すればよい
論文に methods として書きやすい（「BIDS-like directory structure with JSON sidecar metadata」）
DICOM SEG は、臨床実装やPMDA申請の段階で NIfTI から変換する形で対応すれば十分です。研究フェーズでは扱いにくさの方が目立ちます。

nnU-Net のデータセット形式に合わせた具体的なディレクトリ構造も作りましょうか？



もともとのAnnotty-HITLはhttps://github.com/annotty/annotty-hil.gitにリポジトリがあります。これをdicom用に一新するのか、これ用のプラグインなど部品として開発するのか、方針を提案してほしいです。
