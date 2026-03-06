# MRI_Annotty 設計方針 (SPEC)

## プロジェクト概要

眼窩MRI（coronal, DICOM）のセグメンテーション学習に、iPadを用いたHITL（Human-in-the-Loop）アノテーションを組み込むシステム。既存の [annotty-hil](https://github.com/annotty/annotty-hil.git) をベースに、医用画像（DICOM/NIfTI）対応を追加する。

---

## 1. 画像形式パイプライン

| 段階 | 形式 | 理由 |
|------|------|------|
| 保存・学習基盤 | NIfTI (.nii.gz) | MONAI/nnU-Netのデファクト。3Dボリュームを1ファイルで管理 |
| DICOM → 内部変換 | `dcm2niix` / SimpleITK | spacing, orientation, origin メタデータ保持 |
| サーバー → iPad | W/L適用済み 8-bit PNG（スライス単位） | UIImage互換、軽量、W/Lはサーバー側で制御 |
| iPad → サーバー（マスク） | インデックスカラー PNG（パレットモード） | ロスレス、クラスID=ピクセル値、そのままnumpy変換可能 |
| CoreML推論結果（iPad上） | MLMultiArray → argmaxインデックスマップ | 表示はRGBAオーバーレイ、返却はindex PNG |

### ラベル定義（眼窩MRI）

| ID | 構造 |
|----|------|
| 0 | 背景 |
| 1 | 外直筋 (lateral_rectus) |
| 2 | 内直筋 (medial_rectus) |
| 3 | 上直筋 (superior_rectus) |
| 4 | 下直筋 (inferior_rectus) |
| 5 | 視神経 (optic_nerve) |
| 6 | 眼球 (globe) |

---

## 2. 学習アプローチ

**2D / 2.5D を採用**（3Dは不採用）

### 理由
- coronal限定、スライス厚が厚い（2-3mm）、in-plane 0.5mm → 異方性比 4〜6倍
- スライス間情報量が少なく、3D convのthrough-plane方向が機能しない
- 3Dモデルはパラメータ数が多く、現在のデータ量（TOM500: 500例程度）では不十分
- few-shot adaptation（科研費テーマ）との方向性にも2Dが合致

### 具体的手法
1. **2D UNet / SegFormer** でスライス単位セグメンテーション（データ数 = 症例数 × スライス数）
2. **2.5D**: 前後1〜2枚を追加チャネルとして入力（`spatial_dims=2`のまま）
3. **後処理**: Z方向のconnected component analysis / morphological smoothingで3D整合性を確保
4. **few-shot LoRA adaptation** で少数データへの適応

---

## 3. PNG ↔ NIfTI 整合性設計

**核心: 変換メタデータ（manifest）を往復させる**

### 送信時（サーバー → iPad）
```
NIfTI volume → スライス抽出 → 前処理記録 → PNG + slice_manifest.json
```

### slice_manifest.json
```json
{
  "volume_id": "patient_001_T2",
  "nifti_path": "data/patient_001_T2.nii.gz",
  "slice_index": 15,
  "slice_axis": "coronal",
  "original_shape": [512, 480],
  "affine": [[4x4 matrix]],
  "preprocessing": [
    {"op": "flip", "axis": "horizontal"},
    {"op": "pad", "padding": [0, 16, 0, 16]},
    {"op": "resize", "from": [512, 512], "to": [384, 384]}
  ]
}
```

### 返却時（iPad → サーバー）
```
mask PNG + manifest → 逆変換（resize戻し → unpad → flip戻し） → NIfTIスライスに挿入
```

### 実装制約
- **前処理は可逆操作のみ**（整数比リサイズ、nearest neighborなど）
- **W/L正規化は表示用PNGのみ**、マスクには影響させない
- **augmentationはアノテーション画像には適用しない**（学習時のみ）
- **NIfTIのaffineは絶対に変更しない**（元ボリュームのaffine/headerをコピー）
- **round-tripテスト必須**
- **iPad側はピクセル座標のみ**（座標変換の責務はすべてサーバー側）

---

## 4. アノテーション保存形式

**NIfTI ラベルマップ + BIDS風命名 + JSON sidecar**

### ディレクトリ構造
```
dataset/
  raw/
    sub-001/
      sub-001_T2w.nii.gz
  labels/
    sub-001/
      sub-001_T2w_dseg.nii.gz       # ラベルマップ
      sub-001_T2w_dseg.json          # メタデータ
```

### JSON sidecar
```json
{
  "labels": {"0": "background", "1": "lateral_rectus", ...},
  "annotator": "annotator_A",
  "tool": "HIL-Next/Annotty",
  "model_version": "v0.3_2.5D_unet",
  "annotation_type": "model_assisted_corrected",
  "slices_modified": [8, 9, 10, 14, 15],
  "time_spent_seconds": 320,
  "created_at": "2026-03-06T10:30:00+09:00"
}
```

### 選定理由
- nnU-Net / MONAI にそのまま投入可能
- HITLプロベナンス追跡可能（誰が、どのモデルで、どのスライスを修正したか）
- BIDS準拠への移行が容易
- DICOM SEGは臨床実装/PMDA申請段階で変換対応すれば十分

---

## 5. API通信設計

### サーバー → iPad レスポンス
```json
{
  "image": "base64エンコードPNG or URL",
  "slice_index": 15,
  "total_slices": 30,
  "window_center": 300,
  "window_width": 600,
  "original_shape": [512, 512],
  "spacing": [0.5, 0.5, 3.0]
}
```

### iPad → サーバー リクエスト（マスク + ストローク）
```json
{
  "mask_png": "base64...",
  "strokes": [
    {
      "class_id": 1,
      "points": [[x, y, pressure, timestamp], ...],
      "tool": "brush|eraser"
    }
  ],
  "slice_index": 15,
  "time_spent_seconds": 45
}
```

---

## 6. アーキテクチャ方針：プラグイン/アダプター（B案採用）

既存の [annotty-hil](https://github.com/annotty/annotty-hil.git) に対して、**サーバー側アダプター追加**を主軸とし、iPad側の改修を最小限に抑える。

### 設計の要点

**iPad側の改修をほぼ不要にする2つの判断:**

1. **症例単位インポート/エクスポート**: NIfTIボリュームをサーバー側でスライス分解し、1症例（5〜10枚のPNG）をまとめてiPadにインポート。アノテーション後にまとめてサーバーに戻す。→ 既存の画像リストUIがそのままスライスナビゲーションとして機能
2. **グレースケール → RGB変換はサーバー側**: W/L適用後のグレースケールをRGB PNG（3ch同値）に変換してからiPadに送る。→ iPadは通常のRGB画像として扱うだけ

### 変更範囲

| コンポーネント | 変更内容 | 規模 |
|--------------|---------|------|
| **サーバー: `medical_adapter/`** (新規) | DICOM→NIfTI変換、W/L→RGB PNG生成、症例単位スライス分解、manifest管理 | 中 |
| **サーバー: `main.py`** (既存) | 症例インポート/エクスポートのエンドポイント追加 | 小 |
| **サーバー: マスク再統合** (新規) | 返却されたindex PNG群 → NIfTIラベルマップ再構成、affine復元 | 中 |
| **iPad: クラス定義UI** | 8色常時表示 → 定義済みのみ表示 + 「Add Class」ボタン | 小〜中 |
| **iPad: label_config.json** | プロジェクトルートへのJSON読み書き | 小 |
| **iPad: それ以外** | 変更なし（描画UI, Metal, CoreML, Export等はそのまま） | なし |

### annotty-hil 既存アーキテクチャの活用

| 既存機能 | 活用方法 |
|---------|---------|
| 画像リストUI | スライス一覧としてそのまま使用（ファイル名に `slice_03.png` 等） |
| PUT `/submit/{id}` マスク送信 | 各スライスのマスクPNGをそのまま送信 |
| ExportプラグインUI | PNG/COCO/YOLO出力はそのまま利用可能 |
| CoreML推論 | 2D UNetモデルをサーバーで変換・配信する既存フローをそのまま活用 |
| 5-fold CV学習 | サーバー側の学習パイプラインをMONAI/医用画像対応に拡張 |

### label_config.json による双方向クラス定義

PCとiPadのどちらからでもクラス定義を開始でき、双方向に同期する仕組み。

**ファイル仕様:**
```json
{
  "classes": [
    {"id": 1, "name": "lateral_rectus",  "color": [255, 0, 0]},
    {"id": 2, "name": "medial_rectus",   "color": [255, 128, 0]},
    {"id": 3, "name": "superior_rectus", "color": [255, 255, 0]},
    {"id": 4, "name": "inferior_rectus", "color": [0, 255, 0]},
    {"id": 5, "name": "optic_nerve",     "color": [0, 255, 255]},
    {"id": 6, "name": "globe",           "color": [0, 0, 255]}
  ]
}
```

**配置場所:** プロジェクトルート直下
```
case_001/
  images/
    slice_01.png
    slice_02.png
  annotations/
  labels/
  label_config.json   ← ここ
```

**フロー A: PC → iPad（JSONあり）**
1. サーバー側で症例フォルダ生成時に `label_config.json` を同梱
2. iPad側で `openProject()` 時にJSONを検出
3. `classNames` に名前を反映（既存のpresetColorsの色順にマッピング）

**フロー B: iPad → PC（JSONなし → iPad上で定義）**
1. JSONなしのフォルダをiPadで開く → デフォルト名 "Class 1", "Class 2" ...
2. アノテーターがSettings画面のTextFieldでクラス名を入力（既存UI）
3. プロジェクト保存時に `label_config.json` を自動生成
4. フォルダごとPCに返却 → サーバー側でJSON読み込み → NIfTI sidecarに反映

**iPad側の改修:**

1. **`ProjectFileService`**: プロジェクトルートに `label_config.json` の読み書きメソッド追加
2. **`CanvasViewModel`**: `openProject()` 時にJSON検出→`classNames`反映、クラス変更時にJSON自動保存
3. **`RightPanelView`**: クラス定義UIの変更（下記参照）

**色の対応ルール:**
- `label_config.json` の `color` フィールドはサーバー側のマスクPNG生成用
- iPad側の表示色は既存の `presetColors`（赤,橙,黄,緑,水,青,紫,桃）の順番で固定
- サーバー側がこの色順に合わせてマスクPNGを生成すれば、自動検出で整合する
- → iPad側の色表示コードは変更不要

### クラス定義UI（iPad側の改修）

**課題:** 現状は8色が常時表示され、名前未定義のまま塗れてしまう。医用画像では「何の構造か不明なマスク」がサーバーに戻ると危険。

**方針:** 2クラス以上の場合、クラスを明示的に定義してから使う。

**UIの変更（RightPanelView）:**

```
[現在]                    [改修後]
┌──────────┐             ┌──────────┐
│ ● 赤     │             │ ● 外直筋  │  ← 定義済みのみ表示
│ ● 橙     │             │ ● 内直筋  │
│ ● 黄     │             │ ● 視神経  │
│ ● 緑     │             ├──────────┤
│ ● 水     │             │ ＋ Add    │  ← 新規クラス追加
│ ● 青     │             └──────────┘
│ ● 紫     │
│ ● 桃     │
└──────────┘
```

**「Add Class」フロー:**
1. 「＋」ボタンをタップ
2. クラス名を入力するダイアログ表示（例: "superior_rectus"）
3. 次の未使用プリセット色を自動割当（赤→橙→黄→...の順）
4. RightPanelのリストに追加、即座に描画可能に
5. `label_config.json` に自動保存

**動作ルール:**

| 状態 | 挙動 |
|------|------|
| クラス0個 | 「＋ Add Class」のみ表示。描画不可 |
| クラス1個 | バイナリセグメンテーション。名前入力は任意（従来互換） |
| クラス2個以上 | 全クラスが名前付き必須。名前なしではAddできない |
| label_config.jsonあり | 読み込み時に定義済みクラスをリストに反映 |
| label_config.jsonなし | 空リストから開始。iPadで定義→JSON自動生成 |
| 最大数 | 8クラスまで（既存の `MaskClass.maxClasses` 制約） |

**影響範囲（iPad側）:**

| ファイル | 変更内容 |
|---------|---------|
| `RightPanelView.swift` | presetColors全表示 → 定義済みのみ + Addボタン |
| `ImageSettingsOverlayView.swift` | クラス名TextField: 定義済み分のみ表示 |
| `CanvasViewModel.swift` | `classNames` → 定義済みクラスの動的リスト管理、JSON読み書き |
| `ProjectFileService.swift` | `label_config.json` のload/save追加 |
| `MetalRenderer` / `Shaders.metal` | 変更なし（classIDベースなので影響なし） |
| `ColorMaskParser` | 変更なし（自動検出ロジックはそのまま） |

### ワークフロー

```
[サーバー側]                          [iPad側]
DICOM取り込み                          |
  → dcm2niix → NIfTI保存              |
  → W/L適用 → RGB PNG × N枚生成       |
  → manifest.json 生成                |
  → 症例フォルダとしてパッケージ       |
  ─── インポート ──────────────────→  画像リストに表示
                                      Apple Pencilでアノテーション
                                      (既存UIそのまま)
  ←── エクスポート ────────────────  マスクPNG × N枚を返却
マスクPNG受信                          |
  → manifest参照で逆変換              |
  → NIfTIラベルマップに再統合          |
  → JSON sidecar生成                  |
  → 学習データセットに追加             |
```

---

## 7. 技術スタック（想定）

- **サーバー**: FastAPI (Python)
- **医用画像処理**: MONAI, SimpleITK, nibabel, dcm2niix
- **学習**: MONAI (2D/2.5D UNet, SegResNet等), PyTorch
- **iPad**: SwiftUI, CoreML, Apple Pencil対応
- **通信**: REST API (JSON + base64 PNG)

---

## 8. 対象プロジェクトの区別

| プロジェクト | 対象 | 画像種別 |
|-------------|------|----------|
| **MRI_Annotty** (本プロジェクト) | 眼窩MRI外眼筋セグメンテーション | DICOM → NIfTI → grayscale PNG |
| **CorneAI** (別プロジェクト) | 角膜スリットランプ画像 | スマホ撮影RGB、JPEG/PNGそのまま |

Annottyプラットフォームとしては、画像ソースに応じた前処理アダプターを差し替える設計により、両プロジェクトで同一のアノテーションUI・マスク形式を共用可能。
