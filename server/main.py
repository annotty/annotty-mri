"""
FastAPI サーバー
眼窩MRIセグメンテーション HIL アノテーションシステム
NIfTI → スライスPNG → iPad annotation → NIfTI再統合 のHILループを実現
"""
import os
import re
import logging
import threading
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

from config import (
    BEST_MODEL_PATH, COREML_PATH, LOG_DIR,
    STATIC_DIR, SERVER_HOST, SERVER_PORT, MIN_IMAGES_FOR_TRAINING,
    DEFAULT_MAX_EPOCHS, N_FOLDS,
)
from data_manager import DataManager

# === ログ設定 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orbital MRI HIL Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DataManager ===
dm = DataManager()

# === グローバル学習ステータス (スレッドセーフ) ===
training_status = {
    "status": "idle",
    "epoch": 0,
    "max_epochs": 0,
    "best_dice": 0.0,
    "current_fold": 0,
    "n_folds": N_FOLDS,
    "started_at": None,
    "completed_at": None,
}
training_lock = threading.Lock()
training_cancel_event = threading.Event()


# =====================================================
# セキュリティ: パストラバーサル対策
# =====================================================
def validate_case_id(case_id: str) -> bool:
    """症例IDとして安全かチェック"""
    if not re.match(r"^[\w\-\.]+$", case_id):
        return False
    if ".." in case_id or "/" in case_id or "\\" in case_id:
        return False
    return True


def validate_image_id(image_id: str) -> bool:
    """ファイル名として安全かチェック"""
    if not re.match(r"^[\w\-\.]+\.png$", image_id):
        return False
    if ".." in image_id or "/" in image_id or "\\" in image_id:
        return False
    return True


# =====================================================
# デフォルト症例の解決（iPad互換API用）
# =====================================================
def _get_default_case_id() -> str | None:
    """未ラベルスライスが残っている最初の症例IDを返す（iPad互換エンドポイント用）。
    全症例完了なら最後の症例を返す。"""
    cases = dm.list_cases()
    if not cases:
        return None
    for c in cases:
        if c["unlabeled_slices"] > 0:
            return c["case_id"]
    return cases[-1]["case_id"]


# =====================================================
# GET /info - サーバー情報（iPad互換フィールド付き）
# =====================================================
@app.get("/info")
def get_info():
    stats = dm.get_stats()
    return {
        "name": "Orbital MRI HIL Server",
        # iPad互換フィールド（旧API: totalImages / labeledImages / unlabeledImages）
        "total_images": stats["total_slices"],
        "labeled_images": stats["labeled_slices"],
        "unlabeled_images": stats["unlabeled_slices"],
        # 新API用フィールド
        "total_cases": stats["total_cases"],
        "total_slices": stats["total_slices"],
        "labeled_slices": stats["labeled_slices"],
        "unlabeled_slices": stats["unlabeled_slices"],
        "model_loaded": os.path.exists(BEST_MODEL_PATH),
        "training_status": training_status["status"],
        "total_training_pairs": stats["total_training_pairs"],
    }


# =====================================================
# iPad互換エンドポイント（旧フラットAPI → デフォルト症例にプロキシ）
# =====================================================
@app.get("/images")
def compat_list_images():
    """iPad互換: GET /images → 最初の症例の画像一覧"""
    case_id = _get_default_case_id()
    if case_id is None:
        return {"images": []}
    return {"images": dm.list_images(case_id)}


@app.get("/images/{image_id}/download")
def compat_download_image(image_id: str):
    """iPad互換: GET /images/{id}/download"""
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})
    case_id = _get_default_case_id()
    if case_id is None:
        return JSONResponse(status_code=404, content={"error": "no cases available"})
    path = dm.get_image_path(case_id, image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "image not found"})
    return FileResponse(path, media_type="image/png")


@app.get("/labels/{image_id}/download")
def compat_download_label(image_id: str):
    """iPad互換: GET /labels/{id}/download"""
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})
    case_id = _get_default_case_id()
    if case_id is None:
        return JSONResponse(status_code=404, content={"error": "no cases available"})
    path = dm.get_annotation_path(case_id, image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "label not found"})
    return FileResponse(path, media_type="image/png")


@app.put("/submit/{image_id}")
async def compat_submit_label(image_id: str, file: UploadFile = File(...)):
    """iPad互換: PUT /submit/{id}"""
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})
    case_id = _get_default_case_id()
    if case_id is None:
        return JSONResponse(status_code=404, content={"error": "no cases available"})
    content = await file.read()
    try:
        dm.save_annotation(case_id, image_id, content)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return {"status": "saved", "image_id": image_id}


@app.get("/next")
def compat_get_next(strategy: str = "sequential"):
    """iPad互換: GET /next"""
    case_id = _get_default_case_id()
    if case_id is None:
        return {"image_id": None, "message": "no cases available"}
    image_id = dm.get_next_unlabeled(case_id, strategy=strategy)
    if image_id is None:
        return {"image_id": None, "message": "all slices labeled"}
    return {"image_id": image_id}


@app.get("/label_config")
def compat_get_label_config():
    """iPad互換: GET /label_config → デフォルト症例のクラス定義"""
    case_id = _get_default_case_id()
    if case_id is None:
        return JSONResponse(status_code=404, content={"error": "no cases available"})
    config = dm.get_label_config(case_id)
    if config is None:
        return JSONResponse(status_code=404, content={"error": "label_config not found"})
    return config


# =====================================================
# GET /cases - 症例一覧
# =====================================================
@app.get("/cases")
def list_cases():
    return {"cases": dm.list_cases()}


# =====================================================
# GET /cases/{case_id}/images - 症例内の画像一覧
# =====================================================
@app.get("/cases/{case_id}/images")
def list_case_images(case_id: str):
    if not validate_case_id(case_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id"})
    if not dm.case_exists(case_id):
        return JSONResponse(status_code=404, content={"error": "case not found"})
    return {"case_id": case_id, "images": dm.list_images(case_id)}


# =====================================================
# GET /cases/{case_id}/images/{image_id}/download - 画像ダウンロード
# =====================================================
@app.get("/cases/{case_id}/images/{image_id}/download")
def download_image(case_id: str, image_id: str):
    if not validate_case_id(case_id) or not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id or image_id"})

    path = dm.get_image_path(case_id, image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "image not found"})

    return FileResponse(path, media_type="image/png")


# =====================================================
# PUT /cases/{case_id}/submit/{image_id} - マスクアップロード
# =====================================================
@app.put("/cases/{case_id}/submit/{image_id}")
async def submit_label(case_id: str, image_id: str, file: UploadFile = File(...)):
    if not validate_case_id(case_id) or not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id or image_id"})

    content = await file.read()
    try:
        dm.save_annotation(case_id, image_id, content)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"status": "saved", "case_id": case_id, "image_id": image_id}


# =====================================================
# GET /cases/{case_id}/labels/{image_id}/download - マスクダウンロード
# =====================================================
@app.get("/cases/{case_id}/labels/{image_id}/download")
def download_label(case_id: str, image_id: str):
    if not validate_case_id(case_id) or not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id or image_id"})

    path = dm.get_annotation_path(case_id, image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "label not found"})

    return FileResponse(path, media_type="image/png")


# =====================================================
# GET /cases/{case_id}/next - 次の未ラベルスライス
# =====================================================
@app.get("/cases/{case_id}/next")
def get_next_sample(case_id: str, strategy: str = "sequential"):
    if not validate_case_id(case_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id"})

    image_id = dm.get_next_unlabeled(case_id, strategy=strategy)
    if image_id is None:
        return {"case_id": case_id, "image_id": None, "message": "all slices labeled"}
    return {"case_id": case_id, "image_id": image_id}


# =====================================================
# GET /cases/{case_id}/manifest - スライスmanifest取得
# =====================================================
@app.get("/cases/{case_id}/manifest")
def get_manifest(case_id: str):
    if not validate_case_id(case_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id"})

    manifest = dm.get_manifest(case_id)
    if manifest is None:
        return JSONResponse(status_code=404, content={"error": "manifest not found"})
    return manifest


# =====================================================
# GET /cases/{case_id}/label_config - クラス定義取得
# =====================================================
@app.get("/cases/{case_id}/label_config")
def get_label_config(case_id: str):
    if not validate_case_id(case_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id"})

    config = dm.get_label_config(case_id)
    if config is None:
        return JSONResponse(status_code=404, content={"error": "label_config not found"})
    return config


# =====================================================
# POST /cases/{case_id}/reconstruct - NIfTIラベルマップ再統合
# =====================================================
@app.post("/cases/{case_id}/reconstruct")
def reconstruct_nifti(case_id: str):
    if not validate_case_id(case_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id"})
    if not dm.case_exists(case_id):
        return JSONResponse(status_code=404, content={"error": "case not found"})

    try:
        from medical_adapter.reconstructor import reconstruct_label_volume
        result = reconstruct_label_volume(case_id)
        logger.info(f"NIfTI再統合完了: {case_id} → {result['output_path']}")
        return result
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.error(f"NIfTI再統合エラー: {case_id} - {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# POST /infer/{case_id}/{image_id} - 推論実行
# =====================================================
@app.post("/infer/{case_id}/{image_id}")
def infer(case_id: str, image_id: str):
    if not validate_case_id(case_id) or not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid case_id or image_id"})

    image_path = dm.get_image_path(case_id, image_id)
    if image_path is None:
        return JSONResponse(status_code=404, content={"error": "image not found"})

    if not os.path.exists(BEST_MODEL_PATH):
        return JSONResponse(
            status_code=503,
            content={"error": "model not available, train first"},
        )

    try:
        from inference import run_inference
        mask_bytes = run_inference(image_path, BEST_MODEL_PATH)
        if mask_bytes is None:
            return JSONResponse(
                status_code=503,
                content={"error": "model not available, train first"},
            )
        logger.info(f"推論完了: {case_id}/{image_id}")
        return Response(content=mask_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"推論エラー: {case_id}/{image_id} - {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# POST /train - バックグラウンド学習開始
# =====================================================
@app.post("/train")
def start_training(background_tasks: BackgroundTasks, max_epochs: int = DEFAULT_MAX_EPOCHS):
    training_pairs = dm.get_all_training_pairs()
    if len(training_pairs) < MIN_IMAGES_FOR_TRAINING:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"ラベル付き画像が{len(training_pairs)}枚しかありません。"
                f"最低{MIN_IMAGES_FOR_TRAINING}枚必要です。"
            },
        )

    with training_lock:
        if training_status["status"] == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "training already running"},
            )
        training_status["status"] = "running"
        training_status["epoch"] = 0
        training_status["max_epochs"] = N_FOLDS * max_epochs
        training_status["best_dice"] = 0.0
        training_status["current_fold"] = 0
        training_status["n_folds"] = N_FOLDS
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status.pop("error", None)

    training_cancel_event.clear()
    logger.info(f"学習開始: max_epochs={max_epochs}, ペア数={len(training_pairs)}")
    background_tasks.add_task(run_training_task, training_pairs, max_epochs)
    return {"status": "started", "max_epochs": max_epochs, "training_pairs": len(training_pairs)}


def run_training_task(training_pairs: list[tuple[str, str]], max_epochs: int):
    """バックグラウンド学習タスク"""
    from trainer import train_model, TrainingCancelled
    try:
        best_dice, version = train_model(
            training_pairs=training_pairs,
            model_save_path=BEST_MODEL_PATH,
            max_epochs=max_epochs,
            status_callback=update_training_status,
            cancel_event=training_cancel_event,
        )
        with training_lock:
            training_status["status"] = "completed"
            training_status["best_dice"] = best_dice
            training_status["version"] = version
            training_status["completed_at"] = datetime.now().isoformat()
        logger.info(f"学習完了: best_dice={best_dice:.4f}, version={version}")
    except TrainingCancelled:
        with training_lock:
            training_status["status"] = "cancelled"
            training_status["completed_at"] = datetime.now().isoformat()
        logger.info("学習キャンセル完了")
    except Exception as e:
        with training_lock:
            training_status["status"] = "error"
            training_status["error"] = str(e)
            training_status["completed_at"] = datetime.now().isoformat()
        logger.error(f"学習エラー: {e}")


def update_training_status(epoch: int, dice: float, fold_idx: int = 0):
    """学習中にepochごとに呼ばれるコールバック"""
    with training_lock:
        training_status["epoch"] = epoch
        training_status["best_dice"] = max(training_status["best_dice"], dice)
        training_status["current_fold"] = fold_idx


# =====================================================
# POST /train/cancel - 学習キャンセル
# =====================================================
@app.post("/train/cancel")
def cancel_training():
    with training_lock:
        if training_status["status"] != "running":
            return JSONResponse(
                status_code=409,
                content={"error": "training is not running"},
            )
    training_cancel_event.set()
    logger.info("学習キャンセル要求受信")
    return {"status": "cancelling"}


# =====================================================
# GET /status - 学習ステータス
# =====================================================
@app.get("/status")
def get_training_status():
    return training_status


# =====================================================
# GET /models/latest - CoreMLモデルダウンロード
# =====================================================
@app.get("/models/latest")
def download_latest_model():
    """CoreML .mlpackage を ZIP 圧縮して配信"""
    import shutil
    import tempfile

    if not os.path.exists(COREML_PATH):
        return JSONResponse(
            status_code=404,
            content={"error": "CoreML model not available, convert first"},
        )

    try:
        tmp_dir = tempfile.mkdtemp()
        zip_base = os.path.join(tmp_dir, "SegmentationModel.mlpackage")
        zip_path = shutil.make_archive(zip_base, "zip", COREML_PATH)
        logger.info(f"CoreMLモデル配信: {zip_path}")
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="SegmentationModel.mlpackage.zip",
        )
    except Exception as e:
        logger.error(f"CoreMLモデル配信エラー: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# GET /models/versions - 全バージョンのサマリーリスト
# =====================================================
@app.get("/models/versions")
def list_model_versions():
    from version_manager import list_all_versions
    return {"versions": list_all_versions()}


# =====================================================
# POST /models/versions/{version}/restore - バージョン復元
# =====================================================
@app.post("/models/versions/{version}/restore")
def restore_model_version(version: str):
    with training_lock:
        if training_status["status"] == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "training in progress, wait for completion"},
            )

    from version_manager import restore_version
    try:
        restore_version(version)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    logger.info(f"モデル復元完了: {version}")
    return {"status": "restored", "version": version}


# =====================================================
# POST /models/convert - CoreML変換実行
# =====================================================
@app.post("/models/convert")
def start_conversion(background_tasks: BackgroundTasks):
    if not os.path.exists(BEST_MODEL_PATH):
        return JSONResponse(
            status_code=404,
            content={"error": "PyTorch model not found, train first"},
        )

    if training_status["status"] == "running":
        return JSONResponse(
            status_code=409,
            content={"error": "training in progress, wait for completion"},
        )

    background_tasks.add_task(run_conversion_task)
    logger.info("CoreML変換開始")
    return {"status": "conversion_started"}


def run_conversion_task():
    try:
        from convert_coreml import convert_to_coreml
        convert_to_coreml()
        logger.info("CoreML変換完了")
    except Exception as e:
        logger.error(f"CoreML変換エラー: {e}")


# =====================================================
# 静的ファイル配信（将来のWebフロントエンド用）
# =====================================================
if os.path.isdir(STATIC_DIR) and os.listdir(STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/web", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    logger.info(f"静的ファイル配信有効: /web → {STATIC_DIR}")


# =====================================================
# エントリーポイント
# =====================================================
if __name__ == "__main__":
    logger.info(f"サーバー起動: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"モデルパス: {BEST_MODEL_PATH}")
    stats = dm.get_stats()
    logger.info(
        f"データ統計: {stats['total_cases']}症例, "
        f"{stats['total_slices']}スライス "
        f"(labeled={stats['labeled_slices']})"
    )
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
