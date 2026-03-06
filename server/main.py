"""
FastAPI サーバー
眼底血管セグメンテーション HIL アノテーションシステム
annotation → train → deploy のHILループを実現
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

app = FastAPI(title="Retinal HIL Server")

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
def validate_image_id(image_id: str) -> bool:
    """ファイル名として安全かチェック"""
    if not re.match(r"^[\w\-\.]+\.png$", image_id):
        return False
    if ".." in image_id or "/" in image_id or "\\" in image_id:
        return False
    return True


# =====================================================
# GET /info - サーバー情報
# =====================================================
@app.get("/info")
def get_info():
    stats = dm.get_stats()
    return {
        "name": "Retinal HIL Server",
        "total_images": stats["unannotated_images"],
        "labeled_images": stats["unannotated_labeled"],
        "unlabeled_images": stats["unannotated_unlabeled"],
        "model_loaded": os.path.exists(BEST_MODEL_PATH),
        "training_status": training_status["status"],
        # 追加フィールド（後方互換: iPadは無視可能）
        "completed_images": stats["completed_images"],
        "completed_annotations": stats["completed_annotations"],
        "total_training_pairs": stats["total_training_pairs"],
    }


# =====================================================
# GET /images - 画像一覧（unannotated のみ）
# =====================================================
@app.get("/images")
def list_images():
    result = dm.list_unannotated_images()
    return {"images": result}


# =====================================================
# GET /images/{image_id}/download - 画像ダウンロード（unannotated のみ）
# =====================================================
@app.get("/images/{image_id}/download")
def download_image(image_id: str):
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})

    path = dm.get_unannotated_image_path(image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "image not found"})

    return FileResponse(path, media_type="image/png")


# =====================================================
# POST /infer/{image_id} - 推論実行（completed + unannotated 両方検索）
# =====================================================
@app.post("/infer/{image_id}")
def infer(image_id: str):
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})

    image_path = dm.get_image_path(image_id)
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
        logger.info(f"推論完了: {image_id}")
        return Response(content=mask_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"推論エラー: {image_id} - {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# PUT /submit/{image_id} - マスクアップロード（unannotated のみ）
# =====================================================
@app.put("/submit/{image_id}")
async def submit_label(image_id: str, file: UploadFile = File(...)):
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})

    content = await file.read()
    try:
        dm.save_annotation(image_id, content)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"status": "saved", "image_id": image_id}


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
    """学習中にepochごとに呼ばれるコールバック（global epoch + fold情報）"""
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
# GET /next - 次の未ラベル画像を返す（unannotated のみ）
# =====================================================
@app.get("/next")
def get_next_sample(strategy: str = "random"):
    image_id = dm.get_next_unlabeled(strategy=strategy)
    if image_id is None:
        return {"image_id": None, "message": "all images labeled"}
    return {"image_id": image_id}


# =====================================================
# GET /labels/{image_id}/download - 確定済みマスクダウンロード
# =====================================================
@app.get("/labels/{image_id}/download")
def download_label(image_id: str):
    if not validate_image_id(image_id):
        return JSONResponse(status_code=400, content={"error": "invalid image_id"})

    path = dm.get_annotation_path(image_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "label not found"})

    return FileResponse(path, media_type="image/png")


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
    """指定バージョンのモデルを current_pt/ に復元"""
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
    """バックグラウンドで PyTorch → CoreML 変換を実行"""
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
    """バックグラウンドCoreML変換タスク"""
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
        f"データ統計: completed={stats['completed_images']}枚, "
        f"unannotated={stats['unannotated_images']}枚 "
        f"(labeled={stats['unannotated_labeled']})"
    )
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
