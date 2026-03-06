# Retinal HIL Server

FastAPI server for Human-in-the-Loop (HIL) active learning. Works with the Annotty HIL iPad app to enable an iterative annotation → training → model delivery cycle for retinal vessel segmentation.

## System Overview

```
iPad (Annotty HIL)                Server (this directory)
  |                                 |
  |-- GET /images ---------------->| List unannotated images
  |-- GET /images/{id}/download -->| Download image
  |-- PUT /submit/{id} ---------->| Save annotation mask
  |-- POST /train --------------->| Start background training
  |-- POST /train/cancel -------->| Cancel training
  |-- GET /status --------------->| Poll training progress
  |-- GET /models/latest -------->| Download CoreML model
```

> **Note:** Inference (`POST /infer/{id}`) is available on the server but the iPad app uses on-device CoreML inference by default for faster predictions.

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for training)
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/) (`cloudflared`)

### Installation

```bash
cd server/
pip install -r requirements.txt
```

### Start Server

```bash
# Start server + Cloudflare quick tunnel
python main.py
```

This will:
1. Start the FastAPI server on `http://0.0.0.0:8000`
2. Launch a Cloudflare quick tunnel
3. Print the public `https://xxxx.trycloudflare.com` URL — enter this in the iPad app

#### Local network only (no tunnel)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Directory Structure

```
server/
├── main.py                # FastAPI server (entry point)
├── config.py              # Paths & hyperparameters
├── model.py               # U-Net model definition (smp + ResNet34)
├── data_manager.py        # Data access layer
├── trainer.py             # 5-fold CV training worker
├── inference.py           # 5-fold ensemble inference
├── convert_coreml.py      # PyTorch → CoreML conversion
├── version_manager.py     # Model version management
├── requirements.txt       # Dependencies
├── scripts/
│   ├── import_images.py       # Image import script
│   ├── generate_dummy_data.py # Generate dummy test data
│   ├── migrate_data.py        # Data migration script
│   └── test_api.py            # API test script
└── data/                      # ⚠️ Not tracked in git
    ├── images_completed/      # Annotated data (read-only)
    │   ├── images/
    │   └── annotations/
    ├── images_unannotated/    # Images for annotation (iPad writes here)
    │   ├── images/
    │   └── annotations/
    └── models/
        ├── pytorch/
        │   ├── pretrained.pt      # Pre-trained model
        │   ├── current_pt/        # Active model for inference
        │   │   ├── fold_0..4.pt   # 5-fold models (ensemble)
        │   │   └── best.pt        # Best fold (CoreML conversion)
        │   └── versions/          # Version archive
        │       ├── v001/
        │       ├── v002/
        │       └── ...
        └── coreml/
            └── SegmentationModel.mlpackage
```

### Data Separation Design

| Directory | Purpose | Write Access |
|---|---|---|
| `images_completed/` | Completed annotations. Used for training. | Read-only |
| `images_unannotated/` | Images for iPad annotation | iPad writes via `PUT /submit` |

The server never writes to `images_completed/`, preventing accidental overwrites of completed annotations.

## API Endpoints

### Images & Annotations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/info` | Server info (image count, label count, training status) |
| `GET` | `/images` | List unannotated images |
| `GET` | `/images/{image_id}/download` | Download image |
| `GET` | `/labels/{image_id}/download` | Download annotation mask |
| `GET` | `/next?strategy=random` | Get next unlabeled image ID |
| `POST` | `/infer/{image_id}` | Run inference (returns red mask PNG) |
| `PUT` | `/submit/{image_id}` | Upload annotation mask |

### Training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/train?max_epochs=50` | Start background training (5-fold CV) |
| `POST` | `/train/cancel` | Cancel training |
| `GET` | `/status` | Training status (epoch, dice, fold, version) |

### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/models/latest` | Download CoreML model (ZIP) |
| `POST` | `/models/convert` | Run PyTorch → CoreML conversion |
| `GET` | `/models/versions` | List all version summaries |
| `POST` | `/models/versions/{version}/restore` | Restore a specific version (rollback) |

## Model Version Management

### Overview

Each training run automatically saves model files and training logs to `versions/v{NNN}/`. Inference always uses the models in `current_pt/`. Version directories serve as archives.

### Training Auto-Versioning Flow

1. `POST /train` → `get_next_version()` determines the next version number
2. Each epoch records `train_loss` and `val_dice`
3. After training, models from `current_pt/` are copied to the version directory
4. `training_log.json` is written
5. On cancel/error, a partial version is saved (status: `"cancelled"` / `"error"`)

### List Versions

```bash
curl http://localhost:8000/models/versions
```

### Rollback to a Previous Version

```bash
curl -X POST http://localhost:8000/models/versions/v001/restore
```

## Model Architecture

- **Base model**: U-Net ([segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch))
- **Encoder**: ResNet34 (ImageNet pre-trained)
- **Input**: 512x512 RGB
- **Output**: 512x512 1ch (vessel segmentation mask)
- **Loss**: DiceBCELoss (Dice Loss + Binary Cross Entropy)
- **Training**: 5-fold Cross Validation, AdamW + CosineAnnealing
- **Inference**: 5-fold ensemble (average predictions across folds)

## Key Parameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 512 | Input image size |
| `BATCH_SIZE` | 4 | Training batch size |
| `DEFAULT_MAX_EPOCHS` | 50 | Max epochs per fold |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `WEIGHT_DECAY` | 1e-5 | L2 regularization |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `MIN_IMAGES_FOR_TRAINING` | 2 | Minimum images required to start training |
