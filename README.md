# FaceRestore — Face Masking & Reconstruction System

A deep learning-based face reconstruction system that learns to reconstruct missing parts of faces using partial convolutions and self-attention mechanisms. Built with PyTorch, served with Flask + Docker.

## Features

- **Face Detection** — Automatic face detection using OpenCV Haar cascades
- **Region-Based Masking** — One-click masking for eyes, nose, mouth, forehead, cheeks, chin
- **Freehand Masking** — Manual brush tool with adjustable size and mask/erase modes
- **Face Reconstruction** — Partial convolution autoencoder fills masked regions using learned facial priors
- **Self-Attention** — Global attention at the U-Net bottleneck for facial symmetry and coherence
- **Before/After Comparison** — Interactive slider to compare original and reconstructed images
- **Undo/Redo** — Full history support for mask editing
- **Drag & Drop Upload** — Upload images via drag-and-drop or file picker

## Architecture

| Component | Details |
|---|---|
| **Encoder-Decoder** | U-Net with 7 down/up blocks and skip connections |
| **Partial Convolutions** | Mask-aware convolutions with automatic renormalization ([Liu et al., 2018](https://arxiv.org/abs/1804.07723)) |
| **Self-Attention** | SA-GAN style attention at the bottleneck ([Zhang et al., 2018](https://arxiv.org/abs/1805.08318)) |
| **Loss** | VGG-16 perceptual loss + style loss + L1 reconstruction loss |
| **Dataset** | CelebA — 202,599 aligned celebrity faces, preprocessed to 256×256 |
| **Masking** | Random masks from QuickDraw stroke data |

## Project Structure

```
DLpro/
├── inpaint/                          # Main application
│   ├── app/
│   │   ├── app.py                    # Flask backend & API routes
│   │   ├── face_detect.py            # OpenCV Haar cascade face detection
│   │   ├── data.py                   # MongoDB storage layer
│   │   ├── metrics.py                # Prometheus monitoring
│   │   ├── static/
│   │   │   ├── main.js               # Frontend canvas, masking, API
│   │   │   ├── style.css             # Dark theme UI
│   │   │   ├── models/               # Trained model weights
│   │   │   └── random_images/        # 26 sample face images
│   │   └── templates/index.html      # Main UI
│   ├── inpaint/
│   │   ├── module.py                 # PConv + Attention U-Net
│   │   ├── inpainter.py              # Inference wrapper
│   │   ├── data.py                   # DataLoader & mask generator
│   │   └── __init__.py
│   ├── train.py                      # Training script
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── dataset_prep/                     # Training pipeline
│   ├── preprocess.py                 # CelebA crop to 256×256
│   ├── training_package/             # Code packaged for Colab
│   ├── inpaint_training.zip          # Zipped training code
│   └── FaceRestore_Training.ipynb    # Colab notebook
├── README.md
└── .gitignore
```

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | PyTorch |
| Backend | Flask + Gunicorn |
| Frontend | Vanilla JavaScript + CSS |
| Face Detection | OpenCV (Haar cascades) |
| Database | MongoDB |
| Deployment | Docker + Docker Compose |
| Training | Google Colab (T4 GPU) |
| Experiment Tracking | Weights & Biases |

---

## Quick Start — Running the App

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for preprocessing only)

### Run

```bash
cd inpaint
docker-compose up --build
```

Open **http://localhost:8003** in your browser.

### Services

| Service | Port | Purpose |
|---|---|---|
| App | 8003 | FaceRestore web application |
| Admin Mongo | 8004 | Database management UI |
| Prometheus | 9090 | Metrics & monitoring |

### Usage

1. **Upload** — drag & drop a face image or click Upload
2. **Detect Face** — click "Detect Face" to find faces automatically
3. **Mask** — click a region button (eyes, nose, etc.) or draw with the brush
4. **Reconstruct** — click "Inpaint" to fill the masked region
5. **Compare** — use the before/after slider

---

## Training the Model

The training pipeline is split into two phases: local preprocessing (one-time) and cloud training (Google Colab, repeatable).

### Phase 1: Local Preprocessing

**Step 1 — Download CelebA**

1. Go to https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Download `archive.zip` (~1.4 GB)
3. Extract it into `dataset_prep/` so images are at `dataset_prep/archive/img_align_celeba/`

**Step 2 — Crop images to 256×256**

```bash
cd dataset_prep
pip install Pillow tqdm
python preprocess.py
```

This uses all CPU cores to crop 202,599 images and creates `celeba_256.zip`. Takes ~5-10 minutes.

### Phase 2: Colab Training

**Step 3 — Upload to Google Drive**

Upload these 3 files to the **root** of your Google Drive:

| File | Location | Size |
|---|---|---|
| `celeba_256.zip` | `dataset_prep/celeba_256.zip` | ~2-4 GB |
| `inpaint_training.zip` | `dataset_prep/inpaint_training.zip` | ~8 KB |
| `model.state_dict` | `inpaint/app/static/models/model.state_dict` | ~103 MB |

> **Alternative:** You can skip Google Drive and upload files directly to Colab via the sidebar file browser. The tradeoff is that you'll need to re-upload if the session disconnects.

**Step 4 — Open Colab**

1. Go to https://colab.research.google.com
2. Upload `dataset_prep/FaceRestore_Training.ipynb`
3. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)

**Step 5 — Run training cells**

| Cell | What it does |
|---|---|
| Cell 1 | Installs dependencies, sets up W&B logging |
| Cell 2 | Unzips dataset, downloads QuickDraw masks |
| Cell 3 | Verifies GPU availability |
| Cell 4 | Runs training with tqdm progress bars |

The notebook auto-detects whether to start fresh or resume from a checkpoint.

**Step 6 — Anti-idle protection**

Free Colab disconnects after 90 minutes of inactivity. Paste this into the browser console (F12 → Console):

```javascript
function KeepClicking(){
  console.log('Keeping Colab alive...');
  document.querySelector('colab-connect-button')
    .shadowRoot.getElementById('connect').click();
}
setInterval(KeepClicking, 60000);
```

**Step 7 — Monitor training**

- **Terminal:** tqdm progress bar shows per-batch loss in real-time
- **W&B Dashboard:** Loss curves, learning rate, and epoch metrics at https://wandb.ai (persists even if Colab crashes)

### Crash Recovery

Colab sessions are unreliable. The pipeline is designed for this:

| Threat | Protection |
|---|---|
| Session crash | Checkpoints saved after every epoch |
| Idle timeout | JavaScript keep-alive script |
| Time limit | Auto-resume from latest checkpoint |
| Lost logs | W&B cloud logging survives any crash |

If Colab disconnects, just open a new session and run Cells 1-4 again. Training automatically resumes from the last checkpoint.

### Deploy Trained Model

1. Download `checkpoints/last.pth` from Colab
2. Rename to `model.state_dict`
3. Replace `inpaint/app/static/models/model.state_dict`
4. Restart Docker:

```bash
cd inpaint
docker-compose restart
```

The app now uses your fine-tuned model.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main UI |
| GET | `/random` | Load a random sample image |
| POST | `/add_image` | Upload a new image |
| POST | `/apply_mask` | Apply mask and run reconstruction |
| POST | `/detect_face` | Detect faces in the current image |

## Key Dependencies

- PyTorch >= 1.13
- torchvision >= 0.14
- Flask >= 2.0
- OpenCV (headless) >= 4.5
- NumPy, Pillow, matplotlib
- W&B (optional, for training logs)

See `inpaint/requirements.txt` for the full list.
