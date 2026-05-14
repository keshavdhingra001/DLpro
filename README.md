# FaceRestore — Face Masking & Reconstruction System

A deep learning-based face reconstruction system that learns to reconstruct missing parts of faces using partial convolutions and self-attention mechanisms.

## Overview

FaceRestore detects faces in uploaded images, allows users to mask specific facial regions (eyes, nose, mouth, etc.), and uses a trained autoencoder to intelligently reconstruct the masked areas. The system combines partial convolution layers with self-attention to achieve coherent facial reconstruction.

## Features

- **Face Detection** — Automatic face detection using OpenCV Haar cascades
- **Region-Based Masking** — One-click masking for specific facial regions (eyes, nose, mouth, forehead, cheeks, chin)
- **Freehand Masking** — Manual brush tool with adjustable size and mask/erase modes
- **Face Reconstruction** — Autoencoder fills masked regions using learned facial priors
- **Before/After Comparison** — Interactive slider to compare original and reconstructed images
- **Undo/Redo** — Full history support for mask editing
- **Drag & Drop Upload** — Upload images via drag-and-drop or file picker
- **Random Samples** — Pre-loaded sample face images for quick testing

## Architecture

### Model
- **Encoder-Decoder U-Net** with skip connections
- **Partial Convolution layers** — mask-aware convolutions that only operate on valid (unmasked) pixels, with automatic mask renormalization ([Liu et al., 2018](https://arxiv.org/abs/1804.07723))
- **Self-Attention layer** at the bottleneck — enables the model to reason about global facial structure and distant spatial relationships ([Zhang et al., 2018](https://arxiv.org/abs/1805.08318))
- **VGG-16 Perceptual Loss** — ensures reconstructed regions are perceptually similar to ground truth
- **Style Loss** — maintains texture consistency in reconstructed areas

### Training
- **Dataset:** CelebA (202,599 aligned celebrity face images, preprocessed to 256×256)
- **Masking Strategy:** Random masks generated from QuickDraw stroke data
- **Optimization:** Adam optimizer with multi-step learning rate scheduling
- **Fine-tuning:** Transfer learning from pretrained partial convolution weights

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

## Project Structure

```
DLpro/
├── inpaint/                      # Main application
│   ├── app/
│   │   ├── app.py                # Flask backend (API routes)
│   │   ├── face_detect.py        # Face detection & region masking
│   │   ├── data.py               # MongoDB storage layer
│   │   ├── static/
│   │   │   ├── main.js           # Frontend (canvas, masking, API calls)
│   │   │   ├── style.css         # Dark theme UI
│   │   │   ├── models/           # Trained model weights
│   │   │   └── random_images/    # Sample face images
│   │   └── templates/
│   │       └── index.html        # Main UI template
│   ├── inpaint/
│   │   ├── module.py             # Neural network (PConv + Attention)
│   │   ├── inpainter.py          # Inference wrapper
│   │   └── data.py               # Training data loader & mask generator
│   ├── train.py                  # Training script
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── dataset_prep/                 # Training pipeline
│   ├── preprocess.py             # CelebA preprocessing (crop to 256×256)
│   ├── inpaint_training.zip      # Training code package for Colab
│   └── FaceRestore_Training.ipynb # Colab training notebook
└── .gitignore
```

## Setup & Running

### Prerequisites
- Docker & Docker Compose
- Python 3.10+

### Quick Start

```bash
cd inpaint
docker-compose up --build
```

Open **http://localhost:8003** in your browser.

### Services

| Service | Port | Purpose |
|---|---|---|
| App | 8003 | Main web application |
| Admin Mongo | 8004 | Database management UI |
| Prometheus | 9090 | Metrics monitoring |

## Usage

1. **Upload an image** — drag & drop or click the upload button
2. **Detect face** — click "Detect Face" to automatically find faces
3. **Mask a region** — click a region button (eyes, nose, etc.) or use the brush tool
4. **Reconstruct** — click "Inpaint" to fill the masked region
5. **Compare** — use the before/after slider to see the result

## Training Pipeline

The training pipeline is designed for Google Colab (free tier T4 GPU):

1. **Preprocess locally:** Run `dataset_prep/preprocess.py` to crop CelebA images to 256×256
2. **Upload to Colab:** Upload the preprocessed zip and training code
3. **Train:** Run the Colab notebook with tqdm progress bars and W&B logging
4. **Deploy:** Download the trained checkpoint and replace `model.state_dict`

See `dataset_prep/FaceRestore_Training.ipynb` for the complete training notebook.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main UI |
| GET | `/random` | Load a random sample image |
| POST | `/add_image` | Upload a new image |
| POST | `/apply_mask` | Apply mask and run reconstruction |
| POST | `/detect_face` | Detect faces in the current image |

## Requirements

Key dependencies (see `requirements.txt` for full list):
- PyTorch >= 1.13
- torchvision >= 0.14
- Flask >= 2.0
- OpenCV (headless) >= 4.5
- NumPy, Pillow, matplotlib
