# FaceRestore Kaggle Training Guide — Single GPU

> [!IMPORTANT]
> This guide covers training on **Kaggle** with a single T4 GPU.
> For Google Colab training, see `training_guide.md`.

> [!NOTE]
> **Why not DataParallel / AMP?** The partial convolution model uses custom
> mask operations and skip connections that are incompatible with both
> `nn.DataParallel` and mixed precision (AMP). We use `CUDA_VISIBLE_DEVICES=0`
> to force single GPU mode on Kaggle's T4 x2 hardware.

---

## Table of Contents

1. [Why Kaggle?](#why-kaggle-over-colab)
2. [How Kaggle GPU Quota Works](#how-kaggle-gpu-quota-works)
3. [Multi-Account Hopping](#multi-account-hopping-strategy)
4. [PHASE 1: One-Time Setup](#phase-1-one-time-setup-15-min)
5. [PHASE 2: Every Training Session](#phase-2-every-training-session)
6. [PHASE 3: After Session Ends](#phase-3-after-session-ends-or-times-out)
7. [PHASE 4: Deploying the Trained Model](#phase-4-using-the-trained-model-locally)
8. [Troubleshooting](#troubleshooting)
9. [Quick Reference](#quick-reference)

---

## Why Kaggle Over Colab?

| | Colab Free | Kaggle |
|---|---|---|
| GPU time per session | ~3.5 hrs | **12 hrs** |
| Weekly GPU budget | ~1 epoch/Gmail/day | **30 GPU-hrs/week** |
| GPUs used | 1x T4 | 1x T4 (of the 2 available) |
| Epochs per session | ~1 | **~3** |
| Anti-idle hack needed | Yes | **No** |
| Batch size | 16 | 16 |
| Time per epoch | ~3.5 hrs | ~3.5 hrs |
| Can close browser? | No | **Yes** |
| Checkpoints persist? | Via Drive | **Via Output tab (Save & Run All)** |

**Main advantage:** Longer sessions, more weekly hours, background execution, no anti-idle.

---

## How Kaggle GPU Quota Works

> [!WARNING]
> **Key numbers to remember:**
> - **12 hours** = maximum time for a single notebook run
> - **30 GPU-hours** = total weekly budget
> - T4 x2 may burn **2 GPU-hrs per wall-hour** (charged per accelerator selection)
> - So one full 12hr session may use **~24 GPU-hrs** for the week
> - Quota resets on a **rolling 7-day window**
> - Check quota: **kaggle.com → avatar → Settings → GPU Quota**

### What does this mean in practice?

At ~3.5 hrs/epoch:
- One 12hr session = **~3 completed epochs**
- You may have ~6 GPU-hrs left for a short second session (~1 more epoch)
- **Total per week per account: ~3-4 epochs**
- With account hopping: multiply by number of accounts

If the session times out mid-epoch, that incomplete epoch is lost. All completed epochs are safe in the output.

---

## Multi-Account Hopping Strategy

Each Kaggle account gets its own 30 GPU-hrs/week. Hop between accounts to multiply throughput:

| Accounts | Weekly epochs | How |
|---|---|---|
| 1 account | ~3-4 | One 12hr session + maybe one short one |
| 2 accounts | ~6-8 | Alternate accounts |
| 3 accounts | ~9-12 | Rotate across 3 |

### How hopping works step-by-step:

```
Account 1: upload last.pth (epoch 6 from Colab)
            → trains epochs 7-9 (~3 epochs in 12hrs)
            → session ends
            ↓
Download last.pth from Account 1's Output tab
            ↓
Account 2: create/update facerestore-checkpoints dataset with new last.pth
            → trains epochs 10-12
            → session ends
            ↓
Download last.pth from Account 2's Output tab
            ↓
Account 1: quota has reset by now
            → upload new last.pth → trains epochs 13-15
```

**W&B tracks everything across all accounts.** Use the same W&B API key and project name (`facerestore-training`) on every account.

---

## PHASE 1: One-Time Setup (~15 min)

### Step 1: Prepare Your Files Locally

| File | Path on your machine | What it is |
|---|---|---|
| Training code | `d:\DLpro\dataset_prep\inpaint_training_kaggle.zip` | Model architecture + train.py |
| Face images | `d:\DLpro\dataset_prep\celeba_256.zip` | 202,599 preprocessed 256×256 faces |
| Checkpoint | `last.pth` from Google Drive `facerestore_checkpoints/` | Your trained weights |

### Step 2: Create 3 Kaggle Datasets

Go to **https://www.kaggle.com/datasets** → **"+ New Dataset"** for each:

1. Upload `celeba_256.zip` (or the folder of images)
2. Upload `inpaint_training_kaggle.zip`
3. Upload `last.pth`

**Dataset names don't matter** — the notebook searches by file content, not folder name.

### Step 3: Create the Notebook

1. Go to **https://www.kaggle.com/code** → **"+ New Notebook"**
2. **File → Import Notebook** → upload `FaceRestore_Training_Kaggle_Final.ipynb`

### Step 4: Configure Notebook Settings

- **Accelerator**: GPU T4 x2 (only option with T4; we force single GPU via code)
- **Internet**: ON
- **Add Data**: Attach all 3 datasets
- **Secrets**: Add `WANDB_API_KEY` (checkbox = on)

---

## PHASE 2: Every Training Session

### First Time: Test Interactively

Run **Cells 1-5** one by one. Check each output:

**Cell 1** should show:
```
W&B: Authenticated via Kaggle secret
Cell 1 done.
```

**Cell 2** should show:
```
Training code: /kaggle/input/.../inpaint-training-kaggle
Found 202599 face images in /kaggle/input/.../celeb-256
Linked 202599 images
Checkpoint: /kaggle/input/.../last.pth

ALL READY — proceed to Cell 3!
```

**Cell 3** should show:
```
CUDA available:  True
GPU 0: Tesla T4 (15.6 GB)
```

**Cell 4** should show:
```
Copied train.py to inpaint/
```

**Cell 5** — wait 2-3 minutes for:
```
Resumed from ... → continuing at epoch 7
Single GPU mode
Epoch 7 [train]:   0%|  | 16/12662 [00:16<3:11:26, loss=0.5251]
```

Once you see loss values (not NaN), training works.

### For Real Training: Save & Run All

1. **Stop the interactive session**
2. Click **"Save Version"** (top right)
3. Select **"Save & Run All (Commit)"**
4. Click **"Save"**
5. **Close the browser** — training runs in background
6. Come back in ~12 hours

> [!CAUTION]
> **Never train interactively for real runs!** If you accidentally click Stop
> or the session disconnects, all output is lost. "Save & Run All" preserves
> everything permanently.

---

## PHASE 3: After Session Ends (or Times Out)

### What happens when the 12hr limit hits?

1. The currently-running epoch is interrupted and lost
2. All completed epochs are safe in `/kaggle/working/facerestore_checkpoints/`
3. Output is permanently saved (because you used Save & Run All)

### Get Your Checkpoints

1. Go to your notebook page on Kaggle
2. Click the **"Output"** tab
3. Download `last.pth` (or the full zip)

### Prepare for Next Session

**Same account:**
1. Go to your checkpoints dataset → **"New Version"**
2. Upload the new `last.pth`
3. Open notebook → **Save & Run All** again

**Different account (hopping):**
1. Download `last.pth` from Output tab
2. Log into other Kaggle account
3. Create/update checkpoints dataset with new `last.pth`
4. Import same notebook, attach 3 datasets, set T4 x2 + Internet ON
5. **Save & Run All**

---

## PHASE 4: Using the Trained Model Locally

### Extract Model Weights

```python
import torch
data = torch.load('last.pth', map_location='cpu', weights_only=False)
torch.save(data['model'], 'model.state_dict')
```

### Deploy

```powershell
copy model.state_dict d:\DLpro\inpaint\app\static\models\model.state_dict
cd d:\DLpro\inpaint
docker-compose restart
```

Open `http://localhost:8003` — upgraded model is live.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No GPU | Settings → Accelerator → GPU T4 x2 |
| Face images not found | Add Data → attach your celeba dataset |
| Training code not found | Add Data → attach your training code dataset |
| Session stops, output lost | You ran interactively — always use **Save & Run All** |
| Can't find checkpoints after session | Notebook page → Output tab |
| W&B not logging | Check API key in Secrets. Not critical — TensorBoard still works |
| Wrong epoch number | Your last.pth is old format. Use `--start-epoch N` |

---

## Quick Reference

| What | Where |
|---|---|
| Kaggle notebook | `d:\DLpro\dataset_prep\FaceRestore_Training_Kaggle_Final.ipynb` |
| Kaggle training code | `d:\DLpro\dataset_prep\inpaint_training_kaggle.zip` |
| Colab notebook | `d:\DLpro\dataset_prep\FaceRestore_Training.ipynb` |
| Colab training code | `d:\DLpro\dataset_prep\inpaint_training.zip` |
| W&B dashboard | https://wandb.ai → `facerestore-training` |
| Kaggle GPU quota | kaggle.com → avatar → Settings → GPU Quota |

---

## Estimated Timeline (starting from epoch 6)

| Target | With 1 account | With 2 accounts | With 3 accounts |
|---|---|---|---|
| Epoch 10 | ~2 days | ~1 day | ~1 day |
| Epoch 15 | ~4 days | ~2 days | ~1-2 days |
| Epoch 20 | ~6 days | ~3 days | ~2 days |
| Epoch 30 | ~10 days | ~5 days | ~3-4 days |
