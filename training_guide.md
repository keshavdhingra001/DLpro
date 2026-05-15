# FaceRestore Training Guide — Complete Walkthrough

> [!IMPORTANT]
> All training files live in `d:\DLpro\dataset_prep\`. Your original project in `d:\DLpro\inpaint\` is completely untouched.

---

## How the Training Chain Works

Your model improves like this:

```
Original model.state_dict (103MB, from the repo author)
        │
        ▼  fine-tune for a few epochs
    last.pth saved to Google Drive (crash-proof)
        │
        ▼  Colab crashes or time runs out — no data lost
        │
        ▼  new session → Cell 4 AUTO-DETECTS last.pth on Drive
    last.pth updated with even better weights
        │
        ▼  repeat as many sessions as needed
        │
    Final model ← download and use locally
```

**Key concept:** The model NEVER starts from zero. Cell 4 automatically checks if a previous checkpoint exists on Drive and resumes from it. No manual editing needed between sessions.

---

## All Failsafes (Fully Automated)

| Threat | Protection | How it works |
|---|---|---|
| Colab crashes mid-training | Checkpoints on Google Drive | `--output-dir` points directly to Drive |
| Colab idle timeout (90 min) | JavaScript keep-alive | Script clicks Connect button every 60 sec |
| Colab hard time limit (~4-12 hrs) | Auto-resume | Cell 4 detects `last.pth` on Drive automatically |
| Training logs lost on crash | **W&B cloud logging** | Loss/lr/epoch stream to wandb.ai in real-time |
| Weak Colab CPU bottleneck | Pre-cropped 256×256 images | Zero CPU resizing during training |

### How W&B Logging Works

After every epoch, `train.py` sends your training metrics (loss, learning rate, epoch number) to Weights & Biases servers at **https://wandb.ai**. This happens over the internet in real-time.

Even if Colab crashes, your browser dies, or Google wipes the entire instance:
- Your loss curves are safely stored on W&B's servers
- You can view them from any device at any time
- They persist forever

### W&B Login (One-Time Setup Per Session)

Since you already have a W&B account:

1. Go to **https://wandb.ai/authorize** in your browser
2. Copy the API key shown on that page (long random string)
3. When Cell 1 runs in Colab, it will print:
   ```
   wandb: Paste an API key from your profile and hit enter:
   ```
4. Paste your API key and press Enter

That's it. After this, go to **https://wandb.ai** → look for a project called **`facerestore-training`** to see your live training charts.

> [!TIP]
> You need to paste the API key once per Colab session. If Colab disconnects and you start a new session, you'll paste it again in Cell 1.

---

## PHASE 1: Local Preprocessing (One-time, ~20 min)

### Step 1: Download CelebA from Kaggle

1. Go to: **https://www.kaggle.com/datasets/jessicali9530/celeba-dataset**
2. Log in to Kaggle (create free account if needed)
3. Click **Download** (top right)
4. Save `archive.zip` (~1.4 GB) into `d:\DLpro\dataset_prep\`

### Step 2: Extract the Zip

1. Right-click `archive.zip` in the `dataset_prep` folder
2. Click **Extract Here** (or Extract All)
3. You should see a folder called `archive` with image subfolders inside

> [!WARNING]
> Do NOT open the image folder in VS Code. 200,000 files will crash the editor.

### Step 3: Run the Preprocessing Script

Open a terminal in VS Code and run:

```powershell
cd d:\DLpro\dataset_prep
pip install Pillow tqdm
python preprocess.py
```

**What you will see:**
```
Found images in: archive\img_align_celeba\img_align_celeba
Found 202599 images to process.
Using 8 CPU threads...
Cropping: 100%|████████████████| 202599/202599 [03:45<00:00]
Zipping: 100%|████████████████| 202599/202599 [01:20<00:00]
DONE! Output: celeba_256.zip (1847 MB)
```

### Step 4: Upload 3 Files to Google Drive

Open **Google Drive** in your browser. Drag these 3 files into the **root folder** (My Drive):

| File | Where to find it | Size |
|---|---|---|
| `celeba_256.zip` | `d:\DLpro\dataset_prep\celeba_256.zip` | ~1.5-2 GB |
| `inpaint_training.zip` | `d:\DLpro\dataset_prep\inpaint_training.zip` | ~8 KB |
| `model.state_dict` | `d:\DLpro\inpaint\app\static\models\model.state_dict` | 103 MB |

**Upload time at 25 Mbps:** ~11-13 minutes. Wait for all 3 to finish.

> [!WARNING]
> Make sure all files are in the **root** of My Drive, NOT inside a subfolder.

---

## PHASE 2: Colab Training (Every Session — Same Steps)

### Step 5: Open the Colab Notebook

1. Go to **https://colab.research.google.com**
2. Click **File → Upload notebook**
3. Upload `d:\DLpro\dataset_prep\FaceRestore_Training.ipynb`
4. Go to **Runtime → Change runtime type → T4 GPU → Save**

### Step 6: Run Cells 1-3 (Setup)

| Cell | What it does | What you should see |
|---|---|---|
| Cell 1 | Mounts Drive, installs deps, sets up W&B | "Drive mounted + dependencies installed" |
| Cell 2 | Copies files from Drive, unzips, downloads masks | "All ready!" with "Face images: 202599" |
| Cell 3 | Verifies GPU | "GPU: Tesla T4" |

> [!CAUTION]
> If Cell 3 says "No GPU", go to Runtime → Change runtime type → T4 GPU.

### Step 7: Activate Anti-Idle Protection

**Before running Cell 4:**

1. Press **F12** → open DevTools
2. Click **Console** tab
3. Paste and press Enter:

```javascript
function KeepClicking(){
  console.log('Keeping Colab alive...');
  document.querySelector('colab-connect-button').shadowRoot.getElementById('connect').click();
}
setInterval(KeepClicking, 60000);
```

4. Minimize DevTools (don't close it)

### Step 8: Run Cell 4 (Training)

Click Play. You will see:

```
Auto-resume from: /content/model.state_dict       ← first time
Auto-resume from: /content/drive/.../last.pth      ← every time after

W&B cloud logging: ACTIVE (logs at https://wandb.ai)

Training config:
  Batch size: 16
  Workers:    2
  Epochs:     20
  LR:         2e-05

Epoch 1 [train]: 100%|████████████| 12662/12662 [14:23<00:00, loss=0.0342]
epoch=1 train_loss=0.034218 lr=2e-05
```

**You see a live progress bar for every batch. No more blind waiting.**

After each epoch:
- Checkpoint saved to Google Drive ✅
- Metrics logged to W&B cloud ✅
- `last.pth` updated for auto-resume ✅

---

## PHASE 3: If Things Go Wrong

### Colab Disconnects Mid-Training

**Nothing is lost.** Just:
1. Open a new Colab session
2. Upload the same notebook
3. Set runtime to T4 GPU
4. Run Cells 1-4 again (same steps as before)

Cell 4 automatically finds `last.pth` on your Drive and resumes. **No manual changes needed.**

### You Run Out of Free GPU Time

Wait 12-24 hours for quota to reset, then repeat Phase 2. The model continues from where it stopped.

### Training Loss Isn't Decreasing

Expected loss values:
- Epoch 1: ~0.03-0.04
- Epoch 5: ~0.02-0.03
- Epoch 10: ~0.01-0.02

If loss stays flat after 3 full epochs, stop and share the W&B dashboard link with me.

---

## PHASE 4: Using the Trained Model

1. Open Google Drive → `facerestore_checkpoints` folder
2. Download `last.pth`
3. Rename to `model.state_dict`
4. Replace `d:\DLpro\inpaint\app\static\models\model.state_dict`
5. Restart Docker:

```powershell
cd d:\DLpro\inpaint
docker-compose restart
```

6. Open `http://localhost:8003` — upgraded model is live!

---

## Quick Reference

| What | Where |
|---|---|
| Preprocessing script | `d:\DLpro\dataset_prep\preprocess.py` |
| Training code zip | `d:\DLpro\dataset_prep\inpaint_training.zip` |
| Colab notebook | `d:\DLpro\dataset_prep\FaceRestore_Training.ipynb` |
| Original model | `d:\DLpro\inpaint\app\static\models\model.state_dict` |
| Trained checkpoints | Google Drive → `facerestore_checkpoints/` |
| Training logs | **https://wandb.ai** (persists forever) |

---

## Estimated Timeline

| Step | Time | Your effort |
|---|---|---|
| Download CelebA | ~2 min | Click download |
| Extract zip | ~1 min | Right-click extract |
| Preprocess images | ~3-5 min | Run one command |
| Upload to Google Drive | ~11-13 min | Drag and wait |
| Colab setup (Cells 1-3) | ~3 min | Click play 3 times |
| Anti-idle hack | ~1 min | Paste JavaScript |
| Training 20 epochs | ~5-7 hours | Leave it running |
| **Total active effort** | **~20 minutes** | Rest is waiting |
