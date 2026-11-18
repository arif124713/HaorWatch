# Kaggle U-Net training — HaorWatch AI

Training runs on Kaggle GPUs, never on this machine.

## Files
- `train_haorwatch_unet.ipynb` — the notebook to import into Kaggle
- `haorwatch_unet_training.py` — same code as a plain script (cell markers included)

## Steps
1. Open <https://www.kaggle.com>, sign in (Chrome profile 1 is already logged in).
2. `Create` → `New Notebook`.
3. `File` → `Import Notebook` → select `train_haorwatch_unet.ipynb` from this folder.
4. Notebook options (right panel):
   - Accelerator: **GPU T4 x2** or **P100**
   - Internet: **On**
5. `Run All`. Expected duration: roughly 3–6 hours on a T4 (downloads ~10 GB of
   Sen1Floods11 S1Hand/LabelHand chips first).
6. When finished, the Output panel contains `haorwatch_unet_v0.1.0.zip`.

## What the notebook guarantees
- Uses only the official Sen1Floods11 **hand-labelled** chips and **official
  event-level train/val/test splits** (no random chip mixing across events).
- Normalization statistics computed on the training split only.
- Probability threshold chosen on the validation split only, then applied
  unchanged to the untouched test split.
- Evaluates the U-Net **and** a per-chip Otsu baseline on exactly the same
  held-out test pixels, with per-event breakdowns.
- Records dataset audit, license note, split manifest, seed, config, and
  normalization stats into the artifact bundle. No fabricated numbers anywhere.

## After training
```powershell
# from the repo root
Expand-Archive haorwatch_unet_v0.1.0.zip -DestinationPath backend\models\unet\
cd backend
..\venv\Scripts\python.exe -m scripts.train_model --register models\unet\haorwatch_unet_v0.1.0
```
Registration copies the bundle, hashes the checkpoint, and writes
`manifest.json` so `scripts.run_analysis.py` can resolve the latest model
automatically. Evaluation metrics land in `metrics_eval.json` and are linked to
analysis runs through `analysis_metrics` when evaluation is wired into a run.
