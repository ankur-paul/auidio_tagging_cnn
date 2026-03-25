# Training PANNs on Balanced AudioSet (from HuggingFace)

This guide walks you through training a PANN (Pretrained Audio Neural Network)
from scratch using the **balanced AudioSet** subset sourced from
[agkphysics/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) on
HuggingFace.  Every step is explained in detail, including why each one exists
and what it produces.

---

## Table of Contents

1. [Overview of the Pipeline](#1-overview-of-the-pipeline)
2. [System Requirements](#2-system-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Compatibility Fixes (already applied)](#4-compatibility-fixes-already-applied)
5. [Step 1 — Download & Convert the Dataset](#5-step-1--download--convert-the-dataset)
6. [Step 2 — Pack Waveforms into HDF5](#6-step-2--pack-waveforms-into-hdf5)
7. [Step 3 — Create Training Indexes](#7-step-3--create-training-indexes)
8. [Step 4 — Train the Model](#8-step-4--train-the-model)
9. [Monitoring & Resuming Training](#9-monitoring--resuming-training)
10. [Expected Results](#10-expected-results)
11. [Directory Reference](#11-directory-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview of the Pipeline

```
HuggingFace parquet
        │
        ▼
[scripts/convert_hf_to_wav.py]
  → WAV files (32 kHz, mono, 10 s)
  → metadata CSVs
        │
        ▼
[utils/dataset.py  pack_waveforms_to_hdf5]
  → waveforms HDF5   (waveforms + targets packed together)
        │
        ▼
[utils/create_indexes.py  create_indexes]
  → index HDF5   (tells the DataLoader where to find each sample)
        │
        ▼
[pytorch/main.py  train]
  → checkpoints  /  statistics
```

### Why HDF5?

Reading thousands of individual WAV files during training is slow because each
file open is a separate syscall.  Packing everything into a few large HDF5 files
lets PyTorch's DataLoader read random batches with very low overhead.

### What is in the balanced dataset?

| Split | Clips available on HuggingFace |
|-------|-------------------------------|
| balanced train | ≈ 18 683 |
| eval (test)    | ≈ 17 141 |

There are **527 sound classes**.  The balanced set has roughly equal coverage
across classes (≈ 59 clips per class).  Results will be lower than the paper
(which used 2 million clips) but training is _much_ faster.

---

## 2. System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python   | 3.8     | 3.10+       |
| GPU VRAM | 8 GB    | 16 GB+      |
| RAM      | 16 GB   | 32 GB       |
| Disk     | 80 GB   | 120 GB      |
| CUDA     | 11.x    | 12.x        |

### Disk space breakdown

| What | Size |
|------|------|
| HuggingFace parquet cache (downloaded once, can delete after) | ~35 GB |
| WAV files — balanced_train_segments  | ~12 GB |
| WAV files — eval_segments            | ~11 GB |
| HDF5 waveforms (balanced_train + eval) | ~23 GB |
| HDF5 indexes (tiny)                  | ~0.1 GB |
| Checkpoints (saved every 100 k iters) | ~1–3 GB |

> **Tip:** Once the HDF5 files are created you can delete the WAV files to
> reclaim ~23 GB.  The HDF5 waveform files contain everything needed for
> training.

---

## 3. Environment Setup

All commands are run from the **project root** (`audioset_tagging_cnn/`).

### 3.1 Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Install dependencies

The original `requirements.txt` pins very old versions.  Use the modern list
below instead:

```bash
pip install --upgrade pip
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    soundfile \
    librosa \
    h5py \
    torch torchvision torchaudio \
    torchlibrosa \
    datasets \
    huggingface_hub
```

**What each package does:**

| Package | Role |
|---------|------|
| `numpy`, `scipy` | Array maths; `scipy` is used for audio resampling in the conversion script |
| `soundfile` | Write/read WAV files |
| `librosa` | Used by `utils/dataset.py` to load WAV files during HDF5 packing |
| `h5py` | Read/write HDF5 files |
| `torch` + `torchaudio` | Neural network training |
| `torchlibrosa` | On-the-fly log-mel spectrogram inside the model |
| `datasets` + `huggingface_hub` | Download the AudioSet parquets from HuggingFace |

> **GPU note:** the `torch` command above installs the CPU build.  For a
> CUDA-enabled build visit https://pytorch.org/get-started/locally/ and pick
> the right CUDA version.  Example for CUDA 12.1:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 4. Compatibility Fixes (already applied)

The codebase was written for NumPy < 1.20.  `np.bool` was deprecated in NumPy
1.20 and **removed** in NumPy 1.24.  The following files have already been
patched in this repository:

| File | Change |
|------|--------|
| `utils/utilities.py` line 74 | `np.bool` → `bool` |
| `utils/dataset.py` line 171 | `np.bool` → `bool` |
| `utils/create_indexes.py` lines 33, 70 | `np.bool` → `bool` |

If you ever see an `AttributeError: module 'numpy' has no attribute 'bool'`
error, check for any remaining `np.bool` occurrences:

```bash
grep -rn 'np\.bool[^_8]' utils/ pytorch/
```

---

## 5. Step 1 — Download & Convert the Dataset

### What this step does

The HuggingFace dataset stores audio as **FLAC bytes embedded in Parquet
files** at 48 kHz (or 44.1 kHz for ~10 % of clips).  This project expects:

- **WAV files**, mono, **32 kHz**, 16-bit PCM, exactly **10 seconds** each
- Named `Y<YouTube_ID>.wav` (the `Y` prefix avoids shell issues with IDs that
  start with `-`)
- A metadata **CSV** file listing each clip's ID, timestamps, and label IDs

The conversion script `scripts/convert_hf_to_wav.py` handles all of this
automatically.

### Run the conversion

```bash
DATASET_DIR="./datasets/audioset"

python scripts/convert_hf_to_wav.py \
    --dataset_dir "$DATASET_DIR"
```

If you want to store the HuggingFace cache in a specific location (e.g. a
separate large drive) pass `--cache_dir`:

```bash
python scripts/convert_hf_to_wav.py \
    --dataset_dir "$DATASET_DIR" \
    --cache_dir /mnt/bigdrive/hf_cache
```

**This will:**
1. Download the `balanced` train parquet files from HuggingFace (~17 GB).
2. Decode each FLAC clip from the parquet, resample to 32 kHz, pad/truncate to
   10 s, and write `Y<id>.wav` into `datasets/audioset/audios/balanced_train_segments/`.
3. Write `datasets/audioset/metadata/balanced_train_segments.csv`.
4. Repeat for the eval (test) split → `audios/eval_segments/` and
   `metadata/eval_segments.csv`.

> **Time estimate:** ~2–4 hours depending on your internet speed and CPU.  The
> resampling from 48 kHz → 32 kHz is the bottleneck.

### What the CSV looks like

The first 3 lines are header comments (the project skips them).  Each data
line has the format:

```
<YTID>, <start_seconds>, <end_seconds>, "<label_id_1>,<label_id_2>,..."
```

Example:
```
--PJHxphWEs, 0.000, 10.000, "/m/09x0r,/t/dd00088"
```

Label IDs (like `/m/09x0r`) are the **machine-readable IDs** from the AudioSet
ontology.  They are mapped to integer class indices at training time via
`metadata/class_labels_indices.csv` (already in the repo).

### Resulting directory structure

```
datasets/audioset/
├── audios/
│   ├── balanced_train_segments/
│   │   ├── Y--PJHxphWEs.wav
│   │   ├── Y--aE2O5G5WE.wav
│   │   └── ...  (~18 683 files)
│   └── eval_segments/
│       ├── Y--0A5LOc7Hg.wav
│       └── ...  (~17 141 files)
└── metadata/
    ├── balanced_train_segments.csv
    └── eval_segments.csv
```

---

## 6. Step 2 — Pack Waveforms into HDF5

### What this step does

This step reads every WAV file, looks up its labels from the CSV, and packs
**all waveforms + label targets** into a single HDF5 file.  Training with one
HDF5 is dramatically faster than opening thousands of WAV files.

Each HDF5 file contains three datasets:

| HDF5 dataset | Shape | Type | Description |
|---|---|---|---|
| `audio_name` | `(N,)` | bytes string | `Y<id>.wav` |
| `waveform`   | `(N, 320000)` | int16 | raw waveform, 32 kHz, 10 s |
| `target`     | `(N, 527)` | bool | multi-hot label vector |

Plus an attribute `sample_rate = 32000`.

### Run the packing

```bash
DATASET_DIR="./datasets/audioset"
WORKSPACE="./workspaces/audioset_tagging"

# Pack balanced training waveforms
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path="$DATASET_DIR/metadata/balanced_train_segments.csv" \
    --audios_dir="$DATASET_DIR/audios/balanced_train_segments" \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/balanced_train.h5"

# Pack evaluation waveforms
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path="$DATASET_DIR/metadata/eval_segments.csv" \
    --audios_dir="$DATASET_DIR/audios/eval_segments" \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/eval.h5"
```

> **Note:** This script must be run from the project root so that
> `utils/config.py` (which loads `metadata/class_labels_indices.csv` relative
> to the working directory) resolves correctly.

> **Time estimate:** 30–60 minutes per split.

### After this step

```
workspaces/audioset_tagging/
└── hdf5s/
    └── waveforms/
        ├── balanced_train.h5   (~12 GB)
        └── eval.h5             (~11 GB)
```

---

## 7. Step 3 — Create Training Indexes

### What this step does

The training loop does not read waveforms directly from the waveform HDF5.
Instead it reads from a small **index HDF5** that records:

| Field | Description |
|---|---|
| `audio_name` | filename |
| `target` | label vector |
| `hdf5_path` | absolute path to the waveform HDF5 |
| `index_in_hdf5` | integer row index inside that HDF5 |

This indirection allows the `BalancedTrainSampler` to build per-class sample
lists without loading any audio, and makes it easy to combine multiple HDF5
files (e.g. balanced + unbalanced) into one index later.

### Run index creation

```bash
WORKSPACE="./workspaces/audioset_tagging"

# Index for balanced training data
python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/balanced_train.h5" \
    --indexes_hdf5_path="$WORKSPACE/hdf5s/indexes/balanced_train.h5"

# Index for evaluation data
python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/eval.h5" \
    --indexes_hdf5_path="$WORKSPACE/hdf5s/indexes/eval.h5"
```

> **Important:** the `hdf5_path` stored inside the index is the **absolute
> path** you pass as `--waveforms_hdf5_path`.  If you move the waveform HDF5s
> afterwards you must re-run this step.

### After this step

```
workspaces/audioset_tagging/
└── hdf5s/
    ├── waveforms/
    │   ├── balanced_train.h5
    │   └── eval.h5
    └── indexes/
        ├── balanced_train.h5   (tiny, ~5 MB)
        └── eval.h5             (tiny, ~5 MB)
```

---

## 8. Step 4 — Train the Model

### How the training loop works

`pytorch/main.py` does the following in a loop:

1. The **sampler** picks a mini-batch.  With `--balanced balanced` it uses
   `BalancedTrainSampler`, which cycles through all 527 classes and picks one
   clip per class before repeating — this prevents common-class domination.
2. The **dataset** (`AudioSetDataset`) opens the waveform HDF5 and reads the
   raw waveform + target for each clip in the batch.
3. If `--augmentation mixup`, two random clips are linearly blended
   (Mixup augmentation).  The batch size is doubled before the model sees it,
   then blended pairs are combined.
4. The **model** (e.g. `Cnn14`) converts the waveform to a log-mel spectrogram
   on-the-fly via `torchlibrosa`, then passes it through convolutional layers.
5. **Binary cross-entropy** loss is computed (multi-label, one BCE per class).
6. Every **2 000 iterations** the model is evaluated on both the balanced train
   and eval sets and mAP (mean Average Precision) is printed.
7. A checkpoint is saved every **100 000 iterations**.

### Run training

```bash
WORKSPACE="./workspaces/audioset_tagging"

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
    --workspace="$WORKSPACE" \
    --data_type='balanced_train' \
    --sample_rate=32000 \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='Cnn14' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --resume_iteration=0 \
    --early_stop=600000 \
    --cuda
```

### Explanation of key arguments

| Argument | Value | Why |
|---|---|---|
| `--data_type` | `balanced_train` | Tells the script to load `hdf5s/indexes/balanced_train.h5` for training |
| `--model_type` | `Cnn14` | Best single-model architecture in the paper.  See [Available Models](#available-models) |
| `--balanced` | `balanced` | Use `BalancedTrainSampler` — essential when class frequencies vary |
| `--augmentation` | `mixup` | Mixup blends two clips; improves generalisation |
| `--early_stop` | `600000` | Stop after 600 k gradient steps.  With ~18 k clips and batch 32 this is ~1000 passes through the data |
| `--resume_iteration` | `0` | Start from scratch.  Set to e.g. `100000` to resume from a checkpoint |

### Available Models

These are all defined in `pytorch/models.py` and can be passed to `--model_type`:

| Model | Parameters | Notes |
|---|---|---|
| `Cnn6` | ~4.8 M | Fastest, lowest accuracy |
| `Cnn10` | ~4.9 M | Good balance of speed and accuracy |
| `Cnn14` | ~79.6 M | Best accuracy in the paper ← **recommended** |
| `MobileNetV1` | ~3.8 M | Optimised for edge deployment |
| `MobileNetV2` | ~3.2 M | Optimised for edge deployment |
| `Cnn14_16k` | ~79.6 M | Same as Cnn14 but designed for 16 kHz input |

For a quick test run use `Cnn6` or `MobileNetV2`.

### Where outputs are saved

The workspace layout after training:

```
workspaces/audioset_tagging/
├── checkpoints/main/
│   └── sample_rate=32000,...,fmax=14000/
│       └── data_type=balanced_train/
│           └── Cnn14/
│               └── loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/
│                   ├── 0_iterations.pth
│                   ├── 100000_iterations.pth
│                   └── ...
├── statistics/main/.../statistics.pkl
└── logs/main/.../0000.log
```

---

## 9. Monitoring & Resuming Training

### Watching the log

Training prints to both the terminal and a log file.  To follow it:

```bash
tail -f workspaces/audioset_tagging/logs/main/*/0000.log
```

Every 2 000 iterations you will see lines like:

```
Validate bal mAP: 0.412
Validate test mAP: 0.198
iteration: 10000, train time: 51.3 s, validate time: 83.2 s
```

- **bal mAP** — evaluated on the balanced _training_ set (will be higher,
  shows how well the model fits training data)
- **test mAP** — evaluated on the eval set (the real metric)
- mAP starts near 0.005 and should climb steadily

### Resuming from a checkpoint

If training is interrupted, find the latest checkpoint iteration:

```bash
ls workspaces/audioset_tagging/checkpoints/main/*/data_type=balanced_train/Cnn14/*/
```

Then restart with `--resume_iteration=<N>`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
    --workspace="./workspaces/audioset_tagging" \
    --data_type='balanced_train' \
    --sample_rate=32000 \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='Cnn14' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --resume_iteration=100000 \
    --early_stop=600000 \
    --cuda
```

### Reducing memory usage

If you hit out-of-memory errors, reduce the batch size:

```bash
--batch_size=16
```

With Mixup the actual batch fed into the model is `batch_size × 2`, so
`--batch_size=16` uses ~16 GB VRAM.

---

## 10. Expected Results

Training on balanced-only data (~18 k clips) versus the full dataset (~2 M
clips) will produce lower mAP.  Rough expectations:

| Data | Iterations | Test mAP |
|---|---|---|
| Balanced only | 100 000 | ~0.10 – 0.15 |
| Balanced only | 300 000 | ~0.15 – 0.20 |
| Balanced only | 600 000 | ~0.18 – 0.25 |
| Full (paper) | 600 000 | ~0.431 |

The gap is expected — the model needs millions of diverse clips to learn
robustly.  Balanced-only training is still useful for:
- Prototyping / verifying your training setup works end-to-end
- Fine-tuning downstream on a small task
- Studying class-balanced training behaviour

---

## 11. Directory Reference

```
audioset_tagging_cnn/             ← project root (run all commands here)
├── datasets/audioset/            ← raw dataset (created by Step 1)
│   ├── audios/
│   │   ├── balanced_train_segments/   Y*.wav  (32 kHz, 10 s)
│   │   └── eval_segments/             Y*.wav
│   └── metadata/
│       ├── balanced_train_segments.csv
│       └── eval_segments.csv
│
├── workspaces/audioset_tagging/  ← all training artefacts
│   ├── hdf5s/
│   │   ├── waveforms/
│   │   │   ├── balanced_train.h5     ← created by Step 2
│   │   │   └── eval.h5
│   │   └── indexes/
│   │       ├── balanced_train.h5     ← created by Step 3
│   │       └── eval.h5
│   ├── checkpoints/               ← model snapshots (Step 4)
│   ├── statistics/                ← mAP history (Step 4)
│   └── logs/                      ← training logs (Step 4)
│
├── metadata/
│   └── class_labels_indices.csv  ← 527 AudioSet class definitions (in repo)
│
├── pytorch/
│   ├── main.py                   ← training entry point
│   └── models.py                 ← CNN architectures
│
├── utils/
│   ├── config.py                 ← sample_rate, class count, label mappings
│   ├── dataset.py                ← pack_waveforms_to_hdf5
│   ├── create_indexes.py         ← create_indexes
│   └── data_generator.py         ← DataLoader / sampler classes
│
└── scripts/
    ├── convert_hf_to_wav.py      ← NEW: HuggingFace → WAV + CSV
    ├── 2_pack_waveforms_to_hdf5.sh
    ├── 3_create_training_indexes.sh
    └── 4_train.sh
```

---

## 12. Adding External Datasets (ESC-50, FSD50K, UrbanSound8K)

You can boost training data by adding external audio datasets.  The script
`scripts/prepare_external_datasets.py` handles downloading metadata parsing,
label mapping to AudioSet's 527-class ontology, resampling to 16 kHz, and
outputting WAV + CSV in the same format the rest of the pipeline expects.

### 12.1 Download the datasets

These datasets require manual download:

| Dataset | Source | Size |
|---|---|---|
| ESC-50 | `git clone https://github.com/karolpiczak/ESC-50.git` | ~600 MB |
| FSD50K | [Zenodo](https://zenodo.org/record/4060432) | ~30 GB |
| UrbanSound8K | [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) | ~6 GB |

### 12.2 Prepare each dataset

```bash
# ESC-50  (~2 000 clips, 50 classes → mapped to AudioSet MIDs)
python scripts/prepare_external_datasets.py esc50 \
    --dataset_dir /path/to/ESC-50 \
    --output_dir  ./datasets/esc50

# UrbanSound8K  (~8 700 clips, 10 classes → mapped to AudioSet MIDs)
python scripts/prepare_external_datasets.py urbansound8k \
    --dataset_dir /path/to/UrbanSound8K \
    --output_dir  ./datasets/urbansound8k

# FSD50K  (~51 000 clips, 200 classes — already uses AudioSet MIDs)
python scripts/prepare_external_datasets.py fsd50k \
    --dataset_dir /path/to/FSD50K \
    --output_dir  ./datasets/fsd50k \
    --split all
```

### 12.3 Pack waveforms and create indexes

Run the standard packing + indexing steps for each external dataset:

```bash
WORKSPACE="./workspaces/audioset_tagging"

# --- ESC-50 ---
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=./datasets/esc50/metadata/esc50_train.csv \
    --audios_dir=./datasets/esc50/audios \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/esc50.h5"

python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/esc50.h5" \
    --indexes_hdf5_path="$WORKSPACE/hdf5s/indexes/esc50.h5"

# --- UrbanSound8K ---
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=./datasets/urbansound8k/metadata/urbansound8k_train.csv \
    --audios_dir=./datasets/urbansound8k/audios \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/urbansound8k.h5"

python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/urbansound8k.h5" \
    --indexes_hdf5_path="$WORKSPACE/hdf5s/indexes/urbansound8k.h5"

# --- FSD50K ---
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=./datasets/fsd50k/metadata/fsd50k_train.csv \
    --audios_dir=./datasets/fsd50k/audios \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/fsd50k.h5"

python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path="$WORKSPACE/hdf5s/waveforms/fsd50k.h5" \
    --indexes_hdf5_path="$WORKSPACE/hdf5s/indexes/fsd50k.h5"
```

### 12.4 Combine all indexes

```bash
python scripts/combine_all_indexes.py \
    --indexes \
        "$WORKSPACE/hdf5s/indexes/balanced_train.h5" \
        "$WORKSPACE/hdf5s/indexes/esc50.h5" \
        "$WORKSPACE/hdf5s/indexes/urbansound8k.h5" \
        "$WORKSPACE/hdf5s/indexes/fsd50k.h5" \
    --output \
        "$WORKSPACE/hdf5s/indexes/combined_train.h5"
```

### 12.5 Train with combined data

```bash
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
    --workspace="$WORKSPACE" \
    --data_type='combined_train' \
    --sample_rate=16000 \
    --window_size=512 \
    --hop_size=160 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=8000 \
    --model_type='Cnn14' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --resume_iteration=0 \
    --early_stop=600000 \
    --cuda
```

> **Note:** `window_size=512` and `hop_size=160` are the 16 kHz equivalents
> of 1024/320 at 32 kHz (same time resolution).  `fmax=8000` because the
> Nyquist frequency at 16 kHz is 8 kHz.

---

## 13. Troubleshooting

### `AttributeError: module 'numpy' has no attribute 'bool'`
NumPy 1.24+ removed `np.bool`.  The patches in Step 4 should have fixed all
occurrences.  Check for any remaining ones:
```bash
grep -rn 'np\.bool[^_8]' utils/ pytorch/
```
Replace each `np.bool` with `bool`.

### `ModuleNotFoundError: No module named 'torchlibrosa'`
```bash
pip install torchlibrosa
```

### `RuntimeError: CUDA out of memory`
Reduce the batch size:
```bash
--batch_size=16   # or even 8
```

### HDF5 packing prints `File does not exist!` for many clips
This is normal — the HuggingFace dataset does not have 100 % of the original
AudioSet clips (YouTube videos are removed over time).  Missing files are
silently skipped; the HDF5 will still contain all available clips.

### `KeyError` or label mismatch during packing
Ensure you are running `utils/dataset.py` from the **project root** directory,
not from inside `utils/`.  The script imports `config.py` which opens
`metadata/class_labels_indices.csv` relative to the working directory.

### Training mAP stuck at ~0.005 for many iterations
This is normal at the start.  The model needs thousands of iterations to warm
up.  Do not lower the learning rate early.  Check that `--cuda` is set and
the GPU is actually being used:
```bash
watch -n1 nvidia-smi
```

### `trust_remote_code` warning from HuggingFace
This is expected for some versions of the dataset.  It is safe to pass
`trust_remote_code=True` (the conversion script already does this).

### Changing `DATASET_DIR` or `WORKSPACE` after index creation
The index HDF5 stores **absolute paths** to the waveform HDF5 files.  If you
move anything, re-run Step 3 from the new location.
```
