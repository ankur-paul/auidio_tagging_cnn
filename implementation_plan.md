# Adding ESC-50, FSD50K, and UrbanSound8K Training Data for CNN14 at 16 kHz

## Background

The existing pipeline trains CNN14 on AudioSet's 527-class ontology using HDF5-packed waveforms at 32 kHz (down-sampled on-the-fly to 16 kHz). The balanced AudioSet subset from HuggingFace provides ~18k clips — far too few for the model to generalize well. Adding ESC-50, FSD50K, and UrbanSound8K dramatically increases training diversity.

**Key constraint:** the model outputs a 527-dim multi-hot vector (AudioSet classes). Every external dataset's labels must be mapped to the same 527 AudioSet class IDs.

## User Review Required

> [!IMPORTANT]
> **FSD50K already uses AudioSet MID labels** — mapping is 1:1. ESC-50 and UrbanSound8K use text labels that must be manually mapped to AudioSet MIDs. The mappings below are carefully chosen but you should review them for correctness.

> [!IMPORTANT]
> **UrbanSound8K "street_music" maps to generic "Music"** (`/m/04rlf`). This is the closest available AudioSet class but is very broad. Consider dropping it if precision matters.

> [!WARNING]
> All three datasets require **manual download** and cannot be auto-fetched by the script:
> - **ESC-50**: `git clone https://github.com/karolpiczak/ESC-50.git`
> - **FSD50K**: Download from [Zenodo](https://zenodo.org/record/4060432)
> - **UrbanSound8K**: Download from [UrbanSound8K website](https://urbansounddataset.weebly.com/urbansound8k.html) (requires request)

---

## Proposed Changes

### Data Preparation Script

#### [NEW] [prepare_external_datasets.py](file:///home/ankur/projects/audioset_tagging_cnn/scripts/prepare_external_datasets.py)

A single script with three sub-commands: `esc50`, `fsd50k`, `urbansound8k`. Each sub-command:

1. **Reads** the dataset's native metadata (CSV/JSON)
2. **Maps** labels to AudioSet MID format using hardcoded dictionaries
3. **Resamples** audio to 16 kHz mono (matching the HF conversion script's `TARGET_SR=16000`)
4. **Pads/truncates** clips: ESC-50 → 5s→10s (zero-pad), UrbanSound8K → ≤4s→10s (zero-pad), FSD50K → variable→10s (pad or truncate)
5. **Writes** `Y<unique_id>.wav` files + AudioSet-format CSV (3 header lines, then `<ID>, 0.000, 10.000, "<mid1>,<mid2>,..."`)

**Label mappings (ESC-50 → AudioSet MID):**

| ESC-50 label | AudioSet MID | AudioSet display name |
|---|---|---|
| dog | /m/05tny_ | Bark |
| rooster | /m/07st89h | Cluck (→ Chicken, rooster parent) |
| pig | /m/068zj | Pig |
| cow | /m/07rpkh9 | Moo |
| frog | /m/09ld4 | Frog |
| cat | /m/07qrkrw | Meow |
| hen | /m/09b5t | Chicken, rooster |
| insects | /m/03vt0 | Insect |
| sheep | /m/07bgp | Sheep |
| crow | /m/04s8yn | Crow |
| rain | /m/06mb1 | Rain |
| sea_waves | /m/034srq | Waves, surf |
| crackling_fire | /m/07pzfmf | Crackle |
| crickets | /m/09xqv | Cricket |
| chirping_birds | /m/07pggtn | Chirp, tweet |
| water_drops | /m/07r5v4s | Drip |
| wind | /m/03m9d0z | Wind |
| pouring_water | /m/07prgkl | Pour |
| thunderstorm | /m/0jb2l | Thunderstorm |
| toilet_flush | /m/01jt3m | Toilet flush |
| crying_baby | /t/dd00002 | Baby cry, infant cry |
| sneezing | /m/01hsr_ | Sneeze |
| clapping | /m/0l15bq | Clapping |
| breathing | /m/0lyf6 | Breathing |
| coughing | /m/01b_21 | Cough |
| footsteps | /m/07pbtc8 | Walk, footsteps |
| laughing | /m/01j3sz | Laughter |
| brushing_teeth | /m/012xff | Toothbrush |
| snoring | /m/01d3sd | Snoring |
| drinking_sipping | — | *(no good AudioSet match — skipped)* |
| door_wood_knock | /m/07r4wb8 | Knock |
| mouse_click | /m/07qc9xj | Clicking |
| keyboard_typing | /m/0316dw | Typing |
| door_wood_creaks | /m/07qh7jl | Creak |
| can_opening | /m/07pc8lb | Breaking |
| washing_machine | /m/025wky1 | Air conditioning *(closest domestic machine)* |
| vacuum_cleaner | /m/0d31p | Vacuum cleaner |
| clock_alarm | /m/046dlr | Alarm clock |
| clock_tick | /m/07qjznt | Tick |
| glass_breaking | /m/07rn7sz | Shatter |
| helicopter | /m/09ct_ | Helicopter |
| chainsaw | /m/01j4z9 | Chainsaw |
| siren | /m/03kmc9 | Siren |
| car_horn | /m/0912c9 | Vehicle horn, car horn, honking |
| engine | /m/02mk9 | Engine |
| train | /m/07jdr | Train |
| church_bells | /m/03w41f | Church bell |
| airplane | /m/0cmf2 | Fixed-wing aircraft, airplane |
| fireworks | /m/0g6b5 | Fireworks |
| hand_saw | /m/01b82r | Sawing |

**Label mappings (UrbanSound8K → AudioSet MID):**

| US8K label | AudioSet MID |
|---|---|
| air_conditioner | /m/025wky1 |
| car_horn | /m/0912c9 |
| children_playing | /t/dd00013 |
| dog_bark | /m/05tny_ |
| drilling | /m/01d380 |
| engine_idling | /m/07pb8fc |
| gun_shot | /m/032s66 |
| jackhammer | /m/03p19w |
| siren | /m/03kmc9 |
| street_music | /m/04rlf |

**FSD50K:** Uses AudioSet MIDs natively — no mapping needed. Its `vocabulary.csv` maps label names to MIDs; the ground truth CSVs already contain MIDs. Only MIDs present in AudioSet's 527-class vocabulary are kept; the rest are silently dropped.

---

### Index Combiner Script

#### [NEW] [combine_all_indexes.py](file:///home/ankur/projects/audioset_tagging_cnn/scripts/combine_all_indexes.py)

Merges multiple per-dataset index HDF5s (AudioSet balanced + ESC-50 + FSD50K + UrbanSound8K) into a single `combined_train.h5` index. This reuses the same HDF5 schema as [create_indexes.py](file:///home/ankur/projects/audioset_tagging_cnn/utils/create_indexes.py):

```
audio_name   (N,)              S20
target       (N, 527)          bool
hdf5_path    (N,)              S200
index_in_hdf5 (N,)             int32
```

The combined index is then passed to [main.py](file:///home/ankur/projects/audioset_tagging_cnn/pytorch/main.py) via `--data_type=combined_train`.

---

### Training Command Updates

No code changes needed in [main.py](file:///home/ankur/projects/audioset_tagging_cnn/pytorch/main.py). Training uses the combined index like this:

```bash
python3 pytorch/main.py train \
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

> [!NOTE]
> - `window_size=512` and `hop_size=160` are the 16 kHz equivalents of 1024/320 at 32 kHz (same time resolution)
> - `fmax=8000` because Nyquist at 16 kHz is 8 kHz

---

## Verification Plan

### Automated Tests

There are no existing unit tests in this codebase. Verification will be done via a dry-run smoke test:

```bash
# 1. Prepare a tiny subset of ESC-50 (only 2 files) and verify output
python scripts/prepare_external_datasets.py esc50 \
    --dataset_dir /path/to/ESC-50 \
    --output_dir ./datasets/esc50 \
    --dry_run

# 2. Verify the generated CSV has the correct format (3 header lines + data lines with MIDs)
head -10 ./datasets/esc50/metadata/esc50_train.csv

# 3. Pack to HDF5 and create indexes (using existing pipeline tools)
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=./datasets/esc50/metadata/esc50_train.csv \
    --audios_dir=./datasets/esc50/audios \
    --waveforms_hdf5_path=./workspaces/audioset_tagging/hdf5s/waveforms/esc50.h5

python3 utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path=./workspaces/audioset_tagging/hdf5s/waveforms/esc50.h5 \
    --indexes_hdf5_path=./workspaces/audioset_tagging/hdf5s/indexes/esc50.h5

# 4. Combine with AudioSet balanced index
python scripts/combine_all_indexes.py \
    --indexes ./workspaces/audioset_tagging/hdf5s/indexes/balanced_train.h5 \
              ./workspaces/audioset_tagging/hdf5s/indexes/esc50.h5 \
    --output ./workspaces/audioset_tagging/hdf5s/indexes/combined_train.h5

# 5. Verify the combined index has the expected sample count
python -c "import h5py; f=h5py.File('./workspaces/audioset_tagging/hdf5s/indexes/combined_train.h5','r'); print('Samples:', len(f['audio_name'])); print('Classes:', f['target'].shape[1])"
```

### Manual Verification

Since the user doesn't have the external datasets downloaded yet, the full end-to-end test will need to be done by the user after downloading the datasets. The script will print clear progress messages and validation summaries to make issues obvious.
