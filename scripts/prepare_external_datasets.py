"""
Prepare external datasets (ESC-50, FSD50K, UrbanSound8K) for training with
audioset_tagging_cnn.  Each sub-command reads the dataset's native format,
maps labels to AudioSet 527-class MIDs, resamples to 16 kHz mono, pads or
truncates to 10 s, and writes WAV files + an AudioSet-format CSV.

After running this script, use the existing pipeline tools to pack and index:

    python utils/dataset.py pack_waveforms_to_hdf5 ...
    python utils/create_indexes.py create_indexes ...

Usage:
    python scripts/prepare_external_datasets.py esc50 \
        --dataset_dir /path/to/ESC-50 \
        --output_dir  ./datasets/esc50

    python scripts/prepare_external_datasets.py fsd50k \
        --dataset_dir /path/to/FSD50K \
        --output_dir  ./datasets/fsd50k

    python scripts/prepare_external_datasets.py urbansound8k \
        --dataset_dir /path/to/UrbanSound8K \
        --output_dir  ./datasets/urbansound8k
"""

import os
import sys
import csv
import argparse
import numpy as np
import soundfile as sf
from math import gcd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from scipy.signal import resample_poly
except ImportError:
    raise ImportError("scipy is required.  Install: pip install scipy")

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required.  Install: pip install librosa")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR    = 16000
CLIP_SECONDS = 10
CLIP_SAMPLES = TARGET_SR * CLIP_SECONDS  # 160 000

# Path to AudioSet class_labels_indices.csv (relative to project root)
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CLASS_LABELS_CSV = PROJECT_ROOT / "metadata" / "class_labels_indices.csv"


# ---------------------------------------------------------------------------
# AudioSet label loader
# ---------------------------------------------------------------------------
def load_audioset_vocabulary(csv_path=CLASS_LABELS_CSV):
    """Load the 527-class AudioSet vocabulary.

    Returns:
        id_to_ix:  dict  MID string → integer index  (e.g. '/m/09x0r' → 0)
        valid_mids: set  of all valid MID strings
    """
    id_to_ix = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            idx = int(row[0])
            mid = row[1]
            id_to_ix[mid] = idx
    return id_to_ix, set(id_to_ix.keys())


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def high_quality_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling via scipy."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g  = gcd(orig_sr, target_sr)
    up = target_sr // g
    dn = orig_sr   // g
    return resample_poly(audio.astype(np.float64), up, dn).astype(np.float32)


def pad_or_truncate(x: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or truncate to exactly *length* samples."""
    if len(x) < length:
        return np.concatenate([x, np.zeros(length - len(x), dtype=np.float32)])
    return x[:length]


def load_and_prepare(audio_path: str, target_sr: int = TARGET_SR,
                     clip_samples: int = CLIP_SAMPLES) -> np.ndarray:
    """Load an audio file, convert to mono, resample, pad/truncate, clip."""
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = high_quality_resample(audio, sr, target_sr)
    audio = pad_or_truncate(audio, clip_samples)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def write_audioset_csv(rows: list, csv_path: str, dataset_name: str):
    """Write an AudioSet-format CSV (3 header lines + data)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(f"# {dataset_name} metadata mapped to AudioSet ontology\n")
        f.write("# YTID, start_seconds, end_seconds, positive_labels\n")
        f.write("#\n")
        for row in rows:
            f.write(row + "\n")
    print(f"  Saved CSV ({len(rows)} entries) → {csv_path}")


# ---------------------------------------------------------------------------
# Generic parallel worker
# ---------------------------------------------------------------------------
def _process_single_clip(args_tuple):
    """Worker function: load audio, resample, pad/truncate, write WAV.

    Args:
        args_tuple: (audio_path, unique_id, mids, audios_dir)

    Returns:
        csv_row string on success, None on failure.
    """
    audio_path, unique_id, mids, audios_dir = args_tuple

    try:
        audio = load_and_prepare(audio_path)
    except Exception as e:
        print(f"  ⚠ Error processing {audio_path}: {e}")
        return None

    wav_filename = f"Y{unique_id}.wav"
    wav_path = os.path.join(audios_dir, wav_filename)
    sf.write(wav_path, audio, TARGET_SR, subtype="PCM_16")

    label_str = ",".join(mids)
    return f'{unique_id}, 0.000, 10.000, "{label_str}"'


def _run_parallel(work_items, num_workers, dataset_name, total):
    """Execute work items in parallel and return CSV rows + stats."""
    rows = []
    skipped = 0
    done = 0

    if num_workers <= 1:
        for item in work_items:
            result = _process_single_clip(item)
            if result is None:
                skipped += 1
            else:
                rows.append(result)
            done += 1
            if done % 500 == 0 or done == total:
                pct = 100.0 * done / total
                print(f"  [{dataset_name}] {done}/{total} ({pct:.1f}%)")
    else:
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(
                _process_single_clip, work_items, chunksize=32
            ):
                if result is None:
                    skipped += 1
                else:
                    rows.append(result)
                done += 1
                if done % 500 == 0 or done == total:
                    pct = 100.0 * done / total
                    print(f"  [{dataset_name}] {done}/{total} ({pct:.1f}%)")

    return rows, skipped


# ═══════════════════════════════════════════════════════════════════════════
# ESC-50
# ═══════════════════════════════════════════════════════════════════════════
ESC50_LABEL_TO_MIDS = {
    "dog":              ["/m/05tny_"],            # Bark
    "rooster":          ["/m/09b5t"],              # Chicken, rooster
    "pig":              ["/m/068zj"],              # Pig
    "cow":              ["/m/07rpkh9"],            # Moo
    "frog":             ["/m/09ld4"],              # Frog
    "cat":              ["/m/07qrkrw"],            # Meow
    "hen":              ["/m/09b5t"],              # Chicken, rooster
    "insects":          ["/m/03vt0"],              # Insect
    "sheep":            ["/m/07bgp"],              # Sheep
    "crow":             ["/m/04s8yn"],             # Crow
    "rain":             ["/m/06mb1"],              # Rain
    "sea_waves":        ["/m/034srq"],             # Waves, surf
    "crackling_fire":   ["/m/07pzfmf"],            # Crackle
    "crickets":         ["/m/09xqv"],              # Cricket
    "chirping_birds":   ["/m/07pggtn"],            # Chirp, tweet
    "water_drops":      ["/m/07r5v4s"],            # Drip
    "wind":             ["/m/03m9d0z"],            # Wind
    "pouring_water":    ["/m/07prgkl"],            # Pour
    "thunderstorm":     ["/m/0jb2l"],              # Thunderstorm
    "toilet_flush":     ["/m/01jt3m"],             # Toilet flush
    "crying_baby":      ["/t/dd00002"],            # Baby cry, infant cry
    "sneezing":         ["/m/01hsr_"],             # Sneeze
    "clapping":         ["/m/0l15bq"],             # Clapping
    "breathing":        ["/m/0lyf6"],              # Breathing
    "coughing":         ["/m/01b_21"],             # Cough
    "footsteps":        ["/m/07pbtc8"],            # Walk, footsteps
    "laughing":         ["/m/01j3sz"],             # Laughter
    "brushing_teeth":   ["/m/012xff"],             # Toothbrush
    "snoring":          ["/m/01d3sd"],             # Snoring
    "drinking_sipping": [],                        # No good AudioSet match
    "door_wood_knock":  ["/m/07r4wb8"],            # Knock
    "mouse_click":      ["/m/07qc9xj"],            # Clicking
    "keyboard_typing":  ["/m/0316dw"],             # Typing
    "door_wood_creaks": ["/m/07qh7jl"],            # Creak
    "can_opening":      ["/m/07pc8lb"],            # Breaking
    "washing_machine":  ["/m/025wky1"],            # Air conditioning (closest)
    "vacuum_cleaner":   ["/m/0d31p"],              # Vacuum cleaner
    "clock_alarm":      ["/m/046dlr"],             # Alarm clock
    "clock_tick":       ["/m/07qjznt"],            # Tick
    "glass_breaking":   ["/m/07rn7sz"],            # Shatter
    "helicopter":       ["/m/09ct_"],              # Helicopter
    "chainsaw":         ["/m/01j4z9"],             # Chainsaw
    "siren":            ["/m/03kmc9"],             # Siren
    "car_horn":         ["/m/0912c9"],             # Vehicle horn
    "engine":           ["/m/02mk9"],              # Engine
    "train":            ["/m/07jdr"],               # Train
    "church_bells":     ["/m/03w41f"],             # Church bell
    "airplane":         ["/m/0cmf2"],              # Fixed-wing aircraft
    "fireworks":        ["/m/0g6b5"],              # Fireworks
    "hand_saw":         ["/m/01b82r"],             # Sawing
}


def prepare_esc50(args):
    """Prepare ESC-50 dataset.

    Expected layout:
        dataset_dir/
        ├── meta/esc50.csv
        └── audio/
            └── *.wav  (44.1 kHz, 5 s)
    """
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    num_workers = args.num_workers
    id_to_ix, valid_mids = load_audioset_vocabulary()

    meta_csv = dataset_dir / "meta" / "esc50.csv"
    audio_dir = dataset_dir / "audio"

    if not meta_csv.exists():
        print(f"ERROR: Cannot find {meta_csv}")
        print("Make sure --dataset_dir points to the ESC-50 repository root.")
        sys.exit(1)

    # Read ESC-50 metadata
    with open(meta_csv, "r") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    audios_dir = output_dir / "audios"
    os.makedirs(audios_dir, exist_ok=True)

    # Build work items
    work_items = []
    skipped_no_map = 0
    skipped_no_file = 0

    for sample in samples:
        filename  = sample["filename"]
        category  = sample["category"]
        esc_fold  = sample.get("fold", "0")

        mids = ESC50_LABEL_TO_MIDS.get(category, [])
        mids = [m for m in mids if m in valid_mids]
        if not mids:
            skipped_no_map += 1
            continue

        audio_path = audio_dir / filename
        if not audio_path.exists():
            skipped_no_file += 1
            continue

        base_name = Path(filename).stem
        unique_id = f"esc50_{esc_fold}_{base_name}"
        work_items.append((str(audio_path), unique_id, mids, str(audios_dir)))

    total = len(work_items)
    print(f"\n[ESC-50] Processing {total} clips with {num_workers} workers …")

    rows, process_errors = _run_parallel(work_items, num_workers, "ESC-50", total)

    csv_path = output_dir / "metadata" / "esc50_train.csv"
    write_audioset_csv(rows, str(csv_path), "ESC-50")

    print(f"\n  ✓ Processed: {len(rows)} clips")
    if skipped_no_map:
        print(f"  ⚠ Skipped (no AudioSet mapping): {skipped_no_map}")
    if skipped_no_file:
        print(f"  ⚠ Skipped (file not found): {skipped_no_file}")
    if process_errors:
        print(f"  ⚠ Skipped (processing errors): {process_errors}")
    print(f"  WAV files → {audios_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# UrbanSound8K
# ═══════════════════════════════════════════════════════════════════════════
US8K_CLASSID_TO_MIDS = {
    0: ["/m/025wky1"],   # air_conditioner → Air conditioning
    1: ["/m/0912c9"],    # car_horn → Vehicle horn
    2: ["/t/dd00013"],   # children_playing → Children playing
    3: ["/m/05tny_"],    # dog_bark → Bark
    4: ["/m/01d380"],    # drilling → Drill
    5: ["/m/07pb8fc"],   # engine_idling → Idling
    6: ["/m/032s66"],    # gun_shot → Gunshot, gunfire
    7: ["/m/03p19w"],    # jackhammer → Jackhammer
    8: ["/m/03kmc9"],    # siren → Siren
    9: ["/m/04rlf"],     # street_music → Music
}

US8K_CLASSID_TO_NAME = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing",
    3: "dog_bark", 4: "drilling", 5: "engine_idling",
    6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music",
}


def prepare_urbansound8k(args):
    """Prepare UrbanSound8K dataset.

    Expected layout:
        dataset_dir/
        ├── metadata/UrbanSound8K.csv
        ├── fold1/
        │   └── *.wav
        ├── fold2/
        ...
        └── fold10/
    """
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    num_workers = args.num_workers
    id_to_ix, valid_mids = load_audioset_vocabulary()

    meta_csv  = dataset_dir / "metadata" / "UrbanSound8K.csv"
    audio_root = dataset_dir

    if not meta_csv.exists():
        print(f"ERROR: Cannot find {meta_csv}")
        print("Make sure --dataset_dir points to the UrbanSound8K root.")
        sys.exit(1)

    with open(meta_csv, "r") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    audios_dir = output_dir / "audios"
    os.makedirs(audios_dir, exist_ok=True)

    # Build work items
    work_items = []
    skipped_no_file = 0

    for sample in samples:
        filename  = sample["slice_file_name"]
        fold      = sample["fold"]
        class_id  = int(sample["classID"])

        mids = US8K_CLASSID_TO_MIDS.get(class_id, [])
        mids = [m for m in mids if m in valid_mids]
        if not mids:
            continue

        audio_path = audio_root / f"fold{fold}" / filename
        if not audio_path.exists():
            skipped_no_file += 1
            continue

        base_name = Path(filename).stem
        unique_id = f"us8k_{fold}_{base_name}"
        work_items.append((str(audio_path), unique_id, mids, str(audios_dir)))

    total = len(work_items)
    print(f"\n[UrbanSound8K] Processing {total} clips with {num_workers} workers …")

    rows, process_errors = _run_parallel(work_items, num_workers, "UrbanSound8K", total)

    csv_path = output_dir / "metadata" / "urbansound8k_train.csv"
    write_audioset_csv(rows, str(csv_path), "UrbanSound8K")

    print(f"\n  ✓ Processed: {len(rows)} clips")
    if skipped_no_file:
        print(f"  ⚠ Skipped (file not found): {skipped_no_file}")
    if process_errors:
        print(f"  ⚠ Skipped (processing errors): {process_errors}")
    print(f"  WAV files → {audios_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# FSD50K
# ═══════════════════════════════════════════════════════════════════════════
def _load_fsd50k_ground_truth(dataset_dir: Path, split_name: str):
    """Load FSD50K ground truth from the standard CSV layout.

    Expected:
        dataset_dir/FSD50K.ground_truth/dev.csv   (columns: fname, labels, mids, split)
        dataset_dir/FSD50K.ground_truth/eval.csv

    Returns list of dicts: [{"fname": "12345", "mids": ["/m/xxx", ...]}, ...]
    """
    gt_csv = dataset_dir / "FSD50K.ground_truth" / f"{split_name}.csv"

    if gt_csv.exists():
        print(f"  Loading CSV ground truth: {gt_csv}")
        samples = []
        with open(gt_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["fname"]
                mids_raw = row.get("mids", "") or row.get("labels", "")
                mids = [m.strip() for m in mids_raw.split(",") if m.strip()]
                samples.append({"fname": fname, "mids": mids})
        return samples

    # Error — ground truth CSV not found
    print(f"ERROR: Cannot find FSD50K ground truth for '{split_name}' split.")
    print(f"  Searched for: {gt_csv}")
    print()
    print(f"  The ground truth CSV files are required. You can download them from:")
    print(f"  https://zenodo.org/record/4060432")
    print(f"  Look for FSD50K.ground_truth.zip and extract it into: {dataset_dir}")
    sys.exit(1)


def _resolve_fsd50k_audio_dir(dataset_dir: Path, split_name: str, candidates: list) -> Path:
    """Resolve FSD50K split audio directory, supporting nested layouts.

    Handles common structures such as:
      - dataset_dir/dev_audio/*.wav
      - dataset_dir/dev_audio/FSD50K.dev_audio/*.wav
      - dataset_dir/FSD50K.dev_audio/*.wav
    """
    exts = ("*.wav", "*.flac")

    def has_audio_files(path: Path) -> bool:
        return any(True for ext in exts for _ in path.glob(ext))

    for candidate in candidates:
        root = dataset_dir / candidate
        if not root.exists():
            continue

        if root.is_file():
            continue

        if has_audio_files(root):
            return root

        subdirs = [d for d in root.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if has_audio_files(subdir):
                return subdir

    tried = ", ".join(candidates)
    print(f"ERROR: Cannot find audio files for {split_name}. Tried roots: {tried}")
    print(f"Make sure {split_name} audio files (.wav/.flac) are extracted under one of these paths.")
    sys.exit(1)


def prepare_fsd50k(args):
    """Prepare FSD50K dataset.

    FSD50K already uses AudioSet MIDs as labels!  We just need to:
    1. Read ground truth CSVs (dev / eval)
    2. Filter MIDs to the 527-class AudioSet vocabulary
    3. Resample to 16 kHz, pad/truncate to 10 s
    4. Write WAV + AudioSet-format CSV

    Expected layout:
        dataset_dir/
        ├── FSD50K.ground_truth/
        │   ├── dev.csv        (fname, labels, mids, split)
        │   └── eval.csv       (fname, labels, mids, split)
        ├── dev_audio/          (or FSD50K.dev_audio/)
        │   └── *.wav
        └── eval_audio/         (or FSD50K.eval_audio/)
            └── *.wav
    """
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    split       = args.split
    num_workers = args.num_workers
    id_to_ix, valid_mids = load_audioset_vocabulary()

    audios_dir = output_dir / "audios"
    os.makedirs(audios_dir, exist_ok=True)

    audio_dir_candidates = {
        "dev":  ["dev_audio", "FSD50K.dev_audio"],
        "eval": ["eval_audio", "FSD50K.eval_audio"],
    }

    splits_to_process = []
    if split in ("dev", "all"):
        splits_to_process.append("dev")
    if split in ("eval", "all"):
        splits_to_process.append("eval")

    for split_name in splits_to_process:
        # Resolve audio dir (supports nested extracted layouts)
        audio_dir = _resolve_fsd50k_audio_dir(
            dataset_dir,
            split_name,
            audio_dir_candidates[split_name],
        )

        # Load ground truth
        samples = _load_fsd50k_ground_truth(dataset_dir, split_name)

        # Build work items
        work_items = []
        skipped_no_mid = 0
        skipped_no_file = 0

        for sample in samples:
            fname = sample["fname"]
            mids  = sample["mids"]

            mids = [m for m in mids if m in valid_mids]
            if not mids:
                skipped_no_mid += 1
                continue

            # Find audio file
            audio_path = None
            for ext in [".wav", ".flac"]:
                candidate = audio_dir / f"{fname}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break

            if audio_path is None:
                skipped_no_file += 1
                continue

            unique_id = f"fsd50k_{split_name}_{fname}"
            work_items.append((str(audio_path), unique_id, mids, str(audios_dir)))

        total = len(work_items)
        print(f"\n[FSD50K/{split_name}] Processing {total} clips with {num_workers} workers …")

        rows, process_errors = _run_parallel(
            work_items, num_workers, f"FSD50K/{split_name}", total
        )

        # Write CSV — dev goes to train CSV, eval goes to test CSV
        if split_name == "eval":
            csv_name = "fsd50k_eval.csv"
        else:
            csv_name = "fsd50k_train.csv"
        csv_path = output_dir / "metadata" / csv_name
        write_audioset_csv(rows, str(csv_path), f"FSD50K-{split_name}")

        print(f"\n  ✓ {split_name}: {len(rows)} clips processed")
        if skipped_no_mid:
            print(f"  ⚠ Skipped (no valid AudioSet MID): {skipped_no_mid}")
        if skipped_no_file:
            print(f"  ⚠ Skipped (file not found): {skipped_no_file}")
        if process_errors:
            print(f"  ⚠ Skipped (processing errors): {process_errors}")
        print(f"  WAV files → {audios_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Prepare external datasets for audioset_tagging_cnn training."
    )
    subparsers = parser.add_subparsers(dest="mode")

    # --- ESC-50 ---
    p_esc = subparsers.add_parser("esc50", help="Prepare ESC-50 dataset")
    p_esc.add_argument("--dataset_dir", type=str, required=True,
                       help="Path to the ESC-50 repository root")
    p_esc.add_argument("--output_dir", type=str, default="./datasets/esc50",
                       help="Output directory for prepared data")
    p_esc.add_argument("--num_workers", type=int, default=cpu_count(),
                       help=f"Number of parallel workers (default: {cpu_count()})")

    # --- UrbanSound8K ---
    p_us8k = subparsers.add_parser("urbansound8k", help="Prepare UrbanSound8K dataset")
    p_us8k.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the UrbanSound8K root directory")
    p_us8k.add_argument("--output_dir", type=str, default="./datasets/urbansound8k",
                        help="Output directory for prepared data")
    p_us8k.add_argument("--num_workers", type=int, default=cpu_count(),
                        help=f"Number of parallel workers (default: {cpu_count()})")

    # --- FSD50K ---
    p_fsd = subparsers.add_parser("fsd50k", help="Prepare FSD50K dataset")
    p_fsd.add_argument("--dataset_dir", type=str, required=True,
                       help="Path to the FSD50K root directory")
    p_fsd.add_argument("--output_dir", type=str, default="./datasets/fsd50k",
                       help="Output directory for prepared data")
    p_fsd.add_argument("--split", type=str, default="all",
                       choices=["dev", "eval", "all"],
                       help="Which FSD50K split(s) to process (default: all)")
    p_fsd.add_argument("--num_workers", type=int, default=cpu_count(),
                       help=f"Number of parallel workers (default: {cpu_count()})")

    args = parser.parse_args()

    if args.mode == "esc50":
        prepare_esc50(args)
    elif args.mode == "urbansound8k":
        prepare_urbansound8k(args)
    elif args.mode == "fsd50k":
        prepare_fsd50k(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
