"""
Download the balanced AudioSet from HuggingFace (agkphysics/AudioSet) and
convert it to the format expected by audioset_tagging_cnn:

  dataset_dir/
  ├── audios/
  │   ├── balanced_train_segments/
  │   │   └── Y<video_id>.wav  (16 kHz, mono, 16-bit PCM, 10 s)
  │   └── eval_segments/
  │       └── Y<video_id>.wav
  └── metadata/
      ├── balanced_train_segments.csv
      └── eval_segments.csv

Usage:
    python scripts/convert_hf_to_wav.py \
        --dataset_dir ./datasets/audioset \
        [--cache_dir /path/to/hf/cache]   # optional HuggingFace cache dir
        [--num_workers 8]                  # parallel workers (default: CPU count)
"""

import os
import io
import argparse
import numpy as np
import soundfile as sf
from math import gcd
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Lazy-import heavy deps so error messages are clear if they are missing
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, Audio
except ImportError:
    raise ImportError(
        "The 'datasets' library is required.  Install it with:\n"
        "    pip install datasets huggingface_hub"
    )

try:
    from scipy.signal import resample_poly
except ImportError:
    raise ImportError("scipy is required.  Install it with: pip install scipy")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR    = 16000   # sample rate expected by the project
CLIP_SECONDS = 10      # every AudioSet clip is exactly 10 s
CLIP_SAMPLES = TARGET_SR * CLIP_SECONDS   # 160 000 samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resampling using scipy."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g  = gcd(orig_sr, target_sr)
    up = target_sr // g
    dn = orig_sr   // g
    return resample_poly(audio.astype(np.float64), up, dn).astype(np.float32)


def pad_or_truncate(x: np.ndarray, length: int) -> np.ndarray:
    """Ensure audio is exactly *length* samples."""
    if len(x) < length:
        return np.concatenate([x, np.zeros(length - len(x), dtype=np.float32)])
    return x[:length]


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------
def _process_single_clip(args_tuple):
    """Process a single clip: resample, pad/truncate, write WAV.

    Args:
        args_tuple: (video_id, audio_array, orig_sr, labels, audios_dir)

    Returns:
        (csv_row_string, None) on success,  (None, video_id) on skip.
    """
    video_id, audio_payload, labels, audios_dir = args_tuple

    try:
        if audio_payload is None:
            return (None, video_id)

        audio_bytes = audio_payload.get("bytes")
        audio_path = audio_payload.get("path")

        if audio_bytes is not None:
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio, orig_sr = sf.read(audio_buffer, dtype="float32", always_2d=False)
        elif audio_path:
            audio, orig_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        else:
            return (None, video_id)

        # Convert to mono
        if audio.ndim == 2:
            if audio.shape[0] >= audio.shape[1]:
                audio = np.mean(audio, axis=1)
            else:
                audio = np.mean(audio, axis=0)

        # Resample
        audio = resample(audio, int(orig_sr), TARGET_SR)

        # Pad / truncate
        audio = pad_or_truncate(audio, CLIP_SAMPLES)

        # Clip amplitude
        audio = np.clip(audio, -1.0, 1.0)

        # Write WAV
        wav_filename = f"Y{video_id}.wav"
        wav_path = os.path.join(audios_dir, wav_filename)
        sf.write(wav_path, audio, TARGET_SR, subtype="PCM_16")

        # CSV row
        label_str = ",".join(labels)
        csv_row = f'{video_id}, 0.000, 10.000, "{label_str}"'

        return (csv_row, None)
    except Exception:
        return (None, video_id)


# ---------------------------------------------------------------------------
# Core conversion function
# ---------------------------------------------------------------------------
def convert_split(
    hf_split,
    audios_dir: str,
    csv_path:   str,
    split_name: str,
    num_workers: int = 1,
) -> None:
    """
    Iterate *hf_split* (a HuggingFace Dataset object), write one WAV file per
    clip and build the AudioSet-style metadata CSV.
    """
    os.makedirs(audios_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    total   = len(hf_split)
    print(f"\n[{split_name}] Processing {total} clips with {num_workers} workers …")

    # Prepare work items — extract data before passing to workers
    def _generate_work_items():
        for i, sample in enumerate(hf_split):
            video_id   = sample["video_id"]
            audio_info = sample["audio"]
            labels     = sample["labels"]

            if audio_info is None:
                yield (video_id, None, labels, audios_dir)
            else:
                yield (video_id, audio_info, labels, audios_dir)

    rows    = []
    skipped = 0
    done    = 0

    if num_workers <= 1:
        # Single-process path (avoids pickling overhead)
        for item in _generate_work_items():
            csv_row, skip_id = _process_single_clip(item)
            if skip_id:
                skipped += 1
            else:
                rows.append(csv_row)
            done += 1
            if done % 1000 == 0 or done == total:
                pct = 100.0 * done / total
                print(f"  [{split_name}] {done}/{total}  ({pct:.1f}%)  skipped={skipped}")
    else:
        # Multi-process path
        with Pool(processes=num_workers) as pool:
            for csv_row, skip_id in pool.imap_unordered(
                _process_single_clip, _generate_work_items(), chunksize=32
            ):
                if skip_id:
                    skipped += 1
                else:
                    rows.append(csv_row)
                done += 1
                if done % 1000 == 0 or done == total:
                    pct = 100.0 * done / total
                    print(f"  [{split_name}] {done}/{total}  ({pct:.1f}%)  skipped={skipped}")

    # Write CSV
    with open(csv_path, "w") as f:
        f.write("# AudioSet metadata generated from agkphysics/AudioSet (HuggingFace)\n")
        f.write("# YTID, start_seconds, end_seconds, positive_labels\n")
        f.write("#\n")
        for row in rows:
            f.write(row + "\n")

    print(f"\n  Saved {len(rows)} WAV files  →  {audios_dir}")
    print(f"  Saved metadata CSV          →  {csv_path}")
    if skipped:
        print(f"  WARNING: {skipped} clips had no audio and were skipped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert agkphysics/AudioSet (balanced) from HuggingFace to WAV + CSV."
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="./datasets/audioset",
        help="Root directory where the converted dataset will be stored."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Optional directory for HuggingFace's local cache (downloaded parquets)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=cpu_count(),
        help=f"Number of parallel workers (default: {cpu_count()} = CPU count)."
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    num_workers = args.num_workers

    # ------------------------------------------------------------------
    # Balanced training split
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading balanced TRAIN split from HuggingFace …")
    print("(Parquet files will be downloaded on the first run — ~17 GB)")
    print("=" * 60)

    print(f"Using {num_workers} parallel workers.")

    train_ds = load_dataset(
        "agkphysics/AudioSet",
        name="balanced",
        split="train",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    train_ds = train_ds.cast_column("audio", Audio(decode=False))

    convert_split(
        hf_split    = train_ds,
        audios_dir  = os.path.join(dataset_dir, "audios", "balanced_train_segments"),
        csv_path    = os.path.join(dataset_dir, "metadata", "balanced_train_segments.csv"),
        split_name  = "balanced_train",
        num_workers = num_workers,
    )

    # ------------------------------------------------------------------
    # Evaluation (test) split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading EVAL (test) split from HuggingFace …")
    print("(Parquet files will be downloaded on the first run — ~17 GB)")
    print("=" * 60)

    eval_ds = load_dataset(
        "agkphysics/AudioSet",
        name="balanced",
        split="test",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    eval_ds = eval_ds.cast_column("audio", Audio(decode=False))

    convert_split(
        hf_split    = eval_ds,
        audios_dir  = os.path.join(dataset_dir, "audios", "eval_segments"),
        csv_path    = os.path.join(dataset_dir, "metadata", "eval_segments.csv"),
        split_name  = "eval",
        num_workers = num_workers,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All done!  Dataset layout:")
    print(f"  {dataset_dir}/")
    print( "  ├── audios/")
    print( "  │   ├── balanced_train_segments/   ← Y<id>.wav files (training)")
    print( "  │   └── eval_segments/             ← Y<id>.wav files (evaluation)")
    print( "  └── metadata/")
    print( "      ├── balanced_train_segments.csv")
    print( "      └── eval_segments.csv")
    print("=" * 60)
    print("\nNext step: run  scripts/2_pack_waveforms_to_hdf5.sh")


if __name__ == "__main__":
    main()
