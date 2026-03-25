"""
Combine multiple per-dataset index HDF5 files into a single training index.

This produces a merged HDF5 that can be passed to main.py for training on
AudioSet + external datasets together.

Usage:
    python scripts/combine_all_indexes.py \
        --indexes \
            ./workspaces/audioset_tagging/hdf5s/indexes/balanced_train.h5 \
            ./workspaces/audioset_tagging/hdf5s/indexes/esc50.h5 \
            ./workspaces/audioset_tagging/hdf5s/indexes/fsd50k.h5 \
            ./workspaces/audioset_tagging/hdf5s/indexes/urbansound8k.h5 \
        --output \
            ./workspaces/audioset_tagging/hdf5s/indexes/combined_train.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py

# Add utils to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import config


def combine_indexes(index_paths: list, output_path: str):
    """Merge multiple index HDF5 files into one.

    Each input HDF5 must have datasets: audio_name, target, hdf5_path, index_in_hdf5
    (the same schema produced by utils/create_indexes.py).
    """
    classes_num = config.classes_num
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nCombining {len(index_paths)} index files → {output_path}")
    print(f"  AudioSet classes: {classes_num}\n")

    with h5py.File(output_path, 'w') as out_hf:
        out_hf.create_dataset('audio_name',    shape=(0,),              maxshape=(None,),              dtype='S20')
        out_hf.create_dataset('target',        shape=(0, classes_num),  maxshape=(None, classes_num),  dtype=bool)
        out_hf.create_dataset('hdf5_path',     shape=(0,),              maxshape=(None,),              dtype='S200')
        out_hf.create_dataset('index_in_hdf5', shape=(0,),              maxshape=(None,),              dtype=np.int32)

        total = 0

        for path in index_paths:
            if not os.path.exists(path):
                print(f"  ⚠ SKIPPING (file not found): {path}")
                continue

            with h5py.File(path, 'r') as in_hf:
                n_existing = len(out_hf['audio_name'])
                n_new      = len(in_hf['audio_name'])
                n_total    = n_existing + n_new

                # Validate target shape
                if in_hf['target'].shape[1] != classes_num:
                    print(f"  ⚠ SKIPPING (wrong classes_num={in_hf['target'].shape[1]}): {path}")
                    continue

                # Resize and append
                out_hf['audio_name'].resize((n_total,))
                out_hf['audio_name'][n_existing:n_total] = in_hf['audio_name'][:]

                out_hf['target'].resize((n_total, classes_num))
                out_hf['target'][n_existing:n_total] = in_hf['target'][:]

                out_hf['hdf5_path'].resize((n_total,))
                out_hf['hdf5_path'][n_existing:n_total] = in_hf['hdf5_path'][:]

                out_hf['index_in_hdf5'].resize((n_total,))
                out_hf['index_in_hdf5'][n_existing:n_total] = in_hf['index_in_hdf5'][:]

                total += n_new
                print(f"  ✓ {os.path.basename(path):30s}  +{n_new:>7,d} samples  (running total: {n_total:,d})")

    print(f"\n  Combined index: {total:,d} total samples")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple index HDF5 files into one for multi-dataset training."
    )
    parser.add_argument(
        "--indexes", nargs="+", required=True,
        help="Paths to index HDF5 files to combine."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write the combined index HDF5."
    )

    args = parser.parse_args()
    combine_indexes(args.indexes, args.output)


if __name__ == "__main__":
    main()
