"""
prepare_data.py

Prepares the VoxCeleb1 dataset for CancelVoice training.
Resamples all clips to 16 kHz, splits into train/val/test partitions,
and writes a manifest CSV for use by train_cancelvoice.py.

Usage:
    python scripts/prepare_data.py \
        --vox-root datasets/voxceleb1 \
        --out-dir  data/prepared \
        --split    0.8 0.1 0.1

Output:
    data/prepared/train.csv
    data/prepared/val.csv
    data/prepared/test.csv

Each CSV contains columns:
    speaker_id, clip_path, duration_s, split
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import librosa
import soundfile as sf

TARGET_SR = 16000
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VoxCeleb1 data for CancelVoice.")
    parser.add_argument(
        "--vox-root",
        type=Path,
        required=True,
        help="Root directory of VoxCeleb1 dataset (contains speaker subdirectories).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/prepared"),
        help="Output directory for manifest CSVs and resampled audio.",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios. Must sum to 1.0.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Discard clips longer than this many seconds (default: 10.0).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Discard clips shorter than this many seconds (default: 1.0).",
    )
    return parser.parse_args()


def find_audio_files(root: Path) -> list[tuple[str, Path]]:
    """Walk the VoxCeleb1 directory tree and collect (speaker_id, filepath) pairs."""
    entries = []
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for audio_file in speaker_dir.rglob("*.wav"):
            entries.append((speaker_id, audio_file))
        for audio_file in speaker_dir.rglob("*.m4a"):
            entries.append((speaker_id, audio_file))
    return entries


def resample_and_save(src: Path, dst: Path) -> float | None:
    """
    Load audio, resample to TARGET_SR, save as WAV.
    Returns duration in seconds, or None if loading fails.
    """
    try:
        y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, y, TARGET_SR, subtype="PCM_16")
        return len(y) / TARGET_SR
    except Exception as exc:
        print(f"  Warning: could not process {src}: {exc}")
        return None


def assign_splits(
    entries: list[tuple[str, Path, float]],
    ratios: tuple[float, float, float],
    seed: int = RANDOM_SEED,
) -> list[tuple[str, Path, float, str]]:
    """
    Assign each entry a split label (train / val / test).
    Stratified by speaker so each speaker appears in all splits.
    """
    random.seed(seed)

    # group by speaker
    by_speaker: dict[str, list] = {}
    for speaker_id, path, duration in entries:
        by_speaker.setdefault(speaker_id, []).append((speaker_id, path, duration))

    labelled = []
    train_r, val_r, _ = ratios

    for speaker_clips in by_speaker.values():
        random.shuffle(speaker_clips)
        n = len(speaker_clips)
        n_train = max(1, int(n * train_r))
        n_val   = max(1, int(n * val_r))

        for i, (sid, path, dur) in enumerate(speaker_clips):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            labelled.append((sid, path, dur, split))

    return labelled


def write_manifest(rows: list[tuple], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_id", "clip_path", "duration_s", "split"])
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {out_path}")


def main():
    args = parse_args()

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    print(f"Scanning VoxCeleb1 at: {args.vox_root}")
    raw_entries = find_audio_files(args.vox_root)
    print(f"Found {len(raw_entries)} audio files across "
          f"{len(set(e[0] for e in raw_entries))} speakers.")

    audio_out_dir = args.out_dir / "audio"
    processed = []

    print(f"\nResampling to {TARGET_SR} Hz and filtering by duration...")
    for i, (speaker_id, src_path) in enumerate(raw_entries):
        rel = src_path.relative_to(args.vox_root)
        dst_path = audio_out_dir / rel.with_suffix(".wav")

        duration = resample_and_save(src_path, dst_path)
        if duration is None:
            continue
        if duration < args.min_duration or duration > args.max_duration:
            continue

        processed.append((speaker_id, dst_path, round(duration, 3)))

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(raw_entries)} files...")

    print(f"\nKept {len(processed)} clips after duration filtering.")

    labelled = assign_splits(processed, (train_r, val_r, test_r))

    splits = {"train": [], "val": [], "test": []}
    for row in labelled:
        splits[row[3]].append(row)

    for split_name, rows in splits.items():
        write_manifest(rows, args.out_dir / f"{split_name}.csv")

    print("\nDone. Summary:")
    for split_name, rows in splits.items():
        speakers = len(set(r[0] for r in rows))
        print(f"  {split_name:5}: {len(rows):5} clips | {speakers} speakers")


if __name__ == "__main__":
    main()
