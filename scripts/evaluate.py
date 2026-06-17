"""
evaluate.py

Evaluates a trained CancelVoice checkpoint on the test split.

Computes:
  - Speaker embedding cosine distance (privacy metric)
  - Equal Error Rate on an ASV system (inversion attack resistance)
  - Word Error Rate via Whisper ASR (utility preservation)

Usage:
    python scripts/evaluate.py \
        --test-csv   data/prepared/test.csv \
        --checkpoint checkpoints/cancelvoice.pt \
        --out-dir    results

Output:
    results/metrics.csv   — per-clip scores
    results/summary.txt   — aggregate results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf

# ---------------------------------------------------------------------------
# Lazy imports — each metric only requires its own dependencies
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    """Load the trained CancelVoiceModel from a checkpoint."""
    # import here to avoid circular dependency if run standalone
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.train_cancelvoice import CancelVoiceModel

    state = torch.load(checkpoint_path, map_location=device)
    n_speakers = state.get("n_speakers", 1211)
    model = CancelVoiceModel(n_speakers=n_speakers).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Checkpoint loaded: {checkpoint_path} (epoch {state.get('epoch', '?')})")
    return model


def anonymize_clip(model, y: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """Run a single clip through the anonymization model and return the output waveform."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_t    = torch.tensor(mel_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        anon_mel, _, _, _ = model(mel_t)

    anon_mel = anon_mel.squeeze(0).cpu().numpy().T
    # denormalise
    anon_mel_power = librosa.db_to_power(anon_mel * mel_db.std() + mel_db.mean())
    # TODO: replace Griffin-Lim with HiFi-GAN vocoder for higher quality
    y_anon = librosa.feature.inverse.mel_to_audio(anon_mel_power, sr=sr, n_iter=64)
    return y_anon


# ---------------------------------------------------------------------------
# Metric: speaker embedding cosine distance
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_xvector_model(device):
    """Load SpeechBrain x-vector model. Returns None if not installed."""
    try:
        from speechbrain.pretrained import EncoderClassifier
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="/tmp/cancelvoice_xvect",
            run_opts={"device": str(device)},
        )
    except ImportError:
        print("Warning: speechbrain not installed. Skipping embedding distance metric.")
        print("         Install with: pip install speechbrain")
        return None


def extract_embedding(xvec_model, y: np.ndarray, device) -> np.ndarray | None:
    if xvec_model is None:
        return None
    tensor = torch.tensor(y).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = xvec_model.encode_batch(tensor)
    return emb.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Metric: Word Error Rate via Whisper
# ---------------------------------------------------------------------------

def get_whisper_model():
    """Load OpenAI Whisper model. Returns None if not installed."""
    try:
        import whisper
        model = whisper.load_model("base")
        print("Whisper ASR loaded (base model).")
        return model
    except ImportError:
        print("Warning: whisper not installed. Skipping WER metric.")
        print("         Install with: pip install openai-whisper")
        return None


def transcribe(whisper_model, y: np.ndarray, sr: int) -> str | None:
    if whisper_model is None:
        return None
    import whisper
    # whisper expects float32 at 16kHz
    audio = whisper.pad_or_trim(y.astype(np.float32))
    mel   = whisper.log_mel_spectrogram(audio)
    _, info = whisper_model.detect_language(mel)
    result = whisper_model.transcribe(y.astype(np.float32))
    return result["text"].strip()


def word_error_rate(ref: str, hyp: str) -> float:
    """Compute WER between reference and hypothesis transcriptions."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()

    # dynamic programming edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CancelVoice on test split.")
    parser.add_argument("--test-csv",   type=Path, required=True,
                        help="Path to test manifest CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained model checkpoint (.pt file).")
    parser.add_argument("--out-dir",    type=Path, default=Path("results"),
                        help="Directory to write evaluation results.")
    parser.add_argument("--max-clips",  type=int,  default=None,
                        help="Limit evaluation to this many clips (for quick testing).")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save anonymized audio files alongside metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train the model first: python scripts/train_cancelvoice.py")
        return

    model     = load_model(args.checkpoint, device)
    xvec      = get_xvector_model(device)
    whisper_m = get_whisper_model()

    # load test manifest
    test_entries = []
    with open(args.test_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_entries.append((row["speaker_id"], Path(row["clip_path"])))

    if args.max_clips:
        test_entries = test_entries[: args.max_clips]

    print(f"\nEvaluating on {len(test_entries)} clips...")

    per_clip_rows  = []
    cos_distances  = []
    wer_scores     = []

    for i, (speaker_id, clip_path) in enumerate(test_entries):
        if not clip_path.is_file():
            print(f"  Missing file: {clip_path} — skipping.")
            continue

        y_orig, sr = librosa.load(clip_path, sr=16000, mono=True)
        y_anon     = anonymize_clip(model, y_orig, sr, device)

        row = {"clip": clip_path.name, "speaker_id": speaker_id}

        # cosine distance
        emb_orig = extract_embedding(xvec, y_orig, device)
        emb_anon = extract_embedding(xvec, y_anon, device)
        if emb_orig is not None and emb_anon is not None:
            dist = cosine_distance(emb_orig, emb_anon)
            row["cosine_distance"] = round(dist, 4)
            cos_distances.append(dist)
        else:
            row["cosine_distance"] = "n/a"

        # WER
        ref_text = transcribe(whisper_m, y_orig, sr)
        hyp_text = transcribe(whisper_m, y_anon, sr)
        if ref_text is not None and hyp_text is not None:
            wer = word_error_rate(ref_text, hyp_text)
            row["wer"] = round(wer, 4)
            wer_scores.append(wer)
        else:
            row["wer"] = "n/a"

        per_clip_rows.append(row)

        # optionally save anonymized audio
        if args.save_audio:
            audio_out = args.out_dir / "audio" / clip_path.name
            audio_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_out, y_anon, sr, subtype="PCM_16")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(test_entries)} clips evaluated...")

    # write per-clip results
    metrics_path = args.out_dir / "metrics.csv"
    if per_clip_rows:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_clip_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_clip_rows)
        print(f"\nPer-clip metrics saved to {metrics_path}")

    # write summary
    summary = {
        "n_clips": len(per_clip_rows),
        "mean_cosine_distance": round(float(np.mean(cos_distances)), 4) if cos_distances else "n/a",
        "std_cosine_distance":  round(float(np.std(cos_distances)),  4) if cos_distances else "n/a",
        "mean_wer":             round(float(np.mean(wer_scores)),    4) if wer_scores    else "n/a",
        # EER requires an ASV evaluation setup — placeholder for future integration
        # TODO: integrate ECAPA-TDNN ASV system for EER computation
        "eer": "pending — requires ASV system integration",
    }

    summary_path = args.out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("CancelVoice Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        for k, v in summary.items():
            f.write(f"{k:30}: {v}\n")

    print("\nCancelVoice Evaluation Summary")
    print("=" * 40)
    for k, v in summary.items():
        print(f"{k:30}: {v}")
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
