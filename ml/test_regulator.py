#!/usr/bin/env python3
"""
Test the Regulator TFLite model on training/validation data.
Mirrors the feature extraction from train.py exactly.
"""

import numpy as np
import librosa
from pathlib import Path
import json
import argparse
import time

import tensorflow as tf


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / 'config.json'
    with open(config_path) as f:
        return json.load(f)


def load_labels(data_dir: Path = Path('labeled-audio-data')) -> dict:
    labels_dict = {}
    for file in (data_dir / 'pro').glob('*.wav'):
        labels_dict[file] = 'pro'
    for file in (data_dir / 'con').glob('*.wav'):
        labels_dict[file] = 'con'
    return labels_dict


def extract_features(audio_file: Path, config: dict) -> np.ndarray | None:
    """Extract features from audio file, matching training pipeline exactly."""
    sample_rate = config['sample_rate']
    duration = config['duration']
    n_fft = config['n_fft']
    hop_length = config['hop_length']
    expected_frames = config['expected_frames']
    clip_samples = int(sample_rate * duration)
    feature_set_hop_length = clip_samples  # non-overlapping clips, same as training

    try:
        y, sr = librosa.load(audio_file, sr=sample_rate)
        all_features = []

        for start in range(0, len(y) - clip_samples + 1, feature_set_hop_length):
            clip = y[start:start + clip_samples]
            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, clip_samples - len(clip)))

            centroid = librosa.feature.spectral_centroid(
                y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            features = np.vstack([centroid])

            if features.shape[1] < expected_frames:
                features = np.pad(features, ((0, 0), (0, expected_frames - features.shape[1])))
            elif features.shape[1] > expected_frames:
                features = features[:, :expected_frames]

            all_features.append(features.T)  # (n_frames, 1)

        if not all_features:
            clip = y
            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, clip_samples - len(clip)))
            centroid = librosa.feature.spectral_centroid(
                y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            features = np.vstack([centroid])
            if features.shape[1] < expected_frames:
                features = np.pad(features, ((0, 0), (0, expected_frames - features.shape[1])))
            elif features.shape[1] > expected_frames:
                features = features[:, :expected_frames]
            all_features.append(features.T)

        return np.array(all_features)  # (n_clips, n_frames, 1)

    except Exception as e:
        print(f"  [ERROR] {audio_file.name}: {e}")
        return None


def run_inference(interpreter, batch: np.ndarray) -> np.ndarray:
    """Run TFLite inference, handling both fixed and dynamic batch sizes."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_batch = input_details[0]['shape'][0]

    results = []
    for sample in batch:
        inp = sample[np.newaxis].astype(np.float32)  # (1, n_frames, 1)
        if expected_batch != 1:
            # Resize input tensor if model uses fixed batch != 1
            interpreter.resize_input_tensor(input_details[0]['index'], inp.shape)
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        results.append(out[0])

    return np.array(results)  # (n, 2)


def evaluate(model_dir: str = 'models', data_dir: str = 'labeled-audio-data', split: float = 0.2):
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)

    # ── Load config ──────────────────────────────────────────────────────────
    print("Loading config...")
    config = load_config(model_dir)
    label_names = config['labels']  # ['pro', 'con']

    # ── Load TFLite model ────────────────────────────────────────────────────
    tflite_path = model_dir / 'regulator.tflite'
    print(f"Loading TFLite model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input  shape : {input_details[0]['shape']}")
    print(f"  Output shape : {output_details[0]['shape']}")

    # ── Load & extract features ──────────────────────────────────────────────
    labels_dict = load_labels(data_dir)
    print(f"\nFound {len(labels_dict)} labeled files")

    X, y, file_labels = [], [], []
    for i, (filepath, label) in enumerate(labels_dict.items()):
        print(f"  [{i+1}/{len(labels_dict)}] {filepath.name}", end='\r')
        features = extract_features(filepath, config)
        if features is not None:
            label_idx = label_names.index(label)
            X.extend(features)
            y.extend([label_idx] * len(features))
            file_labels.extend([(filepath.name, label)] * len(features))

    print(f"\nExtracted {len(X)} total clips from {len(labels_dict)} files")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # ── Reproduce train/val split (same seed as training) ────────────────────
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=split, random_state=42, stratify=y
    )

    splits = {
        'Train': (X[train_idx], y[train_idx]),
        'Validation': (X[val_idx],   y[val_idx]),
        'All':        (X,             y),
    }

    # ── Evaluate each split ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for split_name, (X_split, y_split) in splits.items():
        t0 = time.time()
        probs = run_inference(interpreter, X_split)
        elapsed = time.time() - t0

        preds = np.argmax(probs, axis=1)
        correct = (preds == y_split).sum()
        total = len(y_split)
        accuracy = correct / total

        print(f"\n{split_name} ({total} clips, {elapsed:.2f}s total, {elapsed/total*1000:.1f}ms/clip):")
        print(f"  Accuracy : {accuracy:.4f}  ({correct}/{total})")

        # Per-class breakdown
        for cls_idx, cls_name in enumerate(label_names):
            mask = y_split == cls_idx
            if mask.sum() == 0:
                continue
            cls_correct = (preds[mask] == y_split[mask]).sum()
            cls_total = mask.sum()
            print(f"  {cls_name:>6} : {cls_correct/cls_total:.4f}  ({cls_correct}/{cls_total})")

        # Confusion matrix
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        header = "         " + "  ".join(f"{n:>8}" for n in label_names)
        print(f"  {header}")
        for actual_idx, actual_name in enumerate(label_names):
            row = []
            for pred_idx in range(len(label_names)):
                count = ((y_split == actual_idx) & (preds == pred_idx)).sum()
                row.append(f"{count:>8}")
            print(f"  {actual_name:>8} " + "  ".join(row))

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Regulator TFLite model on training data")
    parser.add_argument('--model-dir', default='models',
                        help='Directory containing regulator.tflite and config.json (default: models)')
    parser.add_argument('--data-dir', default='labeled-audio-data',
                        help='Root data directory with pro/ and con/ subfolders (default: labeled-audio-data)')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation split fraction, must match training (default: 0.2)')
    args = parser.parse_args()

    evaluate(model_dir=args.model_dir, data_dir=args.data_dir, split=args.split)


if __name__ == '__main__':
    main()
