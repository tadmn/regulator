#!/usr/bin/env python3
"""
Step 3: Train TensorFlow Lite model for Regulator
Extracts features via C++ module, trains model, converts to TFLite
"""

import sys, os

# Allow running local custom cpp modules without installing the package:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules/python"))

# Custom local cpp modules
import audio_features

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

default_num_epochs = 50

labeled_audio_data = "/Users/tad/craft/regulator/ml/labeled-audio-data"

# Will throw an error if GPU is not enabled
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Result:", c)


class RegulatorTrainer:
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Audio / STFT settings — must match C++ extractor
        self.sample_rate = 22050
        self.duration    = 3.0
        self.n_fft       = 2048
        self.hop_length  = 512
        self.clip_hop_frames = int(self.sample_rate * 1.5)

        # Derived constants
        self.clip_frames = int(self.sample_rate * self.duration)

        # How many STFT sets fit in one clip.
        # This becomes the runtime parameter passed to the C++ module.
        self.sets_per_clip = (self.clip_frames - self.n_fft) // self.hop_length + 1

        print("Trainer initialized:")
        print(f"  Sample rate:      {self.sample_rate} Hz")
        print(f"  Clip duration:    {self.duration}s  ({self.clip_frames} samples)")
        print(f"  Clip hop:         {self.clip_hop_frames} frames")
        print(f"  Sets per clip:  {self.sets_per_clip}")

    def prepare_dataset(self):
        """
        Extract features for all files in two batched, multithreaded calls
        (one per class) using extractFeaturesWithCounts, which returns both
        the flat clip array and a per-file clip count so we can reconstruct
        file_indices without serialising extraction to one file at a time.

        Returns
        -------
          X            – (total_clips, frames, features)
          y            – (total_clips,)  0=pro, 1=con
          file_indices – (total_clips,)  which source file each clip came from
          file_paths   – list of all source file paths (for reporting)
        """
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        pro_paths = sorted(Path(f'{labeled_audio_data}/pro').glob('*.wav'))
        con_paths = sorted(Path(f'{labeled_audio_data}/con').glob('*.wav'))

        def extract_class(paths, label, label_name, file_id_offset):
            """
            Single batched call for one class.  Returns clips, labels,
            file_indices, valid_paths (files that produced ≥1 clip).
            """
            if not paths:
                empty = np.empty((0, self.sets_per_clip, 0), dtype=np.float32)
                return empty, [], [], []

            print(f"\nExtracting features for {len(paths)} '{label_name}' files …")
            clips, counts = audio_features.extractFeaturesWithCounts(
                [str(p) for p in paths],
                self.sets_per_clip,
                self.clip_hop_frames,
            )

            labels       = []
            file_indices = []
            valid_paths  = []
            file_id      = file_id_offset

            for path, n in zip(paths, counts):
                if n == 0:
                    print(f"  [!] No clips from {path.name}, skipping")
                    continue
                labels.extend([label] * n)
                file_indices.extend([file_id] * n)
                valid_paths.append(str(path))
                file_id += 1

            # Drop clips from empty files (counts==0) — they produced no rows
            # in the array, so clips is already the right size; nothing to trim.
            return clips, labels, file_indices, valid_paths

        pro_clips, pro_labels, pro_file_ids, pro_valid = \
            extract_class(pro_paths, 0, 'pro', file_id_offset=0)

        con_clips, con_labels, con_file_ids, con_valid = \
            extract_class(con_paths, 1, 'con', file_id_offset=len(pro_valid))

        X            = np.concatenate([pro_clips, con_clips], axis=0)
        y            = np.array(pro_labels + con_labels, dtype=np.int64)
        file_indices = np.array(pro_file_ids + con_file_ids, dtype=np.int64)
        file_paths   = pro_valid + con_valid

        n_pro_clips = int((y == 0).sum())
        n_con_clips = int((y == 1).sum())
        n_files     = len(file_paths)

        print(f"\nSuccessfully extracted {X.shape[0]} clips from {n_files} files")
        if pro_valid:
            print(f"  Pro: {len(pro_valid)} files → {n_pro_clips} clips "
                  f"({n_pro_clips/len(pro_valid):.1f} avg clips/file)")
        else:
            print("  Pro: 0 files")
        if con_valid:
            print(f"  Con: {len(con_valid)} files → {n_con_clips} clips "
                  f"({n_con_clips/len(con_valid):.1f} avg clips/file)")
        else:
            print("  Con: 0 files")
        print(f"Feature shape:  {X.shape}")
        print(f"Labels shape:   {y.shape}")

        # Warn about severe class imbalance
        if n_pro_clips > 0 and n_con_clips > 0:
            ratio = max(n_pro_clips, n_con_clips) / min(n_pro_clips, n_con_clips)
            if ratio > 2.0:
                print(f"\n  ⚠ WARNING: clip imbalance ratio {ratio:.1f}x — "
                      "consider class_weight or resampling")

        return X, y, file_indices, file_paths

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #

    def build_model(self, input_shape):
        """Build Conv1D model. input_shape = (frames_per_clip, num_features)."""
        print("\n" + "=" * 70)
        print("BUILDING MODEL")
        print("=" * 70)

        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),

            # Conv block 1
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.3),

            # Conv block 2
            keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.3),

            # Conv block 3
            keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),

            # Dense layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),

            # Output
            keras.layers.Dense(2, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        print("\nModel Summary:")
        model.summary()
        return model

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, X, y, file_indices, file_paths, epochs=50, batch_size=32):
        """
        Split at the FILE level (not clip level) so no source file
        contributes clips to both train and validation sets.
        """
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)

        # ---- file-level split ----
        # GroupShuffleSplit guarantees all clips from one file stay together.
        n_files = file_indices.max() + 1
        if n_files < 5:
            print(f"  ⚠ WARNING: only {n_files} source files — "
                  "file-level split may produce very uneven sets")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=file_indices))

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Report which files landed in val so you can verify
        val_file_ids = set(file_indices[val_idx])
        print(f"\nFile-level split  ({n_files} total source files):")
        print(f"  Train files: {n_files - len(val_file_ids)}  →  {len(X_train)} clips")
        print(f"  Val   files: {len(val_file_ids)}  →  {len(X_val)} clips")
        print("\n  Validation files:")
        for fid in sorted(val_file_ids):
            print(f"    [{fid}] {Path(file_paths[fid]).name}")

        # ---- class weights (computed from train set only) ----
        # Square-root scaling acknowledges the imbalance without letting
        # minority-class gradients dominate and destabilize training.
        # A hard cap of 4.0 prevents explosion with severe imbalances (>16:1).
        import math
        n_total = len(y_train)
        n_pro   = int((y_train == 0).sum())
        n_con   = int((y_train == 1).sum())
        raw_pro = n_total / (2.0 * n_pro) if n_pro > 0 else 1.0
        raw_con = n_total / (2.0 * n_con) if n_con > 0 else 1.0
        class_weight = {
            0: min(math.sqrt(raw_pro), 4.0),
            1: min(math.sqrt(raw_con), 4.0),
        }
        print(f"\nClass weights:  pro={class_weight[0]:.3f}  con={class_weight[1]:.3f}"
              f"  (raw: pro={raw_pro:.2f}  con={raw_con:.2f})")

        # ---- model ----
        model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(
                self.output_dir / 'regulator.keras',
                monitor='val_accuracy', save_best_only=True),
        ]

        print("\nTraining…")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
        print(f"Training accuracy:   {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if train_acc - val_acc > 0.15:
            print("  ⚠ Large gap between train/val accuracy — model may still be "
                  "overfitting. Collect more files or increase dropout.")

        self.plot_history(history)
        model.export(self.output_dir / 'saved_model')

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'],     label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'],     label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        print(f"\nTraining plot saved to: {self.output_dir / 'training_history.png'}")

    def convert_to_tflite(self):
        print("\n" + "=" * 70)
        print("CONVERTING TO TFLITE")
        print("=" * 70)

        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(self.output_dir / 'saved_model'))
        tflite_model = converter.convert()

        tflite_path = self.output_dir / 'regulator.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"TFLite model saved: {tflite_path}  ({size_mb:.2f} MB)")

        self.test_tflite(tflite_model)
        return tflite_path

    def test_tflite(self, tflite_model):
        import time
        print("\nTesting TFLite inference…")

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input shape:  {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")

        dummy = np.random.randn(*input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)

        start = time.time()
        interpreter.invoke()
        end = time.time()

        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"Inference time: {(end - start) * 1000:.2f}ms")
        print(f"Output: {output}")

    def save_config(self):
        config = {
            'sample_rate':     self.sample_rate,
            'duration':        self.duration,
            'n_fft':           self.n_fft,
            'hop_length':      self.hop_length,
            'frames_per_clip': self.sets_per_clip,
            'clip_hop_frames': self.clip_hop_frames,
            'labels':          ['pro', 'con'],
        }
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved: {config_path}")

    # ------------------------------------------------------------------ #
    # Evaluate: run TFLite model on all files and rank by difficulty       #
    # ------------------------------------------------------------------ #

    def evaluate_files(self, tflite_path=None, n_top=10, open_hardest=False, open_easiest=False, open_uncertain=False):
        """
        Run the TFLite model on every labeled audio file and rank them by
        how easy vs. hard they are to classify.

        "Confidence" = the probability assigned to the predicted class.
          - Easiest  = highest confidence (model is very sure)
          - Hardest  = lowest confidence (model is close to 50/50)

        Also flags mis-classifications so you can spot problem files quickly.

        Parallelism
        -----------
        Feature extraction  — one batched multithreaded C++ call per class
                              (same as training; saturates all CPU cores).
        TFLite inference    — ThreadPoolExecutor, one interpreter per thread
                              (tf.lite.Interpreter is not thread-safe, so each
                              worker owns its own instance built from the raw
                              model bytes already in memory).
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        print("\n" + "=" * 70)
        print("FILE-LEVEL EVALUATION")
        print("=" * 70)

        # ---- locate model ----
        if tflite_path is None:
            tflite_path = self.output_dir / 'regulator.tflite'
        tflite_path = Path(tflite_path)
        if not tflite_path.exists():
            raise FileNotFoundError(
                f"TFLite model not found at {tflite_path}. "
                "Run training first (or pass --tflite <path>)."
            )

        with open(tflite_path, 'rb') as f:
            tflite_model = f.read()

        # Probe input/output tensor indices once from a throw-away interpreter.
        _probe = tf.lite.Interpreter(model_content=tflite_model)
        _probe.allocate_tensors()
        input_idx  = _probe.get_input_details()[0]['index']
        output_idx = _probe.get_output_details()[0]['index']
        del _probe

        # Per-thread interpreter cache — created on first use by each worker.
        _local = threading.local()

        def _get_interpreter():
            if not hasattr(_local, 'interp'):
                _local.interp = tf.lite.Interpreter(model_content=tflite_model)
                _local.interp.allocate_tensors()
            return _local.interp

        # ---- collect files ----
        label_dirs = {
            'pro': Path(f'{labeled_audio_data}/pro'),
            'con': Path(f'{labeled_audio_data}/con'),
        }
        label_ids = {'pro': 0, 'con': 1}

        all_wav_paths = []
        for label_name, folder in label_dirs.items():
            for p in sorted(folder.glob('*.wav')):
                all_wav_paths.append((label_name, p))

        n_total_files = len(all_wav_paths)
        if n_total_files == 0:
            print("  [!] No .wav files found in either label directory.")
            return

        # ---- batch feature extraction (multithreaded via C++ module) ----
        # One call per class so we keep the label association.
        print(f"\nExtracting features for {n_total_files} files …")

        def extract_class_clips(label_name):
            paths = [str(p) for ln, p in all_wav_paths if ln == label_name]
            if not paths:
                return np.empty((0, self.sets_per_clip, 1), dtype=np.float32), []
            clips, counts = audio_features.extractFeaturesWithCounts(
                paths, self.sets_per_clip, self.clip_hop_frames)
            return clips, counts

        pro_clips, pro_counts = extract_class_clips('pro')
        con_clips, con_counts = extract_class_clips('con')

        # Split flat clip arrays back into per-file slices.
        def split_by_counts(clips, counts):
            slices, start = [], 0
            for n in counts:
                slices.append(clips[start:start + n])
                start += n
            return slices

        pro_slices = split_by_counts(pro_clips, pro_counts)
        con_slices = split_by_counts(con_clips, con_counts)

        # Rebuild per-file work list with pre-extracted clips.
        work_items = []  # (label_name, wav_path, clips_array)
        pro_iter = iter(zip([p for ln, p in all_wav_paths if ln == 'pro'], pro_slices))
        con_iter = iter(zip([p for ln, p in all_wav_paths if ln == 'con'], con_slices))
        for label_name, wav_path in all_wav_paths:
            if label_name == 'pro':
                path, clips = next(pro_iter)
            else:
                path, clips = next(con_iter)
            work_items.append((label_name, wav_path, clips))

        # ---- parallel TFLite inference ----
        print(f"Running inference on {n_total_files} files …\n")

        completed_count = 0
        count_lock      = threading.Lock()

        def infer_file(label_name, wav_path, clips):
            nonlocal completed_count
            true_class = label_ids[label_name]

            if clips.shape[0] == 0:
                return None   # file too short — skip silently

            interp = _get_interpreter()
            t_start = time.time()
            clip_probs = []
            for clip in clips:
                inp = clip[np.newaxis].astype(np.float32)
                interp.set_tensor(input_idx, inp)
                interp.invoke()
                clip_probs.append(interp.get_tensor(output_idx)[0])
            t_elapsed_ms = (time.time() - t_start) * 1000

            mean_probs = np.mean(clip_probs, axis=0)
            pred_class = int(np.argmax(mean_probs))
            confidence = float(mean_probs[pred_class])

            with count_lock:
                completed_count += 1
                n = completed_count
                w = len(str(n_total_files))
                print(f"  [{n:>{w}}/{n_total_files}]  {wav_path.name:<45}", end='\r')

            return {
                'file':        wav_path.name,
                'path':        str(wav_path),
                'true_label':  label_name,
                'pred_label':  'pro' if pred_class == 0 else 'con',
                'confidence':  confidence,
                'difficulty':  1.0 - confidence,
                'correct':     pred_class == true_class,
                'n_clips':     len(clips),
                'ms_per_clip': t_elapsed_ms / len(clips),
                'probs_pro':   float(mean_probs[0]),
                'probs_con':   float(mean_probs[1]),
            }

        results = []
        n_workers = max(1, (os.cpu_count() or 4) // 2)  # leave headroom for C++ threads
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(infer_file, ln, p, clips): (ln, p)
                for ln, p, clips in work_items
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    ln, p = futures[fut]
                    print(f"\n  [!] Error on {p.name}: {exc}")

        print()   # end the \r progress line cleanly

        if not results:
            print("No results — check that audio files exist.")
            return

        results.sort(key=lambda r: r['difficulty'])

        # ---- summary stats ----
        n_total   = len(results)
        n_correct = sum(r['correct'] for r in results)
        n_wrong   = n_total - n_correct

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Files evaluated : {n_total}")
        print(f"  Correct         : {n_correct}  ({100*n_correct/n_total:.1f} %)")
        print(f"  Mis-classified  : {n_wrong}   ({100*n_wrong/n_total:.1f} %)")

        # ---- easiest files ----
        easiest = results[:n_top]
        print(f"\n{'─'*70}")
        print(f"  EASIEST {n_top} FILES  (model is most confident)")
        print(f"{'─'*70}")
        print(f"  {'File':<35} {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'OK?':>4}  {'Clips':>5}")
        print(f"  {'─'*35} {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*5}")
        for r in easiest:
            ok = "✓" if r['correct'] else "✗ WRONG"
            print(f"  {r['file']:<35} {r['true_label']:>4}  {r['pred_label']:>4}  "
                  f"{r['confidence']:>5.1%}  {ok:>7}  {r['n_clips']:>5}")

        # ---- open easiest files in system audio player ----
        if open_easiest and easiest:
            import subprocess
            print(f"\n  Opening {len(easiest)} easiest files in audio player …")
            for r in easiest:
                try:
                    subprocess.Popen(['open', r['path']])
                except Exception as exc:
                    print(f"  [!] Could not open {r['file']}: {exc}")

        # ---- hardest files ----
        hardest = list(reversed(results[-n_top:]))
        print(f"\n{'─'*70}")
        print(f"  HARDEST {n_top} FILES  (model is least confident)")
        print(f"{'─'*70}")
        print(f"  {'File':<35} {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'OK?':>4}  {'Clips':>5}")
        print(f"  {'─'*35} {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*5}")
        for r in hardest:
            ok = "✓" if r['correct'] else "✗ WRONG"
            print(f"  {r['file']:<35} {r['true_label']:>4}  {r['pred_label']:>4}  "
                  f"{r['confidence']:>5.1%}  {ok:>7}  {r['n_clips']:>5}")

        # ---- open hardest files in system audio player ----
        if open_hardest and hardest:
            import subprocess
            print(f"\n  Opening {len(hardest)} hardest files in audio player …")
            for r in hardest:
                try:
                    subprocess.Popen(['open', r['path']])
                except Exception as exc:
                    print(f"  [!] Could not open {r['file']}: {exc}")

        # ---- mis-classified files (if any) ----
        wrong_files = [r for r in results if not r['correct']]
        if wrong_files:
            print(f"\n{'─'*70}")
            print(f"  ALL MIS-CLASSIFIED FILES  ({len(wrong_files)} total)")
            print(f"{'─'*70}")
            print(f"  {'File':<35} {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'Clips':>5}")
            print(f"  {'─'*35} {'─'*4}  {'─'*4}  {'─'*6}  {'─'*5}")
            for r in sorted(wrong_files, key=lambda r: r['difficulty'], reverse=True):
                print(f"  {r['file']:<35} {r['true_label']:>4}  {r['pred_label']:>4}  "
                      f"{r['confidence']:>5.1%}  {r['n_clips']:>5}")
        else:
            print("\n  ✓ No mis-classified files!")

        # ---- open most uncertain files (closest to 50/50 regardless of correct/wrong) ----
        if open_uncertain:
            import subprocess
            # Sort by closeness to 0.5 confidence — most uncertain first
            most_uncertain = sorted(results, key=lambda r: r['difficulty'], reverse=True)[:n_top]
            print(f"\n  Opening {len(most_uncertain)} most uncertain files in audio player …")
            for r in most_uncertain:
                try:
                    subprocess.Popen(['open', r['path']])
                except Exception as exc:
                    print(f"  [!] Could not open {r['file']}: {exc}")

        # ---- save full ranking to JSON ----
        ranking_path = self.output_dir / 'file_difficulty_ranking.json'
        with open(ranking_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Full ranking saved to: {ranking_path}")

        self._plot_difficulty(results)

    def _plot_difficulty(self, results):
        """Bar chart: one bar per file, coloured by correct/wrong, sorted by difficulty."""
        files      = [r['file'] for r in results]
        confidence = [r['confidence'] for r in results]
        colors     = ['#2ecc71' if r['correct'] else '#e74c3c' for r in results]

        fig, ax = plt.subplots(figsize=(max(10, len(files) * 0.35), 5))
        ax.bar(range(len(files)), confidence, color=colors, edgecolor='none', width=0.8)

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='chance (0.5)')
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=90, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Confidence (predicted class probability)')
        ax.set_title('File Classification Difficulty\n'
                     '(left = easiest, right = hardest   |   green = correct, red = wrong)')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Correct'),
            Patch(facecolor='#e74c3c', label='Wrong'),
            plt.Line2D([0], [0], color='gray', linestyle='--', label='Chance (50 %)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'file_difficulty.png'
        plt.savefig(plot_path, dpi=150)
        print(f"  Difficulty plot saved to: {plot_path}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def run(self, epochs=50):
        print("=" * 70)
        print("REGULATOR MODEL TRAINING")
        print("=" * 70)

        X, y, file_indices, file_paths = self.prepare_dataset()
        print("\nmean:", X.mean())
        print("std:", X.std())

        if len(X) < 50:
            print(f"\nWARNING: Very small dataset ({len(X)} clips). "
                  "Consider collecting more data (recommended: 500+).")

        self.train(X, y, file_indices, file_paths, epochs=epochs)
        self.convert_to_tflite()
        self.save_config()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nModel files saved to: {self.output_dir}/")
        print("  - regulator.tflite        (TensorFlow Lite model)")
        print("  - regulator.keras         (Full Keras model)")
        print("  - config.json             (Configuration)")
        print("  - training_history.png    (Training plot)")



# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train TensorFlow Lite model for Regulator"
    )
    parser.add_argument('--output', default='models',
                        help='Output directory for models (default: models)')
    parser.add_argument('--epochs', type=int, default=default_num_epochs,
                        help=f'Number of training epochs (default: {default_num_epochs})')

    parser.add_argument('--evaluate', action='store_true',
                        help='Run TFLite model on all labeled audio files and '
                             'rank them by classification difficulty')
    parser.add_argument('--tflite', default=None,
                        help='Path to .tflite model file (default: <output>/regulator.tflite)')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of easiest/hardest files to display (default: 10)')
    parser.add_argument('--open-hardest', action='store_true',
                        help='Open the hardest N files in the system audio player after evaluation')
    parser.add_argument('--open-easiest', action='store_true',
                        help='Open the easiest N files in the system audio player after evaluation')
    parser.add_argument('--open-uncertain', action='store_true',
                        help='Open the N files closest to 50/50 confidence (most uncertain) after evaluation')

    args = parser.parse_args()

    try:
        trainer = RegulatorTrainer(args.output)

        if args.evaluate:
            trainer.evaluate_files(tflite_path=args.tflite, n_top=args.top, open_hardest=args.open_hardest, open_easiest=args.open_easiest, open_uncertain=args.open_uncertain)
        else:
            trainer.run(epochs=args.epochs)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()