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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

default_num_epochs = 50

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

        # Clip windowing
        self.clip_hop_ms = 30

        # Derived constants
        self.clip_frames     = int(self.sample_rate * self.duration)
        self.clip_hop_frames = int(self.clip_hop_ms / 1000.0 * self.sample_rate)

        # How many STFT frames fit in one clip.
        # This becomes the runtime parameter passed to the C++ module.
        self.sets_per_clip  = int(self.clip_frames / self.hop_length)  # e.g. 129

        print("Trainer initialized:")
        print(f"  Sample rate:      {self.sample_rate} Hz")
        print(f"  Clip duration:    {self.duration}s  ({self.clip_frames} samples)")
        print(f"  Clip hop:         {self.clip_hop_ms}ms")
        print(f"  Frames per clip:  {self.sets_per_clip}")

    # ------------------------------------------------------------------ #
    # Feature extraction — single bulk C++ call, fully multithreaded      #
    # ------------------------------------------------------------------ #

    def prepare_dataset(self):
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        pro_paths = [str(f) for f in Path('labeled-audio-data/pro').glob('*.wav')]
        con_paths = [str(f) for f in Path('labeled-audio-data/con').glob('*.wav')]

        print(f"Extracting features for {len(pro_paths)} 'pro' files")
        pro_clips = audio_features.extractFeatures(
            pro_paths,
            self.sets_per_clip,
            self.clip_hop_frames,
        )

        print(f"Extracting features for {len(pro_paths)} 'con' files")
        con_clips = audio_features.extractFeatures(
            con_paths,
            self.sets_per_clip,
            self.clip_hop_frames,
        )

        # Build label arrays directly from clip counts — no per-file loop needed
        y_pro = np.zeros(pro_clips.shape[0], dtype=np.int64)   # pro = 0
        y_con = np.ones(con_clips.shape[0],  dtype=np.int64)   # con = 1

        all_clips = np.concatenate([pro_clips, con_clips], axis=0)
        y         = np.concatenate([y_pro,     y_con],     axis=0)

        print(f"\nSuccessfully extracted {all_clips.shape[0]} clips")
        print(f"  Pro clips: {pro_clips.shape[0]}")
        print(f"  Con clips: {con_clips.shape[0]}")
        print(f"Feature shape: {all_clips.shape}")
        print(f"Labels shape:  {y.shape}")

        return all_clips, y

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
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        print("\nModel Summary:")
        model.summary()
        return model

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, X, y, epochs=50, batch_size=32):
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training clips:   {len(X_train)}")
        print(f"Validation clips: {len(X_val)}")

        # input_shape = (frames_per_clip, num_features)
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

        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
        print(f"Training accuracy:   {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

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
        print("\nTesting TFLite inference...")

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
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def run(self, epochs=50):
        print("=" * 70)
        print("REGULATOR MODEL TRAINING")
        print("=" * 70)

        X, y = self.prepare_dataset()

        if len(X) < 50:
            print(f"\nWARNING: Very small dataset ({len(X)} clips). "
                  "Consider collecting more data (recommended: 500+).")

        self.train(X, y, epochs=epochs)
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
    args = parser.parse_args()

    try:
        trainer = RegulatorTrainer(args.output)
        trainer.run(epochs=args.epochs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()