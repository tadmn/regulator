#!/usr/bin/env python3
"""
Step 3: Train TensorFlow Lite model for Regulator
Extracts features, trains model, converts to TFLite
"""

import numpy as np
import librosa
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

default_num_epochs = 4

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
        
        # Feature set settings
        self.sample_rate = 22050
        self.duration = 3.0
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512

        self.feature_set_hop_length = int(self.duration * self.sample_rate)
        
        # Model will see this many time steps
        self.expected_frames = int((self.duration * self.sample_rate) / self.hop_length)
        
        print("Trainer initialized:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {self.duration}s")
        print(f"  MFCC coefficients: {self.n_mfcc}")
        print(f"  Expected frames: {self.expected_frames}")
    
    def load_labels(self):
        """Load labels from folder layout"""
        labels_dict = {}

        for file in list(Path('labeled-audio-data/pro').glob('*.wav')):
            labels_dict[file] = 'pro'

        for file in list(Path('labeled-audio-data/con').glob('*.wav')):
            labels_dict[file] = 'con'
        
        return labels_dict

    def extract_features(self, audio_file):
        """Extract features from audio file"""
        try:
            # Load entire audio file
            y, sr = librosa.load(audio_file, sr=self.sample_rate)

            # Calculate number of clips we can extract
            clip_samples = int(self.sample_rate * self.duration)

            # Extract features from overlapping clips
            all_features = []

            for start_sample in range(0, len(y) - clip_samples + 1, self.feature_set_hop_length):
                # Extract 3-second clip
                clip = y[start_sample:start_sample + clip_samples]

                # Pad if somehow short (edge case)
                if len(clip) < clip_samples:
                    clip = np.pad(clip, (0, clip_samples - len(clip)))

                centroid = librosa.feature.spectral_centroid(
                    y=clip,
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )

                # Combine features
                features = np.vstack([centroid])

                # Ensure consistent length
                if features.shape[1] < self.expected_frames:
                    # Pad
                    pad_width = self.expected_frames - features.shape[1]
                    features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
                elif features.shape[1] > self.expected_frames:
                    # Truncate
                    features = features[:, :self.expected_frames]

                all_features.append(features.T)  # Shape: (n_frames, 1)

            # If audio is too short for even one clip, extract from what we have
            if len(all_features) == 0:
                clip = y
                if len(clip) < clip_samples:
                    clip = np.pad(clip, (0, clip_samples - len(clip)))

                centroid = librosa.feature.spectral_centroid(
                    y=clip,
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )

                features = np.vstack([centroid])

                if features.shape[1] < self.expected_frames:
                    pad_width = self.expected_frames - features.shape[1]
                    features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
                elif features.shape[1] > self.expected_frames:
                    features = features[:, :self.expected_frames]

                all_features.append(features.T)

            # Stack all clips - return shape: (n_clips, n_frames, 1)
            return np.array(all_features)

        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None
    
    def prepare_dataset(self):
        """Load all audio files and extract features"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        labels_dict = self.load_labels()
        print(f"Found {len(labels_dict)} labeled files")
        
        # Count labels
        pro_count = sum(1 for v in labels_dict.values() if v == 'pro')
        con_count = sum(1 for v in labels_dict.values() if v == 'con')
        print(f"  Pro: {pro_count}")
        print(f"  Con: {con_count}")
        
        X = []
        y = []
        filenames = []
        
        print("\nExtracting features...")
        for i, (filename, label) in enumerate(labels_dict.items()):
            print(f"[{i+1}/{len(labels_dict)}] {Path(filename).name}", end='\r')
            
            features = self.extract_features(filename)
            if features is not None:
                label_val = 0 if label == 'pro' else 1
                # Each file can produce multiple clips, so we extend the lists
                X.extend(features)
                y.extend([label_val] * len(features))
                filenames.extend([filename] * len(features))
        
        print(f"\n\nSuccessfully processed {len(X)} files")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Feature shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y, filenames
    
    def build_model(self, input_shape):
        """Build CNN model"""
        print("\n" + "="*70)
        print("BUILDING MODEL")
        print("="*70)
        
        model = keras.Sequential([
            # Input layer
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
            keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Summary:")
        model.summary()
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=32):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                self.output_dir / 'regulator.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # Plot training history
        self.plot_history(history)

        model.export(self.output_dir / 'saved_model')

    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        print(f"\nTraining plot saved to: {self.output_dir / 'training_history.png'}")
    
    def convert_to_tflite(self):
        """Convert model to TensorFlow Lite"""
        print("\n" + "="*70)
        print("CONVERTING TO TFLITE")
        print("="*70)
        
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.output_dir / 'saved_model'))
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        tflite_path = self.output_dir / 'regulator.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"TFLite model saved: {tflite_path}")
        print(f"Model size: {size_mb:.2f} MB")
        
        # Test inference
        self.test_tflite(tflite_model)
        
        return tflite_path
    
    def test_tflite(self, tflite_model):
        """Test TFLite model inference"""
        print("\nTesting TFLite inference...")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Test with dummy data
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        import time
        start = time.time()
        interpreter.invoke()
        end = time.time()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Inference time: {(end-start)*1000:.2f}ms")
        print(f"Output: {output}")
    
    def save_config(self):
        """Save configuration"""
        config = {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'expected_frames': self.expected_frames,
            'labels': ['pro', 'con']
        }
        
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nConfig saved: {config_path}")
    
    def run(self, epochs=50):
        """Run complete training pipeline"""
        print("="*70)
        print("REGULATOR MODEL TRAINING")
        print("="*70)
        
        # Load and prepare data
        X, y, filenames = self.prepare_dataset()
        
        if len(X) < 50:
            print("\nWARNING: Very small dataset. Consider collecting more data.")
            print(f"Current: {len(X)} samples. Recommended: 500+")
        
        # Train
        self.train(X, y, epochs=epochs)
        
        # Convert to TFLite
        self.convert_to_tflite()
        
        # Save config
        self.save_config()
        
        # Final summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nModel files saved to: {self.output_dir}/")
        print(f"  - regulator.tflite (TensorFlow Lite model)")
        print(f"  - regulator.keras (Full Keras model)")
        print(f"  - config.json (Configuration)")
        print(f"  - training_history.png (Training plot)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train TensorFlow Lite model for Regulator"
    )
    parser.add_argument(
        '--output',
        default='models',
        help='Output directory for models (default: models)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=default_num_epochs,
        help=f'Number of training epochs (default: {default_num_epochs})'
    )
    
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
    import sys
    main()
