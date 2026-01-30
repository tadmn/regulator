#!/usr/bin/env python3
"""
Step 1: Collect and segment Spotify audio
Records system audio and automatically splits into segments
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import time

# ============================================================================
# CONFIGURATION - Edit these values to customize behavior
# ============================================================================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'audio-data'  # Where to save audio segments
MIN_DURATION = 8.0  # Minimum segment length in seconds
MAX_DURATION = 60.0  # Maximum segment length in seconds
SILENCE_DURATION = 0.4  # Seconds of silence before splitting
SAMPLE_RATE = 22050  # Sample rate of saved audio in Hz


# ============================================================================


class AudioCollector:
    def __init__(self,
                 output_dir=OUTPUT_DIR,
                 min_duration=MIN_DURATION,
                 max_duration=MAX_DURATION,
                 silence_duration=SILENCE_DURATION,
                 sample_rate=SAMPLE_RATE):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate

        self.buffer = deque(maxlen=int(sample_rate * max_duration * 2))
        self.current_segment = []
        self.silence_frames = 0
        self.segment_count = 0
        self.total_duration = 0

        print(f"Collector initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Segment length: {min_duration}-{max_duration}s")

    def calculate_rms(self, audio):
        """Calculate RMS energy"""
        return np.sqrt(np.mean(audio ** 2))

    def save_segment(self):
        """Save current segment to file"""
        if not self.current_segment:
            return

        # Convert to numpy array
        audio = np.array(self.current_segment)
        duration = len(audio) / self.sample_rate

        # Check minimum duration
        if duration < self.min_duration:
            self.current_segment = []
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"segment_{timestamp}_{self.segment_count:04d}.wav"

        # Save
        sf.write(filename, audio, self.sample_rate)

        # Stats
        rms = self.calculate_rms(audio)
        self.segment_count += 1
        self.total_duration += duration

        print(f"[{self.segment_count:4d}] Saved: {filename.name:<40} "
              f"({duration:5.1f}s, RMS: {rms:.4f})")

        # Clear buffer
        self.current_segment = []
        self.silence_frames = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio"""
        if status:
            print(f"Status: {status}")

        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata[:, 0]

        is_silent = np.all(audio == 0)

        # Add to current segment
        self.current_segment.extend(audio)

        # Track silence
        if is_silent:
            self.silence_frames += frames
        else:
            self.silence_frames = 0

        # Check if we should save
        current_duration = len(self.current_segment) / self.sample_rate
        silence_duration = self.silence_frames / self.sample_rate

        should_save = False

        # Save if we hit max duration
        if current_duration >= self.max_duration:
            should_save = True

        # Save if we have enough duration and silence
        elif current_duration >= self.min_duration and silence_duration >= self.silence_duration:
            should_save = True

        if should_save:
            self.save_segment()

    def list_devices(self):
        """List available audio devices"""
        print("\n" + "=" * 70)
        print("AVAILABLE AUDIO DEVICES")
        print("=" * 70)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            marker = ""
            if device['max_input_channels'] > 0:
                if i == sd.default.device[0]:
                    marker = " [DEFAULT INPUT]"
                print(f"{i:2d}: {device['name']:<50} "
                      f"(in: {device['max_input_channels']}){marker}")
        print("=" * 70 + "\n")

    def start(self, device=None):
        """Start recording"""
        print("\n" + "=" * 70)
        print("AUDIO COLLECTOR - READY")
        print("=" * 70)

        self.list_devices()

        # Find BlackHole 2ch device
        devices = sd.query_devices()
        blackhole_device = None
        for i, dev in enumerate(devices):
            if 'BlackHole 2ch' in dev['name'] and dev['max_input_channels'] > 0:
                blackhole_device = i
                break

        if blackhole_device is not None:
            device = blackhole_device
            print(f"Using BlackHole 2ch device (device {device})")
        else:
            raise RuntimeError(
                "Could not find input device: BlackHole 2ch. Please install BlackHole and configure it as described in the file header.")

        print("\nPress Ctrl+C to stop recording\n")
        print("-" * 70)

        try:
            # Determine number of input channels for the device
            if device is None:
                device_info = sd.query_devices(sd.default.device[0], 'input')
            else:
                device_info = sd.query_devices(device, 'input')
            channels = min(device_info['max_input_channels'], 2)

            with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    callback=self.audio_callback,
                    blocksize=256,
                    device=device
            ):
                while True:
                    time.sleep(1)
                    if self.segment_count > 0:
                        print(f"[STATUS] Segments: {self.segment_count} | "
                              f"Total: {self.total_duration / 60:.1f} min | "
                              f"Buffer: {len(self.current_segment) / self.sample_rate:.1f}s",
                              end='\r')

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("RECORDING STOPPED")
            print("=" * 70)

            # Save any remaining audio
            if len(self.current_segment) / self.sample_rate >= self.min_duration:
                self.save_segment()

            print(f"\nFinal Statistics:")
            print(f"  Total segments: {self.segment_count}")
            print(f"  Total duration: {self.total_duration / 60:.1f} minutes")
            print(f"  Average segment: {self.total_duration / max(self.segment_count, 1):.1f}s")
            print(f"  Saved to: {self.output_dir}/")
            print("\nNext step: Run labeler to classify segments")


def main():
    collector = AudioCollector()
    collector.start()


if __name__ == '__main__':
    main()
