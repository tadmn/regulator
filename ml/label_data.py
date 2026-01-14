#!/usr/bin/env python3
"""
Interactive audio labeling tool with spectrogram visualization
Plays 3 segments (start, middle, end) and allows visual + audio labeling

Requirements:
brew install tkinter
pip3 install librosa matplotlib numpy sounddevice soundfile pillow

Usage:
python3 spectrogram_labeler.py
"""

import tkinter as tk
from tkinter import ttk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import threading
import time

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent


class SpectrogramLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Labeler with Spectrogram")
        self.root.geometry("1200x800")

        # Directories - relative to script location
        self.input_dir = SCRIPT_DIR / "audio-data"
        self.output_dir = SCRIPT_DIR / "labeled-audio-data"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)

        # Get list of files
        self.files = sorted(list(self.input_dir.glob("*.wav")))
        self.current_index = 0

        # Audio data
        self.audio = None
        self.sr = None
        self.duration = 0
        self.playing = False
        self.auto_playing = False
        self.current_segment = 0  # 0=start, 1=middle, 2=end

        # Setup UI
        self.setup_ui()

        # Load first file
        if self.files:
            self.load_current_file()
        else:
            self.show_error("No WAV files found in audio-data/")

    def setup_ui(self):
        """Create the UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 12))
        self.info_label.pack()

        # Spectrogram frame
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Playback buttons
        play_frame = ttk.LabelFrame(control_frame, text="Playback", padding="10")
        play_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(play_frame, text="Play", command=self.start_auto_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(play_frame, text="Stop (Space)", command=self.stop_playback).pack(side=tk.LEFT, padx=2)

        # Label buttons
        label_frame = ttk.LabelFrame(control_frame, text="Label", padding="10")
        label_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(label_frame, text="PRO (P)", command=lambda: self.label_file('pro')).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_frame, text="CON (C)", command=lambda: self.label_file('con')).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_frame, text="UNSURE (U)", command=lambda: self.label_file('unsure')).pack(side=tk.LEFT, padx=2)

        # Navigation buttons
        nav_frame = ttk.LabelFrame(control_frame, text="Navigation", padding="10")
        nav_frame.pack(side=tk.LEFT)

        ttk.Button(nav_frame, text="← Previous", command=self.prev_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next →", command=self.next_file).pack(side=tk.LEFT, padx=2)

        # Keyboard bindings
        self.root.bind('p', lambda e: self.label_file('pro'))
        self.root.bind('c', lambda e: self.label_file('con'))
        self.root.bind('u', lambda e: self.label_file('unsure'))
        self.root.bind('<space>', lambda e: self.stop_playback())
        self.root.bind('<Left>', lambda e: self.prev_file())
        self.root.bind('<Right>', lambda e: self.next_file())

    def load_current_file(self):
        """Load and display the current audio file"""
        if not self.files:
            return

        filepath = self.files[self.current_index]

        try:
            # Load audio
            self.audio, self.sr = librosa.load(str(filepath), sr=None)
            self.duration = len(self.audio) / self.sr

            # Update info
            self.info_label.config(
                text=f"File {self.current_index + 1}/{len(self.files)}: {filepath.name}\n"
                     f"Duration: {self.duration:.2f}s | Sample Rate: {self.sr}Hz"
            )

            # Plot spectrogram
            self.plot_spectrogram()

        except Exception as e:
            self.show_error(f"Error loading {filepath.name}: {str(e)}")

    def plot_spectrogram(self, highlight_segment=None):
        """Plot the spectrogram of the current audio"""
        self.ax.clear()

        # Compute spectrogram
        D = librosa.stft(self.audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', ax=self.ax)
        self.ax.set_title('Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        # Highlight the current segment if playing
        if highlight_segment is not None:
            segment_duration = 3.0

            if highlight_segment == 'start':
                start_time = 0
            elif highlight_segment == 'middle':
                start_time = self.duration / 2 - segment_duration / 2
            else:  # end
                start_time = self.duration - segment_duration

            start_time = max(0, start_time)
            end_time = min(self.duration, start_time + segment_duration)

            # Add highlighted region
            self.ax.axvspan(start_time, end_time, alpha=0.3, color='yellow')

        # Add colorbar
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')

        self.canvas.draw()

    def play_segment(self, segment, auto=False):
        """Play a segment of the audio"""
        if self.audio is None:
            return

        # If manually triggered (not auto), start auto-playing mode
        if not auto:
            self.auto_playing = True

        # Stop any currently playing audio
        if self.playing:
            self.stop_playback()
            time.sleep(0.1)  # Brief pause to ensure playback stopped

        # Calculate segment boundaries
        segment_duration = 3.0  # seconds

        if segment == 'start':
            start_sample = 0
            self.current_segment = 0
        elif segment == 'middle':
            start_sample = int((self.duration / 2 - segment_duration / 2) * self.sr)
            self.current_segment = 1
        else:  # end
            start_sample = int((self.duration - segment_duration) * self.sr)
            self.current_segment = 2

        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio), start_sample + int(segment_duration * self.sr))

        segment_audio = self.audio[start_sample:end_sample]

        # Apply fade in and fade out to prevent clicks/pops
        fade_duration = 0.01  # 10ms fade
        fade_samples = int(fade_duration * self.sr)

        if len(segment_audio) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            segment_audio[:fade_samples] *= fade_in

            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            segment_audio[-fade_samples:] *= fade_out

        # Update spectrogram to highlight current segment
        self.plot_spectrogram(highlight_segment=segment)

        # Play in background thread
        self.playing = True
        threading.Thread(target=self._play_audio, args=(segment_audio, auto), daemon=True).start()

    def _play_audio(self, audio, auto=False):
        """Internal method to play audio"""
        try:
            sd.play(audio, self.sr)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.playing = False
            # If auto-playing is enabled, continue to next segment
            if self.auto_playing:
                self.play_next_segment()

    def stop_playback(self):
        """Stop audio playback"""
        sd.stop()
        self.playing = False
        self.auto_playing = False
        # Remove highlight from spectrogram
        if self.audio is not None:
            self.plot_spectrogram()

    def start_auto_play(self):
        """Start auto-playing segments in a loop"""
        self.auto_playing = True
        self.current_segment = 0
        self.play_segment('start', auto=True)

    def play_next_segment(self):
        """Play the next segment in the loop"""
        if not self.auto_playing:
            return

        segments = ['start', 'middle', 'end']
        self.current_segment = (self.current_segment + 1) % 3
        self.play_segment(segments[self.current_segment], auto=True)

    def label_file(self, label):
        """Label the current file and move to next"""
        if not self.files:
            return

        filepath = self.files[self.current_index]

        # Determine destination
        dest_dir = self.output_dir / label
        dest_dir.mkdir(exist_ok=True)

        # Move file
        try:
            dest_path = dest_dir / filepath.name
            filepath.rename(dest_path)

            # Remove from list
            self.files.pop(self.current_index)

            # Load next file or previous if at end
            if self.current_index >= len(self.files) and self.current_index > 0:
                self.current_index -= 1

            if self.files:
                self.load_current_file()
            else:
                self.show_error("All files labeled!")

        except Exception as e:
            self.show_error(f"Error moving file: {str(e)}")

    def next_file(self):
        """Navigate to next file"""
        if not self.files:
            return

        self.current_index = (self.current_index + 1) % len(self.files)
        self.load_current_file()

    def prev_file(self):
        """Navigate to previous file"""
        if not self.files:
            return

        self.current_index = (self.current_index - 1) % len(self.files)
        self.load_current_file()

    def show_error(self, message):
        """Show error message"""
        self.info_label.config(text=message)


def main():
    # Check if audio-data directory exists - relative to script location
    input_dir = SCRIPT_DIR / "audio-data"
    if not input_dir.exists():
        print("Creating audio-data/ directory...")
        input_dir.mkdir()
        print("\nPlease add WAV files to audio-data/ and run again.")
        return

    # Check if there are files
    files = list(input_dir.glob("*.wav"))
    if not files:
        print("No WAV files found in audio-data/")
        print("Please add WAV files and run again.")
        return

    print(f"Found {len(files)} WAV files")
    print("\nStarting labeling tool...")
    print("\nKeyboard Shortcuts:")
    print("  P - Label as PRO")
    print("  C - Label as CON")
    print("  U - Label as UNSURE")
    print("  Space - Stop playback")
    print("  ← → - Navigate")

    # Create and run GUI
    root = tk.Tk()
    app = SpectrogramLabeler(root)
    root.mainloop()


if __name__ == '__main__':
    main()
