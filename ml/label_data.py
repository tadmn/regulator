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
from pathlib import Path
import time

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Increase buffer size to reduce popping/crackling
sd.default.blocksize = 1024


class SpectrogramLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Labeler")
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
        self.current_segment = 0  # 0=start, 1=middle, 2=end

        # Timing data for estimation
        self.label_times = []
        self.last_label_time = None

        # Setup UI
        self.setup_ui()

        # Load first file
        if self.files:
            self.last_label_time = time.time()  # Start timing
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

        ttk.Button(play_frame, text="Play (Space)", command=self.start_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(play_frame, text="Stop", command=self.stop_playback).pack(side=tk.LEFT, padx=2)

        # Label buttons
        label_frame = ttk.LabelFrame(control_frame, text="Label", padding="10")
        label_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(label_frame, text="PRO (P)", command=lambda: self.label_file('pro')).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_frame, text="CON (C)", command=lambda: self.label_file('con')).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_frame, text="UNSURE (U)", command=lambda: self.label_file('unsure')).pack(side=tk.LEFT, padx=2)

        # Keyboard bindings
        self.root.bind('p', lambda e: self.label_file('pro'))
        self.root.bind('c', lambda e: self.label_file('con'))
        self.root.bind('u', lambda e: self.label_file('unsure'))
        self.root.bind('<space>', lambda e: self.toggle_playback())

    def load_current_file(self):
        """Load and display the current audio file"""
        if not self.files:
            return

        filepath = self.files[self.current_index]

        try:
            # Load audio
            self.audio, self.sr = librosa.load(str(filepath), sr=None)
            self.duration = len(self.audio) / self.sr

            # Calculate time estimate
            time_estimate = self.get_time_estimate()

            # Update info
            info_text = f"{filepath.name}\n"
            info_text += f"Duration: {self.duration:.2f}s | Sample Rate: {self.sr}Hz"
            info_text += f"\nEst. time to label all remaining files: {time_estimate} ({len(self.files)} files)"

            self.info_label.config(text=info_text)

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
            segment_duration = 1.5

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

    def play_segment(self, segment):
        """Play a segment of the audio"""
        if self.audio is None:
            return

        # Stop any currently playing audio
        if self.playing:
            self.stop_playback()

        # Calculate segment boundaries
        segment_duration = 1.5  # seconds

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

        # Play audio
        self.playing = True
        try:
            sd.play(segment_audio, self.sr)
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.playing = False

    def stop_playback(self):
        """Stop audio playback"""
        sd.stop()
        self.playing = False
        # Remove highlight from spectrogram
        if self.audio is not None:
            self.plot_spectrogram()

    def toggle_playback(self):
        """Toggle between play and stop"""
        if self.playing:
            self.stop_playback()
        else:
            self.start_play()

    def start_play(self):
        """Start playing segments in a loop"""
        self.current_segment = 0
        self.play_segment('start')
        self.check_playback_status()

    def check_playback_status(self):
        """Check if audio is still playing and continue to next segment"""
        if not self.playing:
            return

        # Check if audio is still playing
        if sd.get_stream().active:
            # Still playing, check again in 30ms
            self.root.after(30, self.check_playback_status)
        else:
            # Playback finished - move to next segment
            self.playing = False
            segments = ['start', 'middle', 'end']
            self.current_segment = (self.current_segment + 1) % 3
            self.play_segment(segments[self.current_segment])
            self.root.after(30, self.check_playback_status)

    def get_time_estimate(self):
        """Calculate estimated time remaining based on labeling speed"""
        if len(self.label_times) < 2:
            return "..."

        # Calculate average time per file from recent labels
        avg_time = sum(self.label_times) / len(self.label_times)

        # Estimate time for remaining files
        remaining_seconds = avg_time * len(self.files)

        # Format the time estimate
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds / 60)
            seconds = int(remaining_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def label_file(self, label):
        """Label the current file and move to next"""
        if not self.files:
            return

        filepath = self.files[self.current_index]

        # Record time taken for this label
        if self.last_label_time is not None:
            time_taken = time.time() - self.last_label_time
            self.label_times.append(time_taken)
            # Keep only last 10 labels for rolling average
            if len(self.label_times) > 10:
                self.label_times.pop(0)

        self.last_label_time = time.time()

        # Remember if we were playing
        was_playing = self.playing

        # Stop any playing audio
        if self.playing:
            self.stop_playback()

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
                # If we were playing, start playing the new file from the beginning
                if was_playing:
                    self.start_play()
            else:
                self.show_error("All files labeled!")

        except Exception as e:
            self.show_error(f"Error moving file: {str(e)}")


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
    print("  Space - Play/Stop playback")

    # Create and run GUI
    root = tk.Tk()
    app = SpectrogramLabeler(root)
    root.mainloop()


if __name__ == '__main__':
    main()
