#!/usr/bin/env python3
"""
Interactive audio labeling tool with spectrogram visualization
Plays 3 segments (start, middle, end) and allows visual + audio labeling
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

SCRIPT_DIR = Path(__file__).resolve().parent
sd.default.blocksize = 1024


class SpectrogramLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Labeler")
        self.root.geometry("1100x760")
        self.root.configure(bg='#1e1e1e')

        self._apply_theme()

        self.input_dir  = SCRIPT_DIR / "audio-data"
        self.output_dir = SCRIPT_DIR / "labeled-audio-data"
        self.output_dir.mkdir(exist_ok=True)

        self.files = sorted(list(self.input_dir.glob("*.wav")))
        self.current_index = 0

        self.audio    = None
        self.sr       = None
        self.duration = 0
        self.playing  = False
        self.current_segment = 0

        self.label_times     = []
        self.last_label_time = None
        self._cbar           = None

        self.setup_ui()

        if self.files:
            self.last_label_time = time.time()
            self.load_current_file()
        else:
            self.show_error("No WAV files found in audio-data/")

    # ------------------------------------------------------------------
    def _apply_theme(self):
        BG   = '#1e1e1e'
        FG   = '#eeeeee'
        BTN  = '#3a3a3a'
        BTNF = '#ffffff'
        SEL  = '#5294e2'

        s = ttk.Style(self.root)
        s.theme_use('clam')

        s.configure('.',              background=BG,  foreground=FG,  font=('SF Pro Display', 13))
        s.configure('TFrame',         background=BG)
        s.configure('TLabel',         background=BG,  foreground=FG,  font=('SF Pro Display', 13))
        s.configure('TLabelframe',    background=BG,  foreground=FG,  bordercolor='#555')
        s.configure('TLabelframe.Label', background=BG, foreground='#aaa', font=('SF Pro Display', 12))

        s.configure('TButton',
                    background=BTN, foreground=BTNF,
                    padding=(14, 7), font=('SF Pro Display', 13),
                    relief='flat', borderwidth=0)
        s.map('TButton',
              background=[('active', SEL)], foreground=[('active', BTNF)])

        for name, bg in [('Pro.TButton',    '#2a5e2a'),
                         ('Con.TButton',    '#5e2a2a'),
                         ('Unsure.TButton', '#5a4a18')]:
            s.configure(name,
                        background=bg, foreground=BTNF,
                        padding=(14, 7), font=('SF Pro Display', 13),
                        relief='flat', borderwidth=0)
            s.map(name,
                  background=[('active', SEL)], foreground=[('active', BTNF)])

    # ------------------------------------------------------------------
    def setup_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # Info
        self.info_label = ttk.Label(main, text='', font=('SF Pro Display', 13),
                                    justify='center', anchor='center')
        self.info_label.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        # Create figure with two axes: spectrogram + a fixed colorbar axis.
        # Pre-allocating cax means the colorbar NEVER steals space on redraws.
        self.fig, (self.ax, self.cax) = plt.subplots(
            1, 2, figsize=(10, 3.8), dpi=100, facecolor='#1e1e1e',
            gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.05}
        )
        self.ax.set_facecolor('#1e1e1e')
        self.cax.set_facecolor('#1e1e1e')
        self.fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.18)

        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        # Re-apply margins after resize so labels are never clipped on Retina
        self.canvas.mpl_connect('resize_event', self._on_resize)

        # Controls
        ctrl = ttk.Frame(main)
        ctrl.grid(row=2, column=0, sticky='ew', pady=(10, 0))

        pb = ttk.LabelFrame(ctrl, text='Playback', padding=6)
        pb.pack(side='left', padx=(0, 10))
        ttk.Button(pb, text='▶  Play (Space)', command=self.start_play).pack(side='left', padx=3)
        ttk.Button(pb, text='■  Stop',         command=self.stop_playback).pack(side='left', padx=3)

        lb = ttk.LabelFrame(ctrl, text='Label', padding=6)
        lb.pack(side='left')
        ttk.Button(lb, text='✔  PRO  (P)',    style='Pro.TButton',
                   command=lambda: self.label_file('pro')).pack(side='left', padx=3)
        ttk.Button(lb, text='✘  CON  (C)',    style='Con.TButton',
                   command=lambda: self.label_file('con')).pack(side='left', padx=3)
        ttk.Button(lb, text='?  UNSURE  (U)', style='Unsure.TButton',
                   command=lambda: self.label_file('unsure')).pack(side='left', padx=3)

        self.root.bind('p', lambda e: self.label_file('pro'))
        self.root.bind('c', lambda e: self.label_file('con'))
        self.root.bind('u', lambda e: self.label_file('unsure'))
        self.root.bind('<space>', lambda e: self.toggle_playback())

    def _on_resize(self, event):
        self.fig.subplots_adjust(left=0.11, right=0.87, top=0.90, bottom=0.18)

    # ------------------------------------------------------------------
    def load_current_file(self):
        if not self.files:
            return
        fp = self.files[self.current_index]
        try:
            self.audio, self.sr = librosa.load(str(fp), sr=None)
            self.duration = len(self.audio) / self.sr
            est = self.get_time_estimate()
            self.info_label.config(text=(
                f"Duration: {self.duration:.2f}s  |  Sample Rate: {self.sr} Hz  |  "
                f"Est. remaining: {est}  ({len(self.files)} files)"
            ))
            self.plot_spectrogram()
        except Exception as e:
            self.show_error(f"Error loading {fp.name}: {e}")

    def plot_spectrogram(self, highlight_segment=None):
        self.ax.clear()

        D    = librosa.stft(self.audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        img = librosa.display.specshow(S_db, sr=self.sr,
                                       x_axis='time', y_axis='hz',
                                       ax=self.ax, cmap='magma')

        filename = self.files[self.current_index].name if self.files else ''
        self.ax.set_title(filename, color='#eeeeee', fontsize=11, pad=6)
        self.ax.set_xlabel('Time (s)', color='#aaaaaa', fontsize=11)
        self.ax.set_ylabel('Frequency (Hz)', color='#aaaaaa', fontsize=11)
        self.ax.tick_params(colors='#aaaaaa', labelsize=10)
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#444')

        if highlight_segment is not None:
            seg = 1.5
            if highlight_segment == 'start':
                t0 = 0.0
            elif highlight_segment == 'middle':
                t0 = self.duration / 2 - seg / 2
            else:
                t0 = self.duration - seg
            t0 = max(0.0, t0)
            self.ax.axvspan(t0, min(self.duration, t0 + seg), alpha=0.35, color='yellow')

        # Always draw colorbar into the fixed cax — no axes stealing
        self.cax.clear()
        self._cbar = self.fig.colorbar(img, cax=self.cax, format='%+2.0f dB')
        self._cbar.ax.tick_params(colors='#aaaaaa', labelsize=10)
        plt.setp(self._cbar.ax.yaxis.get_ticklabels(), color='#aaaaaa')

        self.canvas.draw()

    # ------------------------------------------------------------------
    def play_segment(self, segment):
        if self.audio is None:
            return
        if self.playing:
            self.stop_playback()

        seg_dur = 1.5
        if segment == 'start':
            s0 = 0;                                                self.current_segment = 0
        elif segment == 'middle':
            s0 = int((self.duration / 2 - seg_dur / 2) * self.sr); self.current_segment = 1
        else:
            s0 = int((self.duration - seg_dur) * self.sr);          self.current_segment = 2

        s0  = max(0, s0)
        s1  = min(len(self.audio), s0 + int(seg_dur * self.sr))
        buf = self.audio[s0:s1].copy()

        fade = int(0.01 * self.sr)
        if len(buf) > 2 * fade:
            buf[:fade]  *= np.linspace(0, 1, fade)
            buf[-fade:] *= np.linspace(1, 0, fade)

        self.plot_spectrogram(highlight_segment=segment)
        self.playing = True
        try:
            sd.play(buf, self.sr)
        except Exception as e:
            print(f"Playback error: {e}")
            self.playing = False

    def stop_playback(self):
        sd.stop()
        self.playing = False
        if self.audio is not None:
            self.plot_spectrogram()

    def toggle_playback(self):
        if self.playing:
            self.stop_playback()
        else:
            self.start_play()

    def start_play(self):
        self.current_segment = 0
        self.play_segment('start')
        self.check_playback_status()

    def check_playback_status(self):
        if not self.playing:
            return
        try:
            active = sd.get_stream().active
        except Exception:
            active = False

        if active:
            self.root.after(30, self.check_playback_status)
        else:
            self.playing = False
            self.current_segment = (self.current_segment + 1) % 3
            self.play_segment(['start', 'middle', 'end'][self.current_segment])
            self.root.after(30, self.check_playback_status)

    # ------------------------------------------------------------------
    def label_file(self, label):
        if not self.files:
            return
        fp = self.files[self.current_index]

        if self.last_label_time is not None:
            self.label_times.append(time.time() - self.last_label_time)
            if len(self.label_times) > 10:
                self.label_times.pop(0)
        self.last_label_time = time.time()

        was_playing = self.playing
        if self.playing:
            self.stop_playback()

        dest = self.output_dir / label
        dest.mkdir(exist_ok=True)
        try:
            fp.rename(dest / fp.name)
            self.files.pop(self.current_index)
            if self.current_index >= len(self.files) and self.current_index > 0:
                self.current_index -= 1
            if self.files:
                self.load_current_file()
                if was_playing:
                    self.start_play()
            else:
                self.show_error("✔  All files labeled!")
        except Exception as e:
            self.show_error(f"Error moving file: {e}")

    def get_time_estimate(self):
        if len(self.label_times) < 2:
            return "..."
        avg = sum(self.label_times) / len(self.label_times)
        s   = avg * len(self.files)
        if s < 60:    return f"{int(s)}s"
        if s < 3600:  return f"{int(s//60)}m {int(s%60)}s"
        return f"{int(s//3600)}h {int((s%3600)//60)}m"

    def show_error(self, msg):
        self.info_label.config(text=msg)


# ------------------------------------------------------------------
def main():
    input_dir = SCRIPT_DIR / "audio-data"
    if not input_dir.exists():
        input_dir.mkdir()
        print("Created audio-data/ — add WAV files and run again.")
        return
    files = list(input_dir.glob("*.wav"))
    if not files:
        print("No WAV files found in audio-data/")
        return

    print(f"Found {len(files)} WAV files")
    print("Shortcuts:  P=PRO  C=CON  U=UNSURE  Space=Play/Stop")

    root = tk.Tk()
    SpectrogramLabeler(root)
    root.mainloop()


if __name__ == '__main__':
    main()