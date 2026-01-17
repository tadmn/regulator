# Regulator

This repository contains scripts for collecting and labeling audio data for machine learning on macOS.

## Overview

The ML pipeline consists of two main stages:

1. **Data Collection** (`collect_data.py`) - Records system audio and automatically segments it
2. **Data Labeling** (`label_data.py`) - Interactive tool for labeling audio segments with visual spectrograms

## Setup (macOS)

### 1. Install System Dependencies

```bash
# Install BlackHole audio driver for system audio capture
brew install blackhole-2ch

# Install tkinter for GUI labeling tool
brew install python-tk
```

**Important:** After installing BlackHole, restart your computer.

### 2. Configure System Audio

1. Open **System Settings > Sound**
2. Set **"BlackHole 2ch"** as your audio output device
3. Audio will now be routed to BlackHole and captured by the collector script

### 3. Install Python Dependencies

```bash
pip3 install sounddevice soundfile numpy scipy librosa matplotlib pillow
```

## Stage 1: Data Collection

The `collect_data.py` script records system audio and automatically splits it into segments based on silence detection.

### Configuration

Edit the following variables in `collect_data.py` to customize behavior:

```python
OUTPUT_DIR = 'audio-data'       # Where to save segments
MIN_DURATION = 8.0              # Minimum segment length (seconds)
MAX_DURATION = 60.0             # Maximum segment length (seconds)
SILENCE_DURATION = 0.4          # Silence threshold for splitting (seconds)
SAMPLE_RATE = 22050             # Audio sample rate (Hz)
```

### Usage

```bash
python3 collect_data.py
```

The script will:
- Automatically detect and use the BlackHole 2ch audio device
- Record system audio continuously
- Split audio into segments based on silence detection
- Save segments as `.wav` files in the `audio-data/` directory
- Display real-time statistics (segment count, duration, buffer status)

Press `Ctrl+C` to stop recording. Any remaining buffered audio will be saved automatically.

### Output

Audio segments are saved as:
```
audio-data/segment_YYYYMMDD_HHMMSS_####.wav
```

Example: `segment_20260116_143022_0001.wav`

## Stage 2: Data Labeling

The `label_data.py` script provides an interactive GUI for labeling audio segments with visual spectrogram display.

### Usage

```bash
python3 label_data.py
```

### Features

- **Visual Spectrogram**: View frequency content over time
- **Smart Playback**: Automatically plays 3 segments (start, middle, end) in a loop
- **Keyboard Shortcuts**:
  - `P` - Label as PRO
  - `C` - Label as CON
  - `U` - Label as UNSURE
  - `Space` - Play/Stop playback
- **Time Estimation**: Shows estimated time to label all remaining files
- **Audio Stability**: Includes fade in/out and increased buffer size to prevent pops/clicks

### Workflow

1. The tool loads audio files from `audio-data/`
2. For each file, it displays a spectrogram visualization
3. Press `Space` to start playback (loops through start/middle/end segments)
4. Press `P`, `C`, or `U` to label and automatically move to the next file
5. Labeled files are moved to `labeled-audio-data/pro/`, `labeled-audio-data/con/`, or `labeled-audio-data/unsure/`

### Output Structure

```
labeled-audio-data/
├── pro/
│   ├── segment_20260116_143022_0001.wav
│   └── ...
├── con/
│   ├── segment_20260116_143045_0002.wav
│   └── ...
└── unsure/
    ├── segment_20260116_143102_0003.wav
    └── ...
```

## Tips

- **Collection**: Let the collector run in the background while playing audio. The longer it runs, the more training data you'll have.
- **Labeling**: The playback loops automatically, so you can listen multiple times before deciding on a label.
- **Audio Quality**: If you hear pops or clicks during labeling, try increasing `sd.default.blocksize` in `label_data.py`.
- **Batch Processing**: Label in focused sessions. The time estimate helps you plan breaks.

## Troubleshooting

### BlackHole not detected
- Verify BlackHole is installed: `brew list | grep blackhole`
- Restart your computer after installation
- Check that BlackHole appears in System Settings > Sound

### No audio during playback
- Verify BlackHole is set as your system output
- Check that audio files have content (not silent)
- Try adjusting system volume

### GUI doesn't launch
- Verify tkinter is installed: `python3 -c "import tkinter"`
- Install with: `brew install python-tk`
