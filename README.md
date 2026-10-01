# FrequencyCleaner

Removes the "AI sound" from generated music. AI music generators leave dozens of steady, narrow high-frequency tones in their output. FrequencyCleaner detects these tones, removes them with notch filters, and can optionally soften the harsh top band with a high shelf.

Tones are found by comparing the spectrum to its local floor: a narrow peak above 5 kHz that rises a set number of dB above the floor is treated as a tone and notched out. A second pass re-detects on the cleaned result to catch leftovers. Filtering is done in the STFT domain (about 2.9 Hz per bin for tones, 11.7 Hz per bin for the shelf), with either a static cut or adaptive spectral subtraction, applied per channel, to mid only, or weighted by stereo coherence.

Input: MP3, WAV, FLAC, OGG, AIFF, M4A, MP4, AAC, Opus. Output: WAV 24-bit, WAV 32-bit float, FLAC 24-bit or MP3 320 kbps.

| Entry point | Description |
|---|---|
| `FrequencyCleaner.exe` | CPU build |
| `FrequencyCleaner-GPU.exe` | CUDA build (NVIDIA GPU) |
| `launch_app_venv.bat` | Run from source using the venv |

## Install

Requires Python 3.10+ on Windows. The app runs from a virtual environment named `venv` in the project folder.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You can instead run `venv_create.bat`, which creates the venv and installs the requirements.

Optional:

- **ffmpeg** on PATH, to open M4A, MP4, AAC and Opus files.
- **CUDA-enabled PyTorch** in the venv, for GPU processing. It is not in `requirements.txt` and must be installed manually. With the venv active, run the command from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version, for example:

  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```

## Usage

1. Run `launch_app_venv.bat`, or one of the exe files.
2. Load a track, or place tracks in the `input` folder.
3. Pick a preset: Gentle, Normal or Strong.
4. Export the cleaned track to the `output` folder, or batch process all tracks.

### Hotkeys

| Key | Action |
|---|---|
| Space | Play / pause |
| Tab | A/B toggle |
| R | Solo residual |
| Left / Right | Seek 5 s |
| Ctrl+Z | Undo |
| Ctrl+Y / Ctrl+Shift+Z | Redo |

## Building

```
build.bat
build-gpu.bat
```

`build-gpu.bat` requires CUDA-enabled PyTorch in the venv.
