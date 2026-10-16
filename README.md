# FrequencyCleaner

An audio cleaner with a live spectrum, editable EQ curves, and automatic detection of narrow spectral outliers. These outliers can be tonal artifacts in generated or recorded music; detection does not establish whether audio is AI-generated.

Outliers are found by comparing the spectrum to its local median floor. Frequency limits, prominence, local median span, maximum peak width, time percentile, minimum level, notch width cap, and count limit are adjustable. A second pass can catch leftovers. Filtering uses overlap-add STFT processing, applied per channel, to mid only, or weighted by stereo coherence. Automatic outlier removal supports fixed reduction or adaptive spectral subtraction; EQ curves use per-frame dynamic gain by default, with a fixed-gain option for each band.

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
3. Choose Gentle, Normal or Strong from the preset dropdown beside the processing mode. Saved presets appear there too; manage them in Settings > Processing. In EQ mode these strengths regenerate automatic dynamic bands and keep manual bands.
4. Use **+ Add curve** to choose a shape, or double-click the spectrum. Select **Soften highs** for broad high-frequency attenuation, with start, end and depth controls.
5. Open **Outliers** to tune automatic detection. Detector changes run a fresh detection.
6. Compare with A/B, then export to the `output` folder or batch process all tracks.

### One live display

**Spectrum** fills the main workspace. The compact track browser shows filenames without indices or a second metadata row. Its divider resizes it; the collapse control in its own header reduces it to a narrow rail with an expand button. Filenames shorten only at the end and their tooltips show the complete path. The top bar shows the full selectable file path, with a copy button. **Outliers** and **History** open their inspectors.

The top controls share a 34-pixel height and 14-pixel font. The **Save** icon opens one menu for cleaned audio, removed audio, and batch export. The **Settings** gear opens one modal with General, Processing, Outliers, Analyzer, Appearance, Export, and History tabs. Changes preview immediately and save automatically. **Dark gray** is the default color profile; Graphite and Black are also available. Blue, teal, amber, and violet accents affect the frequency display while general buttons stay neutral gray (A/B uses distinct source colors). Color choices and the collapsed sidebar state persist between sessions.

Each file has its own saved curves, detector rules, processing settings and fixed cuts. A new track starts with fresh automatic detection; returning to it restores its config, including after restarting the app. An intentionally empty EQ stays empty. **Auto curves** explicitly refreshes suggestions for the current track.

Playback volume (including mute), analyzer smoothing, EQ gain range, peak fade and fixed-cut overlay visibility are remembered between sessions.

Use **Copy config** in the top bar, select another track, then **Paste config** to transfer its curves and processing settings. Subsequent edits remain independent. Time selections belong to each track and are not copied across songs. Copying a config does not change the track being copied.

Use the information button beside the spectrum controls for display and editing help; the graph itself has no hover tooltip. The white curve shows the current audio frame. The filled accent-colored region and its darker outline hold the maximum amplitude at each frequency during playback. It follows the selected A/B/comparison version or residual solo, before the output volume control. Channels are combined by power so opposite-phase stereo remains visible. The left axis shows amplitude in dBFS; the right axis shows EQ gain.

Scrubbing samples the frame at the playhead and preserves the filled peak envelope. Peaks fall slowly at 0.5 dB per second, including while paused, instead of disappearing when seeking or editing a band. The Analyzer settings tab offers slower fade, indefinite hold, and manual reset. Changing tracks resets the envelope.

**Outliers** compares whole-track original and cleaned outlier strength by frequency. Its white trace follows the current audio at the playback position, including seeking and A/B changes. The dashed line is the detector threshold. Scroll to zoom, middle-drag to pan, double-click to reset, or right-click to add a fixed notch.

**Spectrogram** maps outlier strength over time and frequency, with a labeled brightness scale and moving white playback cursor. This measures dB above the local spectral floor, rather than absolute volume. It follows the selected original, cleaned version, or removed audio. Click to seek, Shift-drag to select a loop, scroll to zoom frequency, or right-click to add a fixed notch. Ordinary dragging selects an adaptive noise-profile region. The information button explains these views without canvas hover popups.

Playback, comparison, and loop controls have separate bordered groups. Play/pause and stop are wide, icon-only controls. All bottom buttons are 40 pixels high. A/B is blue for original audio, green for cleaned audio, and amber for additional versions; removed-audio audition has its own amber active state. Other checked buttons use a bright neutral fill.

### Curve editor (default)

Click **+ Add curve** to add a bell immediately. Its separate arrow offers **Bell, Low shelf, High shelf, Low cut / high-pass, High cut / low-pass, Notch, Band pass, and Tilt**, with shape previews. Double-clicking the spectrum adds the selected shape at that frequency. Each band can be selected, disabled, or removed independently. **Ctrl/Shift-click** handles or numbered band buttons to select several, or use **Select all**. Drag, wheel, numeric values, dynamics, bypass and Delete apply to the selection. Group movement preserves frequency ratios; width and scale changes are proportional, and gain/threshold changes use a common dB offset.

- **Left-drag a handle:** frequency horizontally, height or depth vertically.
- **Wheel:** change the selected curve's width. One octave spans a frequency ratio of 2:1.
- **Right-drag a handle:** width horizontally, plateau vertically.
- **Alt+wheel:** scale overall size (width and height together).
- **Ctrl+wheel:** zoom the graph. **Middle-drag:** pan.

Selecting a handle opens a compact floating panel with dials and exact numeric values for frequency, Max gain, width, plateau, and Scale both. **Max gain** changes depth alone; **Scale both** multiplies both width and depth (150% makes both 1.5 times larger). The effective maximum and width are displayed below the dials. Drag a dial vertically, use its wheel, or type a value below it. Shift-drag gives fine adjustment; double-click a dial to reset it. Drag the panel by its grip at the top to move it out of the way. Its position is remembered and stays within the workspace after resizing. On narrow layouts the panel initially docks below the spectrum; dragging its grip makes it float. Plateau applies to bell, notch, and band-pass shapes. Shelves and cuts use smooth transitions; cuts and band pass use the selected depth as a finite attenuation floor. These are spectral EQ shapes rather than analog filter models. Colored EQ regions show the maximum permitted change. The yellow line follows the actual response at the playhead, using the envelopes recorded when the audio was rendered. **Show fixed cuts** optionally adds a subdued overview of automatic notches; the main editor hides them by default. Use **Outliers** to inspect exact outliers. Adaptive reduction and coherence weighting can produce less reduction than the automatic overview.

**Soften highs** opens Audio processing and retains the original Fade from / Full from controls, transition shapes, and left/right mouse endpoint editing. Use it for broad top-end harshness; EQ curves target specific bands. Both modes keep their settings, and only the active mode's manual shaping is applied. Outlier removal works with either mode. Curves and detector settings are included in saved presets, history, and batch processing.

The live spectrum uses separate vertical fills: current audio fades from a light accent at the crest to a dark base, with 72-91% transparency; the peak envelope behind it fades from a muted accent at the crest to near-black. The fill gradients start at the visible crest. Curve outlines keep their own colors.

### Dynamic processing and automatic starting curves

**Auto curves** detects the strongest narrow spectral outliers and creates up to six editable dynamic bell bands. Suggestions default to at least **0.65 octave**, with greater frequency separation to avoid a cluster of needle-like bands. Use the arrow beside **Auto curves** to choose a minimum width and regenerate suggestions. It calibrates each threshold from the track's actual band levels. Tracks with no existing bands receive suggestions automatically. Numbered frequency buttons above the graph select each band. Manually editing an automatic band preserves it when suggestions are refreshed; deleting the final band keeps the EQ empty until Auto curves is requested again.

A curve's **Max gain** defines its colored envelope. Each STFT frame measures the signal in each band, compares it with that band's **Threshold** in dBFS, and applies the **Ratio**, **Attack**, and **Release**. Quiet regions release toward zero adjustment; stronger peaks receive more adjustment, bounded by the maximum. Bands react independently. Stereo channels share the same detector envelope so their image stays stable. The default EQ FFT is 8192 points with 75% overlap; an explicitly chosen FFT size overrides it.

The response used for playback and export varies throughout the track. Rendered response envelopes are stored with comparison versions, so the graph follows the actual selected version at the playhead, including after seeking. Spectrum peak hold persists through scrubbing and curve edits; the EQ maximum region remains the configured limit. Disable **Dynamic** on a band for fixed gain.

Detector settings remain editable while **Fixed cuts** is off. That checkbox enables the legacy automatic notch stage; automatic dynamic bands replace it by default. **Show fixed cuts** (previously Auto cuts) only displays the response of that stage. It does not enable filtering or create dynamic bands.

### Detection rules

The previous **600 Hz** median span and **40 Hz** maximum peak width were heuristics, not universal constants. They remain the starting values and can now be edited. The time percentile defaults to **50%** (median over time); lower values favour more persistent tones, while higher values include more intermittent peaks. Minimum level is relative to the maximum of the track's median spectrum, not absolute dBFS.

The analyzer tab (**Settings > Analyzer**) defaults to smoother 1/12-octave display and offers fine, normal, or smooth fractional-octave smoothing, an EQ gain range, and peak reset. Its 32768-point FFT interpolates bass bins smoothly; only the visualization is smoothed. Outlier detection uses the track's longer-window time statistics (about 2.9 Hz per bin at typical sample rates), so narrow artifacts are easier to inspect in **Outlier detail**.

The analyzer uses native Qt painting with cached axes and EQ regions. A precise 6 ms timer targets **166 Hz**. The capture tool can measure full painting, timer pacing and grouped dragging without an audio device:

```powershell
.\venv\Scripts\python.exe tests/capture_ui.py --output artifacts/ui/benchmark.png --auto-curves --select-all --benchmark
```

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
build_gpu.bat
```

`build_gpu.bat` requires CUDA-enabled PyTorch in the venv.

## UI previews

Run `venv\Scripts\python.exe tests/capture_ui.py --output artifacts/ui/workspace.png` to capture the populated Qt workspace using a track from `input/`. The capture uses isolated app state and real audio/detection, without opening the audio device. Add `--width 1280 --height 820 --inspector outliers`, `--shape-menu`, or `--scrub` to inspect other states.

## Verification

Run `venv\Scripts\python.exe -m unittest discover -s tests -v` for DSP and offscreen desktop interaction checks. Tests use temporary settings/history, synthetic audio, and no audio output device.
