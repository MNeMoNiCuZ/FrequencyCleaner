#!/usr/bin/env python3
"""
FrequencyCleaner — AI Music Artifact Remover
Removes the "AI sound" from generated music: the dozens of steady,
narrow high-frequency tones these generators leave behind (removed with
notches), plus optional softening of the harsh top band (a high shelf).
"""

import re
import sys
import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, asdict, replace
from datetime import datetime
from pathlib import Path


def _preload_torch_dll():
    """On Windows, PyQt6 loads a C++ runtime that makes torch's c10.dll fail to
    initialise later. Loading c10.dll first (≈1 ms) lets torch import lazily."""
    if sys.platform != 'win32':
        return
    try:
        import os, ctypes, importlib.util
        spec = importlib.util.find_spec('torch')
        if spec and spec.submodule_search_locations:
            lib = os.path.join(spec.submodule_search_locations[0], 'lib')
            os.add_dll_directory(lib)
            ctypes.WinDLL(os.path.join(lib, 'c10.dll'))
    except Exception:
        pass


_preload_torch_dll()

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal, ndimage

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QSlider,
    QLabel, QComboBox, QGroupBox, QFileDialog, QMessageBox,
    QFrame, QSizePolicy, QStyle, QStyleOptionSlider,
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QSplitter,
    QGridLayout, QButtonGroup, QProgressBar, QRadioButton, QStackedWidget, QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSettings, QEvent
from PyQt6.QtGui import (QShortcut, QKeySequence, QColor, QIcon, QPixmap, QPainter,
                         QPolygonF, QCursor)
from PyQt6.QtCore import QPointF, QRectF

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# ──────────────────────────────────────────────────────────────
#  DARK THEME
# ──────────────────────────────────────────────────────────────

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QToolTip {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; padding: 4px;
}
QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    color: #cdd6f4;
    outline: none;
}
QListWidget::item { padding: 5px 6px; }
QListWidget::item:selected { background-color: #89b4fa; color: #1e1e2e; border-radius: 4px; }
QListWidget::item:hover    { background-color: #313244; border-radius: 4px; }

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background-color: #313244; border-radius: 4px; }
QMenu {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
}
QMenu::item { padding: 5px 20px; }
QMenu::item:selected { background-color: #313244; }

QLineEdit, QDoubleSpinBox, QSpinBox {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px;
    padding: 3px 6px;
}
QDoubleSpinBox:disabled, QSpinBox:disabled { color: #585b70; border-color: #313244; }

QPushButton {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px;
    padding: 6px 14px;
}
QPushButton:hover    { background-color: #45475a; }
QPushButton:pressed  { background-color: #585b70; }
QPushButton:disabled { background-color: #1e1e2e; color: #585b70; border-color: #313244; }
QPushButton#btn_primary { background-color: #89b4fa; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#btn_primary:hover { background-color: #b4befe; }
QPushButton#btn_primary:disabled { background-color: #313244; color: #585b70; }
QPushButton#btn_small { padding: 4px 8px; }
QPushButton#btn_link { background: transparent; border: none; color: #89b4fa; padding: 2px 0; text-align: left; }
QPushButton#btn_link:hover { color: #b4befe; }

QPushButton#btn_preset {
    background-color: #181825; border: 1px solid #45475a; border-radius: 8px;
    padding: 8px 6px; font-weight: bold;
}
QPushButton#btn_preset:checked { background-color: #a6e3a1; color: #1e1e2e; border-color: #a6e3a1; }
QPushButton#btn_preset:hover:!checked { background-color: #313244; }

QPushButton#btn_ab_a   { background-color: #89b4fa; color: #1e1e2e; font-weight: bold; }
QPushButton#btn_ab_b   { background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; }
QPushButton#btn_ab_c   { background-color: #45475a; color: #f9e2af; font-weight: bold; }
QPushButton#btn_ab_c:checked { background-color: #f9e2af; color: #1e1e2e; }
QPushButton#btn_ab_a:disabled, QPushButton#btn_ab_b:disabled, QPushButton#btn_ab_c:disabled {
    background-color: #1e1e2e; color: #585b70; border-color: #313244;
}

QSlider::groove:horizontal { height: 6px; background: #313244; border-radius: 3px; }
QSlider::handle:horizontal { background: #89b4fa; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
QSlider::handle:horizontal:disabled { background: #45475a; }
QSlider::sub-page:horizontal:disabled { background: #313244; }

QFrame#card { background-color: #1e1e2e; border: 1px solid #313244; border-radius: 8px; }
QPushButton#btn_card_head {
    background: transparent; border: none; color: #89b4fa;
    font-weight: bold; font-size: 13px; padding: 3px 0; text-align: left;
}
QPushButton#btn_card_head:hover { color: #b4befe; }
QGroupBox {
    border: 1px solid #313244; border-radius: 8px;
    margin-top: 16px; padding: 10px 8px 8px 8px;
    color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; left: 10px; }
QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid #585b70; border-radius: 3px; background: #181825; }
QGroupBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; image: url(@CHECK@); }

QComboBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px;
    padding: 4px 8px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #313244; color: #cdd6f4;
    selection-background-color: #89b4fa; selection-color: #1e1e2e;
    border: 1px solid #45475a;
}
QCheckBox { spacing: 6px; }
QRadioButton { spacing: 6px; }
QRadioButton::indicator { width: 13px; height: 13px; border: 1px solid #585b70; border-radius: 7px; background: #181825; }
QRadioButton::indicator:checked {
    border-color: #89b4fa;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 #89b4fa, stop:0.5 #89b4fa, stop:0.6 #181825, stop:1 #181825);
}
QRadioButton:disabled, QCheckBox:disabled { color: #585b70; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #585b70; border-radius: 3px; background: #181825; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; image: url(@CHECK@); }

QTableWidget {
    background-color: #181825; alternate-background-color: #1b1b2b;
    border: 1px solid #313244; border-radius: 6px;
    gridline-color: #313244; outline: none;
}
QTableWidget::item:selected { background-color: #45475a; color: #cdd6f4; }
QTableWidget::indicator { width: 13px; height: 13px; border: 1px solid #585b70; border-radius: 3px; background: #181825; }
QTableWidget::indicator:checked { background: #f38ba8; border-color: #f38ba8; image: url(@CHECK@); }
QHeaderView::section {
    background-color: #1e1e2e; color: #6c7086;
    border: none; border-bottom: 1px solid #313244; padding: 3px 4px;
}
QTableCornerButton::section { background-color: #1e1e2e; border: none; }

QProgressBar {
    background-color: #181825; border: 1px solid #313244; border-radius: 4px;
    max-height: 8px; text-align: center; color: transparent;
}
QProgressBar::chunk { background-color: #f9e2af; border-radius: 3px; }

QLabel#lbl_section { color: #89b4fa; font-weight: bold; font-size: 11px; letter-spacing: 1px; }
QLabel#lbl_hint    { color: #7f849c; font-size: 11px; }
QLabel#lbl_metric  { font-size: 13px; font-weight: bold; }
QLabel#lbl_big     { font-size: 18px; font-weight: bold; }
QLabel#lbl_busy {
    background-color: #f9e2af; color: #1e1e2e;
    font-size: 15px; font-weight: bold;
    border-radius: 16px; padding: 8px 22px;
}
QLabel#lbl_steps { background-color: #181825; border: 1px solid #313244; border-radius: 8px; padding: 6px 10px; }
QFrame#transport   { background-color: #181825; border: 1px solid #313244; border-radius: 8px; }
QFrame#transport QLabel { background-color: transparent; }
QSplitter::handle { background-color: #313244; }
QSplitter::handle:vertical { height: 3px; }
QScrollArea { border: none; }
QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
QStatusBar::item { border: none; }
QScrollBar:vertical { background: #181825; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #181825; height: 8px; border-radius: 4px; }
QScrollBar::handle:horizontal { background: #45475a; border-radius: 4px; min-width: 20px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
""".replace('@CHECK@', (Path(__file__).resolve().parent / 'icons' / 'check.svg').as_posix())

C = {
    'bg':   '#1e1e2e', 'ax_bg': '#181825',
    'orig': '#89b4fa', 'proc':  '#a6e3a1', 'res': '#f9e2af',
    'fade': '#fab387', 'cut':   '#f38ba8', 'off': '#6c7086',
    'grid': '#313244', 'text':  '#cdd6f4',
}


def fmt_time(t: float, decimals: bool = False) -> str:
    t = max(0.0, float(t))
    m, s = divmod(t, 60.0)
    return f'{int(m)}:{s:04.1f}' if decimals else f'{int(m)}:{int(s):02d}'


# ──────────────────────────────────────────────────────────────
#  AUDIO I/O
# ──────────────────────────────────────────────────────────────

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.aiff', '.aif', '.m4a', '.mp4', '.aac', '.opus'}

# key: (label, extension, soundfile format, subtype). 'same' keeps the input container.
EXPORT_FORMATS = {
    'wav24':  ('WAV 24-bit',              '.wav',  'WAV',  'PCM_24'),
    'wav32f': ('WAV 32-bit float',        '.wav',  'WAV',  'FLOAT'),
    'flac24': ('FLAC 24-bit',             '.flac', 'FLAC', 'PCM_24'),
    'same':   ('Same as input if lossless (MP3/OGG/M4A → FLAC 24-bit)', None, None, None),
}

# Only lossless containers are kept on "same as input"; lossy inputs are never re-encoded.
_EXT_TO_SF_FORMAT = {'.wav': 'WAV', '.flac': 'FLAC', '.aiff': 'AIFF', '.aif': 'AIFF'}


def _read_audio_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    """Decode formats libsndfile cannot read (m4a/mp4/aac/opus) through ffmpeg."""
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError('ffmpeg/ffprobe not found on PATH')
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=sample_rate,channels', '-of', 'json', path],
        capture_output=True, check=True, creationflags=flags)
    streams = json.loads(probe.stdout or b'{}').get('streams', [])
    if not streams:
        raise RuntimeError('No audio stream found')
    sr, ch = int(streams[0]['sample_rate']), int(streams[0]['channels'])
    dec = subprocess.run(
        [ffmpeg, '-v', 'error', '-i', path, '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'],
        capture_output=True, check=True, creationflags=flags)
    raw = np.frombuffer(dec.stdout, dtype='<f4')
    n = len(raw) // ch
    return raw[:n * ch].reshape(n, ch).copy(), sr


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Returns (audio (N, ch) float32, sample_rate)."""
    try:
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        return audio, sr
    except Exception as sf_err:
        try:
            return _read_audio_ffmpeg(path)
        except FileNotFoundError:
            raise RuntimeError(f'{sf_err}  (install ffmpeg to open this format)')


def output_path(src: str, out_dir: Path, suffix: str, fmt_key: str) -> tuple[Path, str | None, str | None]:
    """Destination path plus soundfile (format, subtype) for an export."""
    p = Path(src)
    _, ext, fmt, subtype = EXPORT_FORMATS.get(fmt_key, EXPORT_FORMATS['wav24'])
    if ext is None:   # same as input
        sf_fmt = _EXT_TO_SF_FORMAT.get(p.suffix.lower())
        if sf_fmt in sf.available_formats():
            return out_dir / f'{p.stem}{suffix}{p.suffix}', sf_fmt, 'PCM_24'
        _, ext, fmt, subtype = EXPORT_FORMATS['flac24']
    return out_dir / f'{p.stem}{suffix}{ext}', fmt, subtype


def write_audio(path: Path, audio: np.ndarray, sr: int, fmt: str | None, subtype: str | None):
    if subtype != 'FLOAT':
        audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr, format=fmt, subtype=subtype)


def read_json_text(text: str, default):
    try:
        return json.loads(text) if text else default
    except (TypeError, ValueError):
        return default


def read_json(path: Path, default):
    try:
        return read_json_text(path.read_text(encoding='utf-8'), default)
    except OSError:
        return default


def write_json(path: Path, data) -> str | None:
    """Write via a temp file so a crash never leaves half a file. Returns an error or None."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        tmp.replace(path)
        return None
    except (OSError, TypeError, ValueError) as e:
        return str(e)


# ──────────────────────────────────────────────────────────────
#  FILTER SETTINGS + PRESETS
# ──────────────────────────────────────────────────────────────

@dataclass
class FilterSettings:
    preset:         str   = 'normal'     # 'gentle' | 'normal' | 'strong' | 'custom'

    # AI tones (narrow steady peaks → notches)
    tones_on:       bool  = True
    threshold_db:   float = 4.0          # a peak this far above the local floor is a tone
    min_hz:         float = 5000.0       # search for tones above this frequency
    second_pass:    bool  = True         # re-detect on the cleaned result to catch leftovers
    tone_depth_db:  float = -40.0
    notches:        list  = field(default_factory=list)   # [{'freq', 'width', 'on', 'prom'}]

    # High-band softening (shelf)
    shelf_on:       bool  = True
    fade_start:     float = 14000.0
    cutoff:         float = 18000.0
    curve:          str   = 'cosine'     # 'cosine' | 'linear' | 'steep'
    shelf_depth_db: float = -12.0

    # Advanced
    method:         str   = 'static'     # 'static' | 'adaptive' (spectral subtraction)
    oversub:        float = 2.0
    percentile:     float = 50.0
    smoothing:      bool  = True
    attack_ms:      float = 5.0
    release_ms:     float = 50.0
    stereo:         str   = 'lr'         # 'lr' | 'mid' | 'coherence'
    resolution:     int   = 0            # STFT size, 0 = auto
    profile_region: tuple | None = None  # (t0, t1) s for the adaptive noise profile

    _TRANSIENT = ('notches', 'profile_region')

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps({k: v for k, v in d.items() if k not in self._TRANSIENT})

    @classmethod
    def from_json(cls, text: str) -> 'FilterSettings':
        try:
            data = json.loads(text) if text else {}
        except (TypeError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        s = cls.from_dict({k: v for k, v in data.items() if k not in cls._TRANSIENT})
        if s.preset in PRESETS:
            s.apply_preset(s.preset)
        return s

    def to_dict(self, notches: bool = True) -> dict:
        """Full snapshot (history) or, with notches=False, only the knobs (presets)."""
        d = asdict(self)
        if notches:
            d['profile_region'] = list(self.profile_region) if self.profile_region else None
        else:
            for k in self._TRANSIENT:
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'FilterSettings':
        s = cls()
        if not isinstance(data, dict):
            return s
        names = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k not in names:
                continue
            if k == 'notches':
                if isinstance(v, list):
                    s.notches = [dict(n) for n in v if isinstance(n, dict) and 'freq' in n and 'width' in n]
            elif k == 'profile_region':
                try:
                    s.profile_region = (float(v[0]), float(v[1])) if v else None
                except (TypeError, ValueError, IndexError):
                    s.profile_region = None
            else:
                try:
                    setattr(s, k, type(getattr(s, k))(v))
                except (TypeError, ValueError):
                    pass
        return s

    def apply_preset(self, name: str):
        self.preset = name
        for k, v in PRESETS[name]['values'].items():
            setattr(self, k, v)

    def copy(self) -> 'FilterSettings':
        return replace(self, notches=[dict(n) for n in self.notches])


PRESETS = {
    'gentle': {
        'label': 'Gentle',
        'text':  'Removes the clearer AI tones (one pass) and takes a light edge off the very top '
                 '(from 15 kHz).',
        'values': dict(tones_on=True, threshold_db=5.0, min_hz=5000.0, second_pass=False,
                       tone_depth_db=-35.0, shelf_on=True, fade_start=15000.0, cutoff=18500.0,
                       shelf_depth_db=-6.0, method='static'),
    },
    'normal': {
        'label': 'Normal',
        'text':  'Removes all AI tones (two passes) and softens the harsh top band a little '
                 '(from 14 kHz).',
        'values': dict(tones_on=True, threshold_db=4.0, min_hz=5000.0, second_pass=True,
                       tone_depth_db=-40.0, shelf_on=True, fade_start=14000.0, cutoff=18000.0,
                       shelf_depth_db=-12.0, method='static'),
    },
    'strong': {
        'label': 'Strong',
        'text':  'Also catches faint tones and tames the top band hard (from 12 kHz). '
                 'Can dull cymbals — compare with A/B.',
        'values': dict(tones_on=True, threshold_db=3.0, min_hz=4000.0, second_pass=True,
                       tone_depth_db=-60.0, shelf_on=True, fade_start=12000.0, cutoff=17000.0,
                       shelf_depth_db=-24.0, method='static'),
    },
}
PRESET_FIELDS = set(PRESETS['normal']['values'])
DETECT_FIELDS = {'tones_on', 'threshold_db', 'min_hz', 'second_pass'}


# ──────────────────────────────────────────────────────────────
#  CLICK-TO-SEEK SLIDER  (Qt default moves by pageStep on track click)
# ──────────────────────────────────────────────────────────────

class ClickSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderGroove, self)
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderHandle, self)
            if self.orientation() == Qt.Orientation.Horizontal:
                sl_min = groove.x() + handle.width() // 2
                sl_max = groove.right() - handle.width() // 2
                pos = event.pos().x()
            else:
                sl_min = groove.y() + handle.height() // 2
                sl_max = groove.bottom() - handle.height() // 2
                pos = event.pos().y()
            self.setValue(QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), pos - sl_min, sl_max - sl_min))
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────
#  COMPUTE BACKEND
#
#  STFT / ISTFT / PSD frames run on the GPU through PyTorch + CUDA when it is
#  installed and enabled, otherwise on the CPU with scipy. Channels are always
#  processed in parallel threads (FFTs release the GIL).
# ──────────────────────────────────────────────────────────────

class GPU:
    enabled = True
    _torch  = None
    _name   = ''
    _checked = False
    _lock   = threading.Lock()
    _windows: dict = {}

    @classmethod
    def probe(cls):
        """Import torch once (slow) — called in the background at startup."""
        with cls._lock:
            if cls._checked:
                return
            cls._checked = True
            try:
                import torch
                if torch.cuda.is_available():
                    cls._torch = torch
                    cls._name  = torch.cuda.get_device_name(0)
            except Exception:
                cls._torch = None

    @classmethod
    def warm_up(cls):
        """Import torch and pay CUDA / cuFFT start-up before the first clean."""
        cls.probe()
        torch = cls._torch
        if torch is None:
            return
        try:
            for n in (16384, 4096):
                x = torch.zeros(n * 4, device='cuda')
                Z = torch.stft(x, n, n // 4, window=cls.window(n), center=True,
                               pad_mode='constant', return_complex=True)
                torch.istft(Z, n, n // 4, window=cls.window(n), center=True, length=n * 4)
                torch.fft.rfft(x.unfold(0, n, n // 2), dim=1)
            torch.cuda.synchronize()
        except Exception:
            pass

    last_error = ''   # set when a GPU call failed and the CPU took over

    @classmethod
    def fail(cls, e: Exception):
        cls.last_error = str(e).splitlines()[0][:120] if str(e) else type(e).__name__

    @classmethod
    def label(cls) -> str:
        """What processing runs on, for the UI."""
        if cls.torch() is not None:
            return f'GPU · {cls._name}'
        if cls.available():
            return 'CPU (GPU turned off in Settings)'
        return 'CPU (no CUDA GPU / PyTorch found)'

    @classmethod
    def available(cls) -> bool:
        cls.probe()
        return cls._torch is not None

    @classmethod
    def device_name(cls) -> str:
        return cls._name if cls.available() else ''

    @classmethod
    def torch(cls):
        """The torch module when GPU processing is on and possible, else None."""
        return cls._torch if cls.enabled and cls.available() else None

    @classmethod
    def window(cls, n: int):
        with cls._lock:
            w = cls._windows.get(n)
            if w is None:
                w = cls._torch.hann_window(n, periodic=True, dtype=cls._torch.float32, device='cuda')
                cls._windows[n] = w
        return w

    @classmethod
    def tensor(cls, x: np.ndarray):
        return cls._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to('cuda')


_POOL = ThreadPoolExecutor(max_workers=8)


def _pmap(fn, items: list) -> list:
    """Map over channels in parallel (never call from inside a pool task)."""
    if len(items) <= 1:
        return [fn(i) for i in items]
    return list(_POOL.map(fn, items))


def _median_bias(n: int) -> float:
    """Same bias correction scipy's median Welch uses."""
    ii = 2 * np.arange(1, (n - 1) // 2 + 1)
    return float(1.0 + np.sum(1.0 / (ii + 1) - 1.0 / ii))


# ──────────────────────────────────────────────────────────────
#  FILTER PROCESSOR
#
#  On load : median-averaged Welch PSD (2.9 Hz/bin) + its local floor, and a
#            "tone lines" spectrogram (level above the local floor per frame).
#
#  Tones   : any narrow peak in the median PSD that stands ≥ threshold dB above
#            the local floor. A peak only survives median averaging if it is
#            present most of the time — i.e. a steady tone, not a note.
#            A second pass re-detects on the cleaned signal to catch leftovers.
#
#  Apply   : two stages, each STFT → gain on the targeted rows → ISTFT of the
#            removed part, subtracted from the input (bins outside the target
#            pass through exactly).
#              1. tones at high resolution (notches)
#              2. top-band shelf at normal resolution
# ──────────────────────────────────────────────────────────────

class FilterProcessor:

    TONE_RES_HZ   = 2.93    # Hz/bin for tone detection + notches (16384 @ 44.1/48 kHz)
    SHELF_RES_HZ  = 11.7    # Hz/bin for the shelf (4096 @ 44.1/48 kHz)
    VIEW_MIN_HZ   = 2000.0
    VIEW_MAX_COLS = 1400
    VIEW_MAX_ROWS = 480
    STFT_OVERLAP  = 3       # 75 % overlap (hop = nperseg / 4)
    WEIGHT_EPS    = 1e-3

    FLOOR_HZ          = 600.0   # span of the local spectral floor
    MIN_LEVEL_DB      = -90.0   # below this (relative to peak) is codec silence
    TONE_MAX_FWHM_HZ  = 40.0    # wider than this is not a tone
    TONE_MAX_WIDTH_HZ = 30.0    # notch width cap
    TONE_MAX_COUNT    = 400

    def __init__(self):
        self._audio    = None   # (N, ch) float32
        self._mono     = None
        self._sr       = 0
        self._n_ch     = 0
        self._welch_f  = None
        self._welch_db = None   # raw dB
        self._psd_ref  = 0.0
        self._floor    = None   # local floor of the original PSD (relative dB)
        self._audible  = None
        self.bandwidth = 0.0    # highest frequency with real content (e.g. MP3 lowpass)
        self._view     = None

    # ── Properties ───────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._audio is not None

    @property
    def audio(self) -> np.ndarray | None:
        return self._audio

    @property
    def sr(self) -> int:
        return self._sr

    @property
    def n_ch(self) -> int:
        return self._n_ch

    @property
    def duration(self) -> float:
        return self._audio.shape[0] / self._sr if self.loaded else 0.0

    # ── Load ─────────────────────────────────────────────────

    def load(self, audio: np.ndarray, sr: int):
        if audio.ndim != 2 or audio.shape[0] < 4096:
            raise ValueError('Audio is too short to analyse')
        self._audio = np.ascontiguousarray(audio, dtype=np.float32)
        self._sr    = int(sr)
        self._n_ch  = audio.shape[1]
        self._mono  = self._audio.mean(axis=1) if self._n_ch > 1 else self._audio[:, 0]

        GPU.probe()
        self._welch_f, self._welch_db = self._welch(self._mono)
        self._psd_ref = float(self._welch_db.max())
        rel = self._welch_db - self._psd_ref
        self._audible = rel > self.MIN_LEVEL_DB
        last = int(np.nonzero(self._audible)[0][-1]) if np.any(self._audible) else len(rel) - 1
        self.bandwidth = float(self._welch_f[last])
        self._floor = self._local_floor(rel, last)

        # Tone-lines view: level above the (time-invariant) local floor, with each
        # frame's overall loudness taken out, so steady tones stay bright lines.
        db, rows, extent = self._view_db(self._mono)
        white = db - self._floor[rows][:, None] + self._psd_ref
        offset = np.median(white, axis=0, keepdims=True)
        self._view = {'rows': rows, 'offset': offset, 'extent': extent,
                      'orig': self._pool(white - offset)}

    # ── Resolution helpers ───────────────────────────────────

    @staticmethod
    def pow2_for(sr: int, hz_per_bin: float) -> int:
        return int(2 ** round(np.log2(sr / hz_per_bin)))

    @staticmethod
    def _fit(n: int, n_samples: int) -> int:
        while n > n_samples and n > 256:
            n //= 2
        return n

    def _n_samples(self) -> int:
        return self._audio.shape[0] if self.loaded else 1 << 30

    def nperseg_tones(self, s: FilterSettings, sr: int | None = None) -> int:
        n = int(s.resolution) or self.pow2_for(sr or self._sr or 48000, self.TONE_RES_HZ)
        return self._fit(n, self._n_samples())

    def nperseg_shelf(self, s: FilterSettings, sr: int | None = None) -> int:
        n = int(s.resolution) or self.pow2_for(sr or self._sr or 48000, self.SHELF_RES_HZ)
        return self._fit(n, self._n_samples())

    def _psd_frames_gpu(self, x: np.ndarray, nperseg: int, hop: int):
        """Per-segment one-sided PSD (segments, bins) on the GPU, matching scipy's
        welch / spectrogram (hann, constant detrend, density scaling). None → use CPU."""
        torch = GPU.torch()
        if torch is None or len(x) < nperseg:
            return None
        try:
            win = GPU.window(nperseg)
            fr = GPU.tensor(x).unfold(0, nperseg, hop)
            fr = (fr - fr.mean(dim=1, keepdim=True)) * win
            Z = torch.fft.rfft(fr, dim=1)
            del fr
            p = Z.real.square() + Z.imag.square()
            del Z
            p *= 1.0 / (self._sr * float((win * win).sum()))
            p[:, 1:(nperseg + 1) // 2] *= 2.0
            return p
        except Exception as e:   # out of memory etc.
            GPU.fail(e)
            return None

    def _welch(self, x: np.ndarray) -> tuple:
        nperseg = self._fit(self.pow2_for(self._sr, self.TONE_RES_HZ), len(x))
        p = self._psd_frames_gpu(x, nperseg, nperseg // 2)
        if p is not None:
            k = p.shape[0]
            s = p.sort(dim=0).values
            med = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
            psd = (med / _median_bias(k)).cpu().numpy()
            del p, s
            f = np.fft.rfftfreq(nperseg, 1.0 / self._sr)
        else:
            f, psd = signal.welch(x, self._sr, window='hann', nperseg=nperseg, average='median')
        return f, 10.0 * np.log10(psd + 1e-20)

    def _local_floor(self, rel: np.ndarray, last_audible: int) -> np.ndarray:
        """Median of the surrounding ~600 Hz; flat past the codec lowpass so the
        cliff does not make tones just below it look extra prominent."""
        df = float(self._welch_f[1] - self._welch_f[0])
        n = max(int(self.FLOOR_HZ / df) | 1, 31)
        x = rel.copy()
        x[last_audible + 1:] = np.median(x[max(0, last_audible - n):last_audible + 1])
        return ndimage.median_filter(x, size=n, mode='nearest')

    def _view_db(self, x: np.ndarray) -> tuple:
        """Spectrogram rows on the same frequency grid as the Welch PSD, lightly
        smoothed in time (~1 s) so steady tones stand out from the noise."""
        nperseg = self._fit(self.pow2_for(self._sr, self.TONE_RES_HZ), len(x))
        hop = int(np.clip(np.ceil(len(x) / self.VIEW_MAX_COLS), nperseg // 2, nperseg))
        p = self._psd_frames_gpu(x, nperseg, hop)
        if p is not None:
            S = p.T.contiguous().cpu().numpy()
            del p
            f = np.fft.rfftfreq(nperseg, 1.0 / self._sr)
            t = (np.arange(S.shape[1]) * hop + nperseg / 2.0) / self._sr
        else:
            f, t, S = signal.spectrogram(x, self._sr, window='hann', nperseg=nperseg,
                                         noverlap=nperseg - hop, mode='psd')
        top = min(self.bandwidth + 1000.0, self._sr / 2.0)
        rows = (f >= self.VIEW_MIN_HZ) & (f <= top)
        db = (10.0 * np.log10(S[rows] + 1e-20)).astype(np.float32)
        smooth = max(int(round(1.0 / (hop / self._sr))), 1)
        if smooth > 1:
            db = ndimage.uniform_filter1d(db, smooth, axis=1, mode='nearest')
        half = hop / self._sr / 2.0
        extent = (float(t[0] - half), float(t[-1] + half),
                  float(f[rows][0]) / 1000.0, float(f[rows][-1]) / 1000.0)   # y in kHz
        return db, rows, extent

    def _pool(self, img: np.ndarray) -> np.ndarray:
        """Max-pool frequency rows down to screen-ish resolution: thin tone
        lines survive instead of being averaged away."""
        g = int(np.ceil(img.shape[0] / self.VIEW_MAX_ROWS))
        if g <= 1:
            return img.astype(np.float32)
        n = img.shape[0] // g * g
        return img[:n].reshape(n // g, g, -1).max(axis=1).astype(np.float32)

    def _view_of(self, x: np.ndarray) -> np.ndarray:
        """Tone-lines image of a processed signal, relative to the original's floor."""
        db, _, _ = self._view_db(x)
        white = db - self._floor[self._view['rows']][:, None] + self._psd_ref
        return self._pool(white - self._view['offset'])

    def tone_prominence(self) -> np.ndarray:
        """How far each bin of the original stands above its surroundings (dB)."""
        return self._welch_db - self._psd_ref - self._floor

    # ── Display data ─────────────────────────────────────────

    def spectrum_original(self) -> tuple:
        if self._welch_f is None:
            return np.array([]), np.array([])
        return self._welch_f.copy(), self._welch_db - self._psd_ref

    def tone_view(self) -> dict | None:
        return self._view

    # ── Weights (0 = untouched, 1 = fully targeted) ──────────

    @staticmethod
    def shelf_weight(freqs: np.ndarray, fade_start: float, cutoff: float, curve: str) -> np.ndarray:
        w = np.zeros(len(freqs), dtype=np.float64)
        if cutoff > fade_start:
            idx = (freqs >= fade_start) & (freqs < cutoff)
            if np.any(idx):
                t = (freqs[idx] - fade_start) / (cutoff - fade_start)
                if curve == 'linear':
                    w[idx] = t
                elif curve == 'steep':
                    w[idx] = 1.0 - np.cos(t * np.pi / 2.0) ** 4
                else:   # cosine
                    w[idx] = (1.0 - np.cos(t * np.pi)) / 2.0
        w[freqs >= cutoff] = 1.0
        return w

    @staticmethod
    def notch_weight(freqs: np.ndarray, notches: list) -> np.ndarray:
        """Gaussian notches; 'width' is the full width at half weight."""
        passthrough = np.ones(len(freqs), dtype=np.float64)
        for n in notches:
            if not n.get('on', True):
                continue
            half = max(float(n['width']), 0.1) / 2.0
            d = (freqs - float(n['freq'])) / half
            near = np.abs(d) < 4.0
            passthrough[near] *= 1.0 - np.exp(-np.log(2.0) * d[near] ** 2)
        return 1.0 - passthrough

    def shelf_gain_db(self, freqs: np.ndarray, s: FilterSettings) -> np.ndarray:
        if not s.shelf_on:
            return np.zeros(len(freqs))
        w = self.shelf_weight(freqs, s.fade_start, s.cutoff, s.curve)
        g = 1.0 - w * (1.0 - 10.0 ** (min(s.shelf_depth_db, 0.0) / 20.0))
        return 20.0 * np.log10(np.maximum(g, 1e-6))

    # ── Tone detection ───────────────────────────────────────

    def _find_tones(self, rel: np.ndarray, thr: float, min_hz: float) -> tuple:
        f = self._welch_f
        df = float(f[1] - f[0])
        prom = rel - self._floor
        pk, _ = signal.find_peaks(prom, height=thr, distance=2)
        pk = pk[(f[pk] >= min_hz) & self._audible[pk] & (f[pk] < 0.49 * self._sr)]
        if pk.size == 0:
            return pk, prom, np.array([])
        fwhm = signal.peak_widths(prom, pk, rel_height=0.5)[0] * df
        keep = fwhm <= self.TONE_MAX_FWHM_HZ
        return pk[keep], prom, fwhm[keep]

    def detect_tones(self, thr: float, min_hz: float, rel: np.ndarray | None = None) -> list:
        if not self.loaded:
            return []
        if rel is None:
            rel = self._welch_db - self._psd_ref
        pk, prom, fwhm = self._find_tones(rel, thr, min_hz)
        if pk.size > self.TONE_MAX_COUNT:
            order = np.argsort(prom[pk])[::-1][:self.TONE_MAX_COUNT]
            pk, fwhm = pk[order], fwhm[order]
        df = float(self._welch_f[1] - self._welch_f[0])
        widths = np.clip(2.0 * fwhm, 4.0 * df, self.TONE_MAX_WIDTH_HZ)
        tones = [{'freq': round(float(self._welch_f[k]), 1), 'width': round(float(w), 1),
                  'on': True, 'prom': round(float(prom[k]), 1)} for k, w in zip(pk, widths)]
        return sorted(tones, key=lambda t: t['freq'])

    def count_tones(self, rel: np.ndarray, thr: float, min_hz: float) -> int:
        return int(self._find_tones(rel, thr, min_hz)[0].size)

    @staticmethod
    def _merge_tones(base: list, extra: list) -> list:
        out = [dict(t) for t in base]
        for e in extra:
            if not any(abs(e['freq'] - t['freq']) < max(t['width'] / 2.0, 3.0) for t in out):
                out.append(dict(e))
        return sorted(out, key=lambda t: t['freq'])

    # ── Full clean (detect → optional second pass → apply) ────

    def clean(self, s: FilterSettings, detect: bool, progress=None) -> tuple[list, np.ndarray]:
        say = progress or (lambda _msg: None)
        s = s.copy()
        if detect and s.tones_on:
            say('Finding AI tones …')
            s.notches = self.detect_tones(s.threshold_db, s.min_hz)
            if s.second_pass and s.notches:
                say(f'Second pass — {len(s.notches)} tones found, checking for leftovers …')
                tones_only = s.copy()
                tones_only.shelf_on = False
                out = self.apply(tones_only)
                _, d = self._welch(out.mean(axis=1))
                extra = self.detect_tones(s.threshold_db, s.min_hz, d - self._psd_ref)
                s.notches = self._merge_tones(s.notches, extra)
                del out
        say('Cleaning …')
        return s.notches, self.apply(s)

    # ── STFT helpers ─────────────────────────────────────────

    def _stft(self, x: np.ndarray, nperseg: int) -> np.ndarray:
        _, _, Z = signal.stft(x, fs=self._sr, window='hann', nperseg=nperseg,
                              noverlap=nperseg * self.STFT_OVERLAP // 4,
                              return_onesided=True)
        return Z

    def _istft(self, Z: np.ndarray, nperseg: int, n: int) -> np.ndarray:
        _, x = signal.istft(Z, fs=self._sr, window='hann', nperseg=nperseg,
                            noverlap=nperseg * self.STFT_OVERLAP // 4)
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        return x.astype(np.float32)

    def _hop(self, nperseg: int) -> int:
        return nperseg - nperseg * self.STFT_OVERLAP // 4

    def _stft_rows(self, x: np.ndarray, nperseg: int, rows: np.ndarray) -> tuple:
        """Only the targeted rows of the STFT, plus the frame count. Scaled like scipy."""
        torch = GPU.torch()
        if torch is not None:
            try:
                win = GPU.window(nperseg)
                Z = torch.stft(GPU.tensor(x), nperseg, self._hop(nperseg),
                               window=win, center=True, pad_mode='constant', return_complex=True)
                band = Z.index_select(0, torch.from_numpy(rows).to('cuda'))
                band = (band / float(win.sum())).cpu().numpy()
                return band, Z.shape[1]
            except Exception as e:
                GPU.fail(e)
        Z = self._stft(x, nperseg)
        return Z[rows], Z.shape[1]

    def _istft_rows(self, removed: np.ndarray, rows: np.ndarray, nperseg: int,
                    frames: int, n: int) -> np.ndarray:
        torch = GPU.torch()
        if torch is not None and removed.shape[1] == frames:
            try:
                win = GPU.window(nperseg)
                Zr = torch.zeros((nperseg // 2 + 1, frames), dtype=torch.complex64, device='cuda')
                Zr[torch.from_numpy(rows).to('cuda')] = \
                    torch.from_numpy(np.ascontiguousarray(removed, dtype=np.complex64)).to('cuda') \
                    * float(win.sum())
                y = torch.istft(Zr, nperseg, self._hop(nperseg), window=win,
                                center=True, length=n)
                return y.cpu().numpy().astype(np.float32)
            except Exception as e:
                GPU.fail(e)
        Zr = np.zeros((nperseg // 2 + 1, frames), dtype=np.complex64)
        Zr[rows] = removed
        return self._istft(Zr, nperseg, n)

    @staticmethod
    def _coherence(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
        """Magnitude-squared coherence per bin, averaged over time."""
        sxy = np.mean(za * np.conj(zb), axis=1, dtype=np.complex128)
        sxx = np.mean(np.abs(za) ** 2, axis=1, dtype=np.float64)
        syy = np.mean(np.abs(zb) ** 2, axis=1, dtype=np.float64)
        return np.clip(np.abs(sxy) ** 2 / (sxx * syy + 1e-30), 0.0, 1.0)

    def _slew(self, G: np.ndarray, hop: int, attack_ms: float, release_ms: float) -> np.ndarray:
        hop_s = hop / self._sr
        a_up = float(np.exp(-hop_s / (attack_ms / 1000.0))) if attack_ms > 0 else 0.0
        a_dn = float(np.exp(-hop_s / (release_ms / 1000.0))) if release_ms > 0 else 0.0
        if a_up < 1e-4 and a_dn < 1e-4:
            return G
        out  = np.empty_like(G)
        prev = G[:, 0].copy()
        out[:, 0] = prev
        for k in range(1, G.shape[1]):
            cur  = G[:, k]
            a    = np.where(cur > prev, a_up, a_dn).astype(G.dtype)
            prev = a * prev + (1.0 - a) * cur
            out[:, k] = prev
        return out

    def _adaptive_gain(self, band: np.ndarray, s: FilterSettings, hop: int, floor: float) -> np.ndarray:
        mag    = np.abs(band)
        frames = mag.shape[1]
        sel    = slice(None)
        if s.profile_region:
            t0, t1 = sorted(s.profile_region)
            k0 = int(np.clip(round(t0 * self._sr / hop), 0, frames - 1))
            k1 = int(np.clip(round(t1 * self._sr / hop) + 1, 0, frames))
            if k1 - k0 >= 3:
                sel = slice(k0, k1)
        noise = np.percentile(mag[:, sel], s.percentile, axis=1).astype(np.float32)
        G = 1.0 - np.float32(s.oversub) * noise[:, None] / (mag + np.float32(1e-12))
        np.maximum(G, np.float32(floor), out=G)
        if s.smoothing and min(G.shape) >= 3:
            G = ndimage.median_filter(G, size=(3, 3), mode='nearest')
        return self._slew(G, hop, s.attack_ms, s.release_ms)

    # ── Apply ────────────────────────────────────────────────

    def _run_stage(self, audio: np.ndarray, s: FilterSettings, weight_fn, depth_db: float,
                   nperseg: int) -> np.ndarray:
        n     = audio.shape[0]
        hop   = self._hop(nperseg)
        freqs = np.fft.rfftfreq(nperseg, 1.0 / self._sr)
        W     = weight_fn(freqs)
        rows  = np.nonzero(W > self.WEIGHT_EPS)[0]
        if rows.size == 0:
            return audio
        w     = W[rows].astype(np.float32)
        depth = float(10.0 ** (min(depth_db, 0.0) / 20.0))

        mid_only = self._n_ch == 2 and s.stereo == 'mid'
        if mid_only:
            side  = (audio[:, 0] - audio[:, 1]) * 0.5
            chans = [(audio[:, 0] + audio[:, 1]) * 0.5]
        else:
            chans = [audio[:, c] for c in range(audio.shape[1])]

        # Keep only the targeted rows; channels run in parallel.
        stfts  = _pmap(lambda x: self._stft_rows(x, nperseg, rows), chans)
        bands  = [b for b, _ in stfts]
        frames = stfts[0][1]
        del stfts
        if len(bands) == 2 and s.stereo == 'coherence':
            w = w * self._coherence(bands[0], bands[1]).astype(np.float32)

        def finish(pair):
            x, band = pair
            if s.method == 'adaptive':
                G = self._adaptive_gain(band, s, hop, depth)
                removed = band * (w[:, None] * (1.0 - G))
            else:
                removed = band * (w * np.float32(1.0 - depth))[:, None]
            return (x - self._istft_rows(removed, rows, nperseg, frames, n)).astype(np.float32)

        out = _pmap(finish, list(zip(chans, bands)))
        del bands

        if mid_only:
            m = out[0]
            out = [m + side, m - side]
        return np.column_stack(out).astype(np.float32)

    def apply(self, s: FilterSettings) -> np.ndarray | None:
        if not self.loaded:
            return None
        out = self._audio
        if s.tones_on and any(n.get('on', True) for n in s.notches):
            out = self._run_stage(out, s, lambda f: self.notch_weight(f, s.notches),
                                  s.tone_depth_db, self.nperseg_tones(s))
        if s.shelf_on:
            out = self._run_stage(out, s,
                                  lambda f: self.shelf_weight(f, s.fade_start, s.cutoff, s.curve),
                                  s.shelf_depth_db, self.nperseg_shelf(s))
        return out.copy() if out is self._audio else out

    # ── Result analysis ──────────────────────────────────────

    def analyse(self, processed: np.ndarray, s: FilterSettings) -> dict:
        orig     = self._audio
        residual = orig - processed
        e_orig   = float(np.sum(orig.astype(np.float64) ** 2))
        e_res    = float(np.sum(residual.astype(np.float64) ** 2))
        pct      = 100.0 * e_res / max(e_orig, 1e-20)

        proc_mono = processed.mean(axis=1)
        res_mono  = residual.mean(axis=1)
        (f, proc_db), (_, res_db), view_proc, view_res = _pmap(lambda job: job(), [
            lambda: self._welch(proc_mono), lambda: self._welch(res_mono),
            lambda: self._view_of(proc_mono), lambda: self._view_of(res_mono)])
        orig_rel = self._welch_db - self._psd_ref
        proc_rel = proc_db - self._psd_ref

        # Tones still standing out after cleaning — measured against the original
        # floor (lowered by the shelf where it applies), so notch dips can't fake a result.
        tones_before = self.count_tones(orig_rel, s.threshold_db, s.min_hz)
        tones_after  = self.count_tones(proc_rel - self.shelf_gain_db(f, s), s.threshold_db, s.min_hz)

        # Flatness of what was removed: ~0 = tones (good), ~1 = broadband music.
        flatness = None
        w_t = self.notch_weight(f, s.notches) if s.tones_on else np.zeros(len(f))
        w_s = self.shelf_weight(f, s.fade_start, s.cutoff, s.curve) if s.shelf_on else np.zeros(len(f))
        region = ((w_t > 0.05) | (w_s > 0.05)) & self._audible
        if np.count_nonzero(region) >= 4 and e_res > 0:
            p = 10.0 ** (res_db[region] / 10.0)
            flatness = float(np.exp(np.mean(np.log(p + 1e-30))) / (np.mean(p) + 1e-30))

        rms_orig = np.sqrt(e_orig / orig.size)
        rms_res  = np.sqrt(e_res / orig.size)
        peak_res = float(np.max(np.abs(residual))) if residual.size else 0.0
        boost = 1.0
        if rms_res > 1e-12:
            boost = float(np.clip(min(rms_orig / rms_res, 0.9 / max(peak_res, 1e-12)), 1.0, 100.0))

        return {
            'energy_pct':   pct,
            'flatness':     flatness,
            'tones_before': tones_before,
            'tones_after':  tones_after,
            'psd_f':        f,
            'psd_proc':     proc_rel,
            'prom_proc':    proc_rel - self.shelf_gain_db(f, s) - self._floor,
            'view_proc':    view_proc,
            'view_res':     view_res,
            'boost':        boost,
        }


# ──────────────────────────────────────────────────────────────
#  SPECTRUM CANVAS  (averaged spectrum, log frequency)
#
#  Blue = original, green = cleaned, red fill = what was removed.
#  LMB = fade start (orange)   RMB = cutoff (red)   — the high-band softening
#  Shift+LMB = add / drag a tone notch   Shift+RMB = remove a notch
#  MMB = pan   scroll = zoom   double-click = full view
# ──────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(C['ax_bg'])
    for sp in ax.spines.values():
        sp.set_edgecolor(C['grid'])
    ax.tick_params(colors=C['text'], labelsize=8)
    ax.xaxis.label.set_color(C['text'])
    ax.yaxis.label.set_color(C['text'])


class SpectrumCanvas(FigureCanvasQTAgg):
    shelf_changed  = pyqtSignal(float, float)   # (fade_start_hz, cutoff_hz)
    notches_edited = pyqtSignal(list)
    view_changed   = pyqtSignal(float, float)   # (lo_hz, hi_hz) after any zoom/pan

    MIN_FREQ      = 20.0
    MIN_SEP       = 200.0
    PICK_PX       = 7
    DEFAULT_WIDTH = 20.0

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=C['bg'], tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.ax = self.fig.add_subplot(111)
        _style_axes(self.ax)
        self.ax.set_xscale('log')
        self.ax.set_ylim(-100, 5)
        self.ax.set_xlabel('Frequency (Hz)', fontsize=9)
        self.ax.set_ylabel('Level (dB)', fontsize=9)
        self.ax.grid(True, color=C['grid'], alpha=0.5, which='both', lw=0.5)
        self.ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'))
        self.ax.set_title('Spectrum · LMB / RMB = move orange / red lines · Shift+click = add / remove tone · '
                          'scroll = zoom · dbl-click = full', color=C['text'], fontsize=9.5, pad=6)

        self.shelf_on        = True
        self.fade_start_freq = 14000.0
        self.cutoff_freq     = 18000.0
        self.notches: list   = []
        self.max_freq        = 22050.0
        self._zoom_xlim_full = (self.MIN_FREQ, self.max_freq)
        self.ax.set_xlim(*self._zoom_xlim_full)

        self._overlay   = []
        self._line_orig = None
        self._line_proc = None
        self._fill_res  = None
        self._freqs     = None
        self._orig_db   = None

        self._dragging  = None   # 'fade' | 'cut' | int (notch index)
        self._pan_start = None
        self._redraw_overlay()

        self.mpl_connect('button_press_event',   self._on_press)
        self.mpl_connect('motion_notify_event',  self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('scroll_event',         self._on_scroll)

    # ── Public API ────────────────────────────────────────────

    def set_nyquist(self, nyq: float):
        self.max_freq = float(nyq)
        self._zoom_xlim_full = (self.MIN_FREQ, self.max_freq)
        self.set_view(*self._zoom_xlim_full)

    def set_view(self, lo: float, hi: float):
        lo = max(lo, self._zoom_xlim_full[0])
        hi = min(hi, self._zoom_xlim_full[1])
        self.ax.set_xlim(lo, hi)
        self._auto_ylim()
        self.view_changed.emit(lo, hi)
        self.draw_idle()

    def set_shelf(self, on: bool, fade_start: float, cutoff: float):
        self.shelf_on        = bool(on)
        self.fade_start_freq = float(fade_start)
        self.cutoff_freq     = float(cutoff)
        self._redraw_overlay()

    def set_notches(self, notches: list):
        self.notches = [dict(n) for n in notches]
        self._redraw_overlay()

    def plot_original(self, freqs: np.ndarray, pdb: np.ndarray):
        m = freqs > 0
        self._freqs, self._orig_db = freqs[m], pdb[m]
        if self._line_orig is not None:
            self._line_orig.remove()
        self._line_orig, = self.ax.plot(self._freqs, self._orig_db, color=C['orig'],
                                        lw=1.0, alpha=0.9, zorder=3)
        self.clear_processed()
        self._auto_ylim()
        self._legend()
        self.draw_idle()

    def plot_processed(self, freqs: np.ndarray, proc_db: np.ndarray):
        self.clear_processed(redraw=False)
        m = freqs > 0
        f, p = freqs[m], proc_db[m]
        self._line_proc, = self.ax.plot(f, p, color=C['proc'], lw=1.0, alpha=0.95, zorder=4)
        if self._orig_db is not None and len(self._orig_db) == len(p):
            self._fill_res = self.ax.fill_between(
                f, p, self._orig_db, where=self._orig_db > p + 0.5, color=C['cut'],
                alpha=0.55, lw=0, zorder=2, interpolate=True)
        self._legend()
        self.draw_idle()

    def clear_processed(self, redraw: bool = True):
        for name in ('_line_proc', '_fill_res'):
            art = getattr(self, name)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
                setattr(self, name, None)
        if redraw:
            self._legend()
            self.draw_idle()

    def view_center(self) -> float:
        lo, hi = self.ax.get_xlim()
        return float(np.sqrt(max(lo, 1.0) * max(hi, 1.0)))

    # ── Drawing ───────────────────────────────────────────────

    def _redraw_overlay(self):
        for a in self._overlay:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay = []
        ax = self.ax
        if self.shelf_on:
            self._overlay += [
                ax.axvspan(self.fade_start_freq, self.cutoff_freq, alpha=0.12, color=C['fade'], zorder=1),
                ax.axvspan(self.cutoff_freq, max(self.max_freq, self.cutoff_freq), alpha=0.07,
                           color=C['cut'], zorder=1),
                ax.axvline(self.fade_start_freq, color=C['fade'], lw=2, ls='--', zorder=5),
                ax.axvline(self.cutoff_freq, color=C['cut'], lw=2, zorder=5),
            ]
        on = [float(n['freq']) for n in self.notches if n.get('on', True)]
        off = [float(n['freq']) for n in self.notches if not n.get('on', True)]
        tr = ax.get_xaxis_transform()
        if on:
            self._overlay.append(ax.vlines(on, 0.0, 0.05, transform=tr, color=C['cut'], lw=1.2, zorder=6))
        if off:
            self._overlay.append(ax.vlines(off, 0.0, 0.05, transform=tr, color=C['off'], lw=1.0, zorder=6))
        self.draw_idle()

    def _auto_ylim(self):
        if self._orig_db is None:
            return
        lo, hi = self.ax.get_xlim()
        m = (self._freqs >= lo) & (self._freqs <= hi)
        v = self._orig_db[m]
        v = v[np.isfinite(v)]
        if len(v) > 4:
            top = float(np.max(v))
            bottom = max(float(np.percentile(v, 2)) - 8, top - 110)
            self.ax.set_ylim(bottom, top + 5)

    def _legend(self):
        handles, labels = [], []
        for art, lb in [(self._line_orig, 'Original'), (self._line_proc, 'Cleaned'),
                        (self._fill_res, 'Removed')]:
            if art is not None:
                handles.append(art)
                labels.append(lb)
        if handles:
            self.ax.legend(handles, labels, facecolor=C['bg'], edgecolor=C['grid'],
                           labelcolor=C['text'], fontsize=8, loc='lower left')

    def _freq_at(self, xdata) -> float:
        return float(np.clip(xdata, self.MIN_FREQ, self.max_freq))

    def _notch_at_px(self, x_px: float) -> int | None:
        best, best_d = None, self.PICK_PX
        for i, n in enumerate(self.notches):
            px = self.ax.transData.transform((float(n['freq']), 0.0))[0]
            d = abs(px - x_px)
            if d <= best_d:
                best, best_d = i, d
        return best

    # ── Mouse events ─────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.dblclick:
            self.set_view(*self._zoom_xlim_full)
            return
        if event.button == 2:
            self._pan_start = (event.x, self.ax.get_xlim())
            return
        freq = self._freq_at(event.xdata)

        if event.key == 'shift':
            hit = self._notch_at_px(event.x)
            if event.button == 1:
                if hit is None:
                    self.notches.append({'freq': round(freq, 1), 'width': self.DEFAULT_WIDTH, 'on': True})
                    hit = len(self.notches) - 1
                self._dragging = hit
                self._redraw_overlay()
            elif event.button == 3 and hit is not None:
                del self.notches[hit]
                self._redraw_overlay()
                self.notches_edited.emit([dict(n) for n in self.notches])
            return

        if event.button == 1:
            self._dragging = 'fade'
            self.fade_start_freq = min(freq, self.cutoff_freq - self.MIN_SEP)
        elif event.button == 3:
            self._dragging = 'cut'
            self.cutoff_freq = max(freq, self.fade_start_freq + self.MIN_SEP)
        self.shelf_on = True
        self._redraw_overlay()

    def _on_motion(self, event):
        if self._pan_start is not None:
            x0_px, (lo0, hi0) = self._pan_start
            ax_bbox = self.ax.get_window_extent()
            if ax_bbox.width > 0 and lo0 > 0 and hi0 > 0:
                log_lo0, log_hi0 = np.log10(lo0), np.log10(hi0)
                log_span  = log_hi0 - log_lo0
                delta_log = (x0_px - event.x) / ax_bbox.width * log_span
                new_lo, new_hi = log_lo0 + delta_log, log_hi0 + delta_log
                full_lo = np.log10(self._zoom_xlim_full[0])
                full_hi = np.log10(self._zoom_xlim_full[1])
                if new_lo < full_lo:
                    new_lo, new_hi = full_lo, full_lo + log_span
                if new_hi > full_hi:
                    new_hi, new_lo = full_hi, full_hi - log_span
                self.ax.set_xlim(10 ** new_lo, 10 ** new_hi)
                self.view_changed.emit(10 ** new_lo, 10 ** new_hi)
                self.draw_idle()
            return

        if self._dragging is None or event.inaxes != self.ax or event.xdata is None:
            return
        freq = self._freq_at(event.xdata)
        if self._dragging == 'fade':
            self.fade_start_freq = min(freq, self.cutoff_freq - self.MIN_SEP)
        elif self._dragging == 'cut':
            self.cutoff_freq = max(freq, self.fade_start_freq + self.MIN_SEP)
        elif isinstance(self._dragging, int) and self._dragging < len(self.notches):
            self.notches[self._dragging]['freq'] = round(freq, 1)
        self._redraw_overlay()

    def _on_release(self, event):
        if event.button == 2:
            self._pan_start = None
            return
        if self._dragging is None:
            return
        was = self._dragging
        self._dragging = None
        if was in ('fade', 'cut'):
            self.shelf_changed.emit(self.fade_start_freq, self.cutoff_freq)
        else:
            self.notches_edited.emit([dict(n) for n in self.notches])

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.xdata <= 0:
            return
        lo, hi = self.ax.get_xlim()
        if lo <= 0:
            return
        factor = 0.7 if event.button == 'up' else (1.0 / 0.7)
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        log_cursor = np.log10(event.xdata)
        new_log_lo = max(log_cursor + (log_lo - log_cursor) * factor, np.log10(self._zoom_xlim_full[0]))
        new_log_hi = min(log_cursor + (log_hi - log_cursor) * factor, np.log10(self._zoom_xlim_full[1]))
        if new_log_hi - new_log_lo < 0.002:
            return
        self.ax.set_xlim(10 ** new_log_lo, 10 ** new_log_hi)
        self._auto_ylim()
        self.view_changed.emit(10 ** new_log_lo, 10 ** new_log_hi)
        self.draw_idle()


# ──────────────────────────────────────────────────────────────
#  TONE CLOSE-UP  (linear frequency, level above the local floor)
#
#  Blue spikes = AI tones in the original. Green = after cleaning.
#  Anything green still above the yellow line is a tone that is left.
#  RMB = add a notch   scroll = zoom   MMB drag = pan   double-click = reset
# ──────────────────────────────────────────────────────────────

class ToneCloseupCanvas(FigureCanvasQTAgg):
    add_notch_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=C['bg'], tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip('Right-click = add a tone notch there · scroll = zoom · '
                        'middle-drag = pan · double-click = reset')

        self.ax = self.fig.add_subplot(111)
        _style_axes(self.ax)
        self.ax.set_xlabel('Frequency (kHz)', fontsize=9)
        self.ax.set_ylabel('dB above surroundings', fontsize=9)
        self.ax.grid(True, color=C['grid'], alpha=0.5, lw=0.5)
        self.ax.set_title('Tone close-up · blue spikes = AI tones · green = after cleaning · '
                          'green above the yellow line = still there', color=C['text'], fontsize=9.5, pad=6)
        self._full = (4.0, 16.0)
        self._orig = self._proc = self._thr = None
        self._ticks = []
        self._pan = None
        self.mpl_connect('button_press_event',   self._on_press)
        self.mpl_connect('motion_notify_event',  self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('scroll_event',         self._on_scroll)

    def set_original(self, freqs: np.ndarray, prom: np.ndarray, lo_hz: float, hi_hz: float):
        for art in (self._orig, self._proc):
            if art is not None:
                art.remove()
        self._proc = None
        self._freqs = freqs / 1000.0
        self._orig, = self.ax.plot(self._freqs, prom, color=C['orig'], lw=0.8, zorder=3)
        self._full = (lo_hz / 1000.0, hi_hz / 1000.0)
        self.ax.set_xlim(*self._full)
        m = (freqs >= lo_hz) & (freqs <= hi_hz)
        top = float(np.percentile(prom[m], 99.9)) if np.any(m) else 20.0
        self.ax.set_ylim(-12.0, max(top + 4.0, 15.0))
        self.draw_idle()

    def set_processed(self, prom_proc: np.ndarray | None):
        if self._proc is not None:
            self._proc.remove()
            self._proc = None
        if prom_proc is not None:
            self._proc, = self.ax.plot(self._freqs, prom_proc, color=C['proc'], lw=0.8, zorder=4)
        self.draw_idle()

    def set_threshold(self, thr: float):
        if self._thr is not None:
            self._thr.remove()
        self._thr = self.ax.axhline(thr, color=C['res'], lw=1.2, ls='--', zorder=5)
        self.draw_idle()

    def set_notches(self, notches: list):
        for a in self._ticks:
            a.remove()
        self._ticks = []
        on = [float(n['freq']) / 1000.0 for n in notches if n.get('on', True)]
        if on:
            self._ticks.append(self.ax.vlines(on, 0.0, 0.04, transform=self.ax.get_xaxis_transform(),
                                              color=C['cut'], lw=1.0, zorder=6))
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.dblclick:
            self.ax.set_xlim(*self._full)
            self.draw_idle()
        elif event.button == 3:
            self.add_notch_requested.emit(float(event.xdata) * 1000.0)
        elif event.button == 2:
            self._pan = (event.x, self.ax.get_xlim())

    def _on_motion(self, event):
        if self._pan is None:
            return
        x0, (lo, hi) = self._pan
        width = self.ax.get_window_extent().width
        if width <= 0:
            return
        shift = (x0 - event.x) / width * (hi - lo)
        shift = float(np.clip(shift, self._full[0] - lo, self._full[1] - hi))
        self.ax.set_xlim(lo + shift, hi + shift)
        self.draw_idle()

    def _on_release(self, event):
        if event.button == 2:
            self._pan = None

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        lo, hi = self.ax.get_xlim()
        factor = 0.7 if event.button == 'up' else (1.0 / 0.7)
        x = float(event.xdata)
        new_lo = max(self._full[0], x + (lo - x) * factor)
        new_hi = min(self._full[1], x + (hi - x) * factor)
        if new_hi - new_lo < 0.02:
            return
        self.ax.set_xlim(new_lo, new_hi)
        self.draw_idle()


# ──────────────────────────────────────────────────────────────
#  TONE-LINES VIEW  (spectrogram of level above the local floor)
#
#  Every bright horizontal line is a tone that never goes away — the typical
#  AI artifact. In "Cleaned" the lines should be gone; "Removed" shows what
#  was taken out. Follows what you are listening to (A / B / Solo).
#
#  click = seek   LMB drag = noise-profile region (adaptive mode)
#  RMB = add a notch at that frequency   scroll = zoom   double-click = reset
# ──────────────────────────────────────────────────────────────

class ToneLinesCanvas(FigureCanvasQTAgg):
    region_selected     = pyqtSignal(float, float)
    add_notch_requested = pyqtSignal(float)
    seek_requested      = pyqtSignal(float)

    DRAG_PX = 5
    VMIN, VMAX = 3.0, 18.0

    _VIEW_TITLES = {
        'orig': ('Original — every bright horizontal line is a steady AI tone', C['orig']),
        'proc': ('Cleaned — the lines should be gone', C['proc']),
        'res':  ('Removed — what the cleaner takes out', C['res']),
    }

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=C['bg'], tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip('Click = seek · drag = pick noise-profile region (adaptive mode) · '
                        'right-click = add a tone notch there · scroll = zoom · double-click = reset')

        self.ax = self.fig.add_subplot(111)
        _style_axes(self.ax)
        self.ax.set_xlabel('Time', fontsize=9)
        self.ax.set_ylabel('kHz', fontsize=9)
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v)))

        self._images: dict = {}
        self._img    = None
        self._view   = 'orig'
        self._extent = (0.0, 1.0, 2.0, 22.05)
        self._default_ylim = (2.0, 22.05)
        self._overlay: list = []
        self._region_patch  = None
        self._drag_patch    = None
        self._press = None

        self._playhead = self.ax.axvline(0.0, color='#ffffff', lw=1.0, alpha=0.8, animated=True)
        self._bg = None
        self._set_title()

        self.mpl_connect('draw_event',           self._on_draw)
        self.mpl_connect('button_press_event',   self._on_press)
        self.mpl_connect('motion_notify_event',  self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('scroll_event',         self._on_scroll)

    # ── Public API ────────────────────────────────────────────

    def set_original(self, view: dict, focus_lo_hz: float, focus_hi_hz: float):
        self._images = {'orig': view['orig']}
        self._extent = view['extent']
        if self._img is not None:
            self._img.remove()
        self._img = self.ax.imshow(
            view['orig'], origin='lower', aspect='auto', extent=self._extent, cmap='magma',
            vmin=self.VMIN, vmax=self.VMAX, interpolation='antialiased', zorder=0)
        lo = max(self._extent[2], focus_lo_hz / 1000.0)
        hi = min(self._extent[3], focus_hi_hz / 1000.0)
        self._default_ylim = (lo, hi) if hi > lo else (self._extent[2], self._extent[3])
        self.ax.set_xlim(self._extent[0], self._extent[1])
        self.ax.set_ylim(*self._default_ylim)
        self.set_region(None)
        self.set_view(self._view)

    def set_processed(self, proc: np.ndarray, res: np.ndarray):
        self._images['proc'] = proc
        self._images['res']  = res
        self.set_view(self._view)

    def clear_processed(self):
        self._images.pop('proc', None)
        self._images.pop('res', None)
        self.set_view(self._view)

    def set_view(self, which: str):
        self._view = which
        if self._img is not None:
            data = self._images.get(which, self._images.get('orig'))
            if data is not None:
                self._img.set_data(data)
        self._set_title()
        self.draw_idle()

    def set_overlay(self, notches: list, shelf_on: bool, fade_start: float, cutoff: float):
        """Markers at the edges only, so they never hide the lines themselves."""
        for a in self._overlay:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay = []
        ax = self.ax
        ys = [float(n['freq']) / 1000.0 for n in notches if n.get('on', True)]
        if ys:
            tr = ax.get_yaxis_transform()
            self._overlay.append(ax.hlines(ys, 0.0, 0.012, transform=tr, color=C['cut'], lw=0.8, alpha=0.7, zorder=4))
            self._overlay.append(ax.hlines(ys, 0.988, 1.0, transform=tr, color=C['cut'], lw=0.8, alpha=0.7, zorder=4))
        if shelf_on:
            self._overlay.append(ax.axhline(fade_start / 1000.0, color=C['fade'], lw=1.0, ls='--', zorder=4))
        self.draw_idle()

    def set_region(self, region: tuple | None):
        if self._region_patch is not None:
            try:
                self._region_patch.remove()
            except Exception:
                pass
            self._region_patch = None
        if region:
            t0, t1 = sorted(region)
            self._region_patch = self.ax.axvspan(t0, t1, facecolor=C['proc'], alpha=0.18,
                                                 edgecolor=C['proc'], lw=1.2, zorder=3)
        self.draw_idle()

    def set_playhead(self, t: float):
        self._playhead.set_xdata([t, t])
        if self._bg is None:
            return
        self.restore_region(self._bg)
        self.ax.draw_artist(self._playhead)
        self.blit(self.fig.bbox)

    # ── Internal ──────────────────────────────────────────────

    def _set_title(self):
        name, col = self._VIEW_TITLES.get(self._view, self._VIEW_TITLES['orig'])
        self.ax.set_title(f'Tone lines  ·  {name}', color=col, fontsize=10.5, pad=6)

    def _on_draw(self, _event):
        self._bg = self.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self._playhead)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or self._img is None:
            return
        if event.dblclick:
            self.ax.set_ylim(*self._default_ylim)
            self.draw_idle()
            return
        if event.button == 1:
            self._press = (event.x, float(event.xdata))
        elif event.button == 3 and event.ydata is not None:
            self.add_notch_requested.emit(float(event.ydata) * 1000.0)

    def _on_motion(self, event):
        if self._press is None or event.xdata is None:
            return
        x0_px, t0 = self._press
        if abs(event.x - x0_px) < self.DRAG_PX:
            return
        if self._drag_patch is not None:
            self._drag_patch.remove()
        lo, hi = sorted((t0, float(event.xdata)))
        self._drag_patch = self.ax.axvspan(lo, hi, facecolor=C['proc'], alpha=0.12, zorder=3)
        self.draw_idle()

    def _on_release(self, event):
        if self._press is None or event.button != 1:
            return
        x0_px, t0 = self._press
        self._press = None
        if self._drag_patch is not None:
            self._drag_patch.remove()
            self._drag_patch = None
            self.draw_idle()
        t1 = float(event.xdata) if event.xdata is not None else t0
        t_lo, t_hi = max(self._extent[0], 0.0), self._extent[1]
        if abs(event.x - x0_px) < self.DRAG_PX:
            self.seek_requested.emit(float(np.clip(t0, t_lo, t_hi)))
        else:
            a, b = sorted((float(np.clip(t0, t_lo, t_hi)), float(np.clip(t1, t_lo, t_hi))))
            if b - a > 0.05:
                self.region_selected.emit(a, b)

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.ydata is None or self._img is None:
            return
        lo, hi = self.ax.get_ylim()
        factor = 0.7 if event.button == 'up' else (1.0 / 0.7)
        y = float(event.ydata)
        new_lo = max(self._extent[2], y + (lo - y) * factor)
        new_hi = min(self._extent[3], y + (hi - y) * factor)
        if new_hi - new_lo < 0.05:
            return
        self.ax.set_ylim(new_lo, new_hi)
        self.draw_idle()


# ──────────────────────────────────────────────────────────────
#  BACKGROUND WORKERS
# ──────────────────────────────────────────────────────────────

class LoadWorker(QThread):
    done  = pyqtSignal(object)
    error = pyqtSignal(str, int)

    def __init__(self, path: str, generation: int):
        super().__init__()
        self.path = path
        self.generation = generation

    def run(self):
        try:
            audio, sr = read_audio(self.path)
            proc = FilterProcessor()
            proc.load(audio, sr)
            f, p = proc.spectrum_original()
            self.done.emit({'sr': sr, 'path': self.path, 'processor': proc,
                            'spec_f': f, 'spec_p': p, 'generation': self.generation})
        except Exception as e:
            self.error.emit(str(e), self.generation)


class RenderWorker(QThread):
    """Renders a stored version (history entry) so it can be played as A."""
    done  = pyqtSignal(object)
    error = pyqtSignal(str, int)

    def __init__(self, processor: FilterProcessor, settings: FilterSettings, generation: int, tag):
        super().__init__()
        self.processor  = processor
        self.settings   = settings
        self.generation = generation
        self.tag        = tag

    def run(self):
        try:
            _, audio = self.processor.clean(self.settings, detect=False)
            self.done.emit({'audio': audio, 'generation': self.generation, 'tag': self.tag})
        except Exception as e:
            self.error.emit(str(e), self.generation)


class CleanWorker(QThread):
    """Detect (optional) → clean → measure, reporting each step."""
    progress = pyqtSignal(str, int)
    done     = pyqtSignal(object)
    error    = pyqtSignal(str, int)

    def __init__(self, processor: FilterProcessor, settings: FilterSettings,
                 detect: bool, generation: int):
        super().__init__()
        self.processor  = processor
        self.settings   = settings
        self.detect     = detect
        self.generation = generation

    def run(self):
        try:
            say = lambda msg: self.progress.emit(msg, self.generation)
            GPU.last_error = ''
            t0 = time.perf_counter()
            notches, audio = self.processor.clean(self.settings, self.detect, say)
            say('Measuring the result …')
            s = self.settings.copy()
            s.notches = notches
            result = self.processor.analyse(audio, s)
            result.update(seconds=time.perf_counter() - t0,
                          device='CPU' if GPU.torch() is None or GPU.last_error else 'GPU',
                          audio=audio, notches=notches, detected=self.detect and s.tones_on,
                          generation=self.generation, settings=s)
            self.done.emit(result)
        except Exception as e:
            self.error.emit(str(e), self.generation)


class BatchWorker(QThread):
    """Cleans a list of files with one set of settings and saves each result.

    detect_per_file: each track finds its own AI tones with those settings;
    otherwise every track uses settings.notches. manual_notches (path → list)
    override both for tracks whose tone list was edited by hand."""
    progress     = pyqtSignal(str)
    finished_msg = pyqtSignal(str)

    def __init__(self, paths: list, settings: FilterSettings, manual_notches: dict,
                 detect_per_file: bool, out_dir: Path, suffix: str, diff_suffix: str, fmt_key: str,
                 save_removed: bool = True):
        super().__init__()
        self.paths           = paths
        self.settings        = settings
        self.manual_notches  = manual_notches
        self.detect_per_file = detect_per_file
        self.out_dir         = out_dir
        self.suffix          = suffix
        self.diff_suffix     = diff_suffix
        self.fmt_key         = fmt_key
        self.save_removed    = save_removed
        self.cancelled       = False

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lines, errors = [], []
        total = len(self.paths)
        done = 0
        for i, path in enumerate(self.paths):
            if self.cancelled:
                break
            name = Path(path).name
            say = lambda msg: self.progress.emit(f'Batch {i + 1}/{total} · {name}: {msg}')
            try:
                say('loading …')
                audio, sr = read_audio(path)
                proc = FilterProcessor()
                proc.load(audio, sr)
                s = self.settings.copy()
                s.profile_region = None
                detect = self.detect_per_file
                if path in self.manual_notches:
                    s.notches = [dict(n) for n in self.manual_notches[path]]
                    detect = False
                notches, result = proc.clean(s, detect, say)
                say('saving …')
                out, fmt, sub = output_path(path, self.out_dir, self.suffix, self.fmt_key)
                write_audio(out, result, sr, fmt, sub)
                if self.save_removed:
                    diff_out, fmt, sub = output_path(path, self.out_dir, self.diff_suffix, self.fmt_key)
                    write_audio(diff_out, proc.audio - result, sr, fmt, sub)
                n_on = sum(1 for n in notches if n.get('on', True)) if s.tones_on else 0
                lines.append(f'{name} → {out.name}: {n_on} tones removed')
                done += 1
            except Exception as e:
                errors.append(f'{name}: {e}')
        msg = f'{done} of {total} files saved to:\n{self.out_dir.resolve()}'
        if self.cancelled:
            msg = 'Cancelled.  ' + msg
        if lines:
            msg += '\n\n' + '\n'.join(lines)
        if errors:
            msg += '\n\nErrors:\n' + '\n'.join(errors)
        self.finished_msg.emit(msg)


# ──────────────────────────────────────────────────────────────
#  BATCH DIALOG  (clean every track in the list)
# ──────────────────────────────────────────────────────────────

class BatchDialog(QDialog):
    """Pick the tracks, which settings to use, how tones are found, and where to save."""

    def __init__(self, parent, tracks: list, current_label: str, user_presets: list,
                 n_current_tones: int, n_manual: int, out_dir: Path, fmt_key: str,
                 suffix: str, prefs: dict):
        super().__init__(parent)
        self.setWindowTitle('Batch — clean all tracks')
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # ── Tracks ──
        box_t = Card('Tracks', collapsible=False)
        bt = QVBoxLayout(box_t.body)
        self.list_tracks = QListWidget()
        self.list_tracks.setMinimumHeight(140)
        for path, name in tracks:
            it = QListWidgetItem(name)
            it.setData(Qt.ItemDataRole.UserRole, path)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.list_tracks.addItem(it)
        bt.addWidget(self.list_tracks)
        tr = QHBoxLayout()
        b_all, b_none = QPushButton('All'), QPushButton('None')
        for b in (b_all, b_none):
            b.setObjectName('btn_small')
            tr.addWidget(b)
        b_all.clicked.connect(lambda: self._check_all(True))
        b_none.clicked.connect(lambda: self._check_all(False))
        tr.addStretch()
        self.lbl_count = _hint('', wrap=False)
        tr.addWidget(self.lbl_count)
        bt.addLayout(tr)
        self.list_tracks.itemChanged.connect(self._update_count)
        lay.addWidget(box_t)

        # ── Settings ──
        box_s = Card('Settings to use', collapsible=False)
        bs = QGridLayout(box_s.body)
        self.src_group = QButtonGroup(self)
        self.rb_current = QRadioButton('Current settings')
        bs.addWidget(self.rb_current, 0, 0)
        bs.addWidget(_hint(current_label), 0, 1)
        self.rb_strength = QRadioButton('Cleaning strength')
        bs.addWidget(self.rb_strength, 1, 0)
        self.combo_strength = QComboBox()
        for key, p in PRESETS.items():
            self.combo_strength.addItem(p['label'], key)
        bs.addWidget(self.combo_strength, 1, 1)
        self.rb_user = QRadioButton('Saved preset')
        bs.addWidget(self.rb_user, 2, 0)
        self.combo_user = QComboBox()
        for name in user_presets:
            self.combo_user.addItem(name, name)
        if not user_presets:
            self.combo_user.addItem('No saved presets yet', None)
            self.rb_user.setEnabled(False)
            self.combo_user.setEnabled(False)
        bs.addWidget(self.combo_user, 2, 1)
        for i, rb in enumerate((self.rb_current, self.rb_strength, self.rb_user)):
            self.src_group.addButton(rb, i)
        bs.addWidget(_hint('Current = exactly what the panel shows now, including your own tweaks. '
                           'A strength or preset replaces the tone / top-band knobs; the Advanced '
                           'settings stay as they are.'), 3, 0, 1, 2)
        bs.setColumnStretch(1, 1)
        lay.addWidget(box_s)

        # ── AI tones ──
        box_n = Card('AI tones', collapsible=False)
        bn = QVBoxLayout(box_n.body)
        self.rb_detect = QRadioButton('Find the tones in each track separately (recommended)')
        self.rb_list = QRadioButton(f'Use the current tone list for every track ({n_current_tones} tones)')
        self.tone_group = QButtonGroup(self)
        self.tone_group.addButton(self.rb_detect, 0)
        self.tone_group.addButton(self.rb_list, 1)
        bn.addWidget(self.rb_detect)
        bn.addWidget(self.rb_list)
        self.chk_manual = QCheckBox(f'Tracks whose tone list you edited by hand keep that list '
                                    f'({n_manual} track{"s" if n_manual != 1 else ""})')
        self.chk_manual.setEnabled(n_manual > 0)
        bn.addWidget(self.chk_manual)
        lay.addWidget(box_n)

        # ── Output ──
        box_o = Card('Output', collapsible=False)
        bo = QFormLayout(box_o.body)
        dr = QHBoxLayout()
        self.edit_dir = QLineEdit(str(out_dir))
        dr.addWidget(self.edit_dir, stretch=1)
        b_browse = QPushButton('Browse…')
        b_browse.setObjectName('btn_small')
        b_browse.clicked.connect(self._browse)
        dr.addWidget(b_browse)
        bo.addRow('Folder:', dr)
        self.combo_format = QComboBox()
        for key, (label, *_rest) in EXPORT_FORMATS.items():
            self.combo_format.addItem(label, key)
        self.combo_format.setCurrentIndex(max(self.combo_format.findData(fmt_key), 0))
        bo.addRow('Format:', self.combo_format)
        self.edit_suffix = QLineEdit(suffix)
        self.edit_suffix.setPlaceholderText('_cleaned')
        bo.addRow('Name suffix:', self.edit_suffix)
        self.chk_removed = QCheckBox('Also save the removed part of each track (…_removed)')
        bo.addRow(self.chk_removed)
        lay.addWidget(box_o)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_start = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_start.setText('Start')
        self.btn_start.setObjectName('btn_primary')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        for card in (box_t, box_s, box_n, box_o):
            card.body.layout().setContentsMargins(0, 0, 0, 0)

        # Remembered choices
        src = prefs.get('source', 0)
        btn = self.src_group.button(src)
        (btn if btn is not None and btn.isEnabled() else self.rb_current).setChecked(True)
        self.combo_strength.setCurrentIndex(max(self.combo_strength.findData(prefs.get('strength')), 0))
        self.combo_user.setCurrentIndex(max(self.combo_user.findData(prefs.get('user')), 0))
        (self.rb_detect if prefs.get('detect', True) else self.rb_list).setChecked(True)
        self.chk_manual.setChecked(prefs.get('keep_manual', True) and n_manual > 0)
        self.chk_removed.setChecked(prefs.get('save_removed', False))
        self._update_count()

    def _check_all(self, on: bool):
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for i in range(self.list_tracks.count()):
            self.list_tracks.item(i).setCheckState(state)

    def _update_count(self, *_):
        n = len(self.paths())
        self.lbl_count.setText(f'{n} of {self.list_tracks.count()} selected')
        self.btn_start.setEnabled(n > 0)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Save batch results to', self.edit_dir.text())
        if d:
            self.edit_dir.setText(d)

    def paths(self) -> list:
        return [self.list_tracks.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list_tracks.count())
                if self.list_tracks.item(i).checkState() == Qt.CheckState.Checked]

    def source(self) -> tuple:
        """('current', None) | ('strength', key) | ('user', name)"""
        i = self.src_group.checkedId()
        if i == 1:
            return 'strength', self.combo_strength.currentData()
        if i == 2 and self.combo_user.currentData() is not None:
            return 'user', self.combo_user.currentData()
        return 'current', None

    def prefs(self) -> dict:
        return {'source': max(self.src_group.checkedId(), 0),
                'strength': self.combo_strength.currentData(),
                'user': self.combo_user.currentData(),
                'detect': self.rb_detect.isChecked(),
                'keep_manual': self.chk_manual.isChecked(),
                'save_removed': self.chk_removed.isChecked()}

    def detect(self) -> bool:
        return self.rb_detect.isChecked()

    def keep_manual(self) -> bool:
        return self.chk_manual.isEnabled() and self.chk_manual.isChecked()

    def out_dir(self) -> Path:
        return Path(self.edit_dir.text().strip() or '.')

    def fmt_key(self) -> str:
        return self.combo_format.currentData()

    def suffix(self) -> str:
        return self.edit_suffix.text().strip()

    def save_removed(self) -> bool:
        return self.chk_removed.isChecked()


# ──────────────────────────────────────────────────────────────
#  SETTINGS DIALOG
# ──────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):

    def __init__(self, parent, suffix: str, fmt_key: str, detect_on_load: bool, use_gpu: bool):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumWidth(440)

        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.edit_suffix = QLineEdit(suffix)
        self.edit_suffix.setPlaceholderText('_cleaned')
        self.edit_suffix.setToolTip('Appended to the file name on Save / Save All. Leave empty for none.')
        form.addRow('Name suffix:', self.edit_suffix)

        self.combo_format = QComboBox()
        for key, (label, *_rest) in EXPORT_FORMATS.items():
            self.combo_format.addItem(label, key)
        self.combo_format.setCurrentIndex(max(self.combo_format.findData(fmt_key), 0))
        self.combo_format.setToolTip(
            'Lossless output avoids a second lossy generation (MP3 also low-passes near 16 kHz).')
        form.addRow('Output format:', self.combo_format)

        self.chk_detect = QCheckBox('Auto-detect AI tones for each track that is opened')
        self.chk_detect.setToolTip('Off: every track uses the last tone list you had. '
                                   'Press "Auto-detect tones" to find them for the current track.')
        self.chk_detect.setChecked(detect_on_load)
        form.addRow(self.chk_detect)

        gpu = GPU.device_name()
        self.chk_gpu = QCheckBox(f'Process on the GPU ({gpu})' if gpu else
                                 'Process on the GPU (needs PyTorch with CUDA)')
        self.chk_gpu.setChecked(use_gpu and bool(gpu))
        self.chk_gpu.setEnabled(bool(gpu))
        self.chk_gpu.setToolTip('Runs the FFT work through PyTorch + CUDA — several times faster. '
                                'Off: CPU (scipy).')
        form.addRow(self.chk_gpu)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def suffix(self) -> str:
        return self.edit_suffix.text().strip()

    def fmt_key(self) -> str:
        return self.combo_format.currentData()

    def detect_on_load(self) -> bool:
        return self.chk_detect.isChecked()

    def use_gpu(self) -> bool | None:
        """None when the GPU is unavailable (keep the stored preference)."""
        return self.chk_gpu.isChecked() if self.chk_gpu.isEnabled() else None


# ──────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────────────────────

def _hint(text: str, wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName('lbl_hint')
    lbl.setWordWrap(wrap)
    return lbl


def _transport_icon(kind: str, color: str, size: int = 12) -> QIcon:
    """Play / pause / stop drawn at one size — font glyphs for these vary wildly."""
    pm = QPixmap(size * 2, size * 2)   # 2x for high-DPI
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    s = size * 2
    if kind == 'play':
        p.drawPolygon(QPolygonF([QPointF(s * 0.18, s * 0.08), QPointF(s * 0.92, s * 0.5),
                                 QPointF(s * 0.18, s * 0.92)]))
    elif kind == 'pause':
        p.drawRect(QRectF(s * 0.16, s * 0.1, s * 0.24, s * 0.8))
        p.drawRect(QRectF(s * 0.60, s * 0.1, s * 0.24, s * 0.8))
    elif kind == 'right':
        p.drawPolygon(QPolygonF([QPointF(s * 0.32, s * 0.2), QPointF(s * 0.74, s * 0.5),
                                 QPointF(s * 0.32, s * 0.8)]))
    elif kind == 'down':
        p.drawPolygon(QPolygonF([QPointF(s * 0.2, s * 0.32), QPointF(s * 0.8, s * 0.32),
                                 QPointF(s * 0.5, s * 0.74)]))
    else:   # stop
        p.drawRect(QRectF(s * 0.14, s * 0.14, s * 0.72, s * 0.72))
    p.end()
    pm.setDevicePixelRatio(2.0)
    return QIcon(pm)


class Card(QFrame):
    """A panel section: header (collapse arrow, optional on/off box, title) inside
    the card, body below. Unchecked → body disabled, like a checkable QGroupBox."""
    toggled = pyqtSignal(bool)
    collapsed_changed = pyqtSignal(bool)

    def __init__(self, title: str, parent=None, collapsible: bool = True):
        super().__init__(parent)
        self.setObjectName('card')
        self._icon_open   = _transport_icon('down', C['orig'], 10)
        self._icon_closed = _transport_icon('right', C['orig'], 10)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 8)
        outer.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._checkable = False
        self.chk = QCheckBox('On')
        self.chk.hide()
        self.chk.toggled.connect(self._on_checked)
        self.btn_head = QPushButton(title)
        self.btn_head.setObjectName('btn_card_head')
        if collapsible:
            self.btn_head.setIcon(self._icon_open)
            self.btn_head.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_head.clicked.connect(lambda: self.set_collapsed(not self.collapsed()))
        else:
            self.btn_head.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_head.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        head.addWidget(self.btn_head)
        head.addStretch()
        head.addWidget(self.chk)
        outer.addLayout(head)

        self.body = QWidget()
        outer.addWidget(self.body)

    # QGroupBox-like API
    def setCheckable(self, on: bool):
        self._checkable = bool(on)
        self.chk.setVisible(on)
        if on:
            self.chk.setChecked(True)

    def setChecked(self, on: bool):
        self.chk.setChecked(bool(on))

    def isChecked(self) -> bool:
        return self.chk.isChecked() if self._checkable else True

    def setToolTip(self, text: str):
        super().setToolTip(text)
        self.btn_head.setToolTip(text)

    def _on_checked(self, on: bool):
        self.body.setEnabled(on)
        self.toggled.emit(on)

    def collapsed(self) -> bool:
        return self.body.isHidden()

    def set_collapsed(self, on: bool):
        self.body.setVisible(not on)
        self.btn_head.setIcon(self._icon_closed if on else self._icon_open)
        self.collapsed_changed.emit(on)


class WheelTrapList(QListWidget):
    """Scrolling past either end does not scroll the panel around it."""
    def wheelEvent(self, event):
        super().wheelEvent(event)
        event.accept()


class WheelTrapTable(QTableWidget):
    def wheelEvent(self, event):
        super().wheelEvent(event)
        event.accept()


class MainWindow(QMainWindow):
    gpu_ready = pyqtSignal()

    # Built exe: data folders sit next to the exe. From source: the project root.
    BASE_DIR     = (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
                    else Path(__file__).resolve().parent.parent)
    INPUT_DIR    = BASE_DIR / 'input'
    OUTPUT_DIR   = BASE_DIR / 'output'
    PRESET_DIR   = BASE_DIR / 'presets'
    HISTORY_FILE = BASE_DIR / 'history.json'
    HISTORY_MAX  = 200

    DEFAULT_SUFFIX = '_cleaned'
    DIFF_SUFFIX    = '_removed'

    _FIELD_LABELS = {
        'tones_on': 'Remove tones', 'threshold_db': 'Threshold', 'min_hz': 'Search above',
        'second_pass': 'Second pass', 'tone_depth_db': 'Tone reduction',
        'shelf_on': 'Top band', 'fade_start': 'Fade from', 'cutoff': 'Full from', 'curve': 'Curve',
        'shelf_depth_db': 'Top-band reduction', 'method': 'Method', 'oversub': 'Oversub',
        'percentile': 'Percentile', 'smoothing': 'Smoothing', 'attack_ms': 'Attack',
        'release_ms': 'Release', 'stereo': 'Stereo', 'resolution': 'FFT size',
        'profile_region': 'Noise profile',
    }
    _FIELD_UNITS = {
        'threshold_db': ' dB', 'tone_depth_db': ' dB', 'shelf_depth_db': ' dB', 'min_hz': ' Hz',
        'fade_start': ' Hz', 'cutoff': ' Hz', 'attack_ms': ' ms', 'release_ms': ' ms',
        'percentile': ' %', 'oversub': '×',
    }

    _AB_LABELS = [
        ('A  Original', 'btn_ab_a'),
        ('B  Cleaned',  'btn_ab_b'),
    ]
    _MODE_COLORS = {'orig': C['orig'], 'proc': C['proc'], 'res': C['res']}

    _CURVES  = [('Cosine (smooth)', 'cosine'), ('Linear', 'linear'), ('Steep', 'steep')]
    _METHODS = [('Static — fixed reduction', 'static'),
                ('Adaptive — spectral subtraction', 'adaptive')]
    _STEREO  = [('L / R independently', 'lr'),
                ('Mid only (keep Side untouched)', 'mid'),
                ('L / R, scaled by L-R coherence', 'coherence')]
    _RESOLUTIONS = [('Auto', 0), ('4096', 4096), ('8192', 8192), ('16384', 16384), ('32768', 32768)]

    _STEPS = ['Pick a track on the left', 'Cleaning runs automatically',
              'Compare: Tab = A/B, R = hear only what is removed', 'Save']

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FrequencyCleaner — remove the AI sound')
        self.setMinimumSize(1280, 820)

        # ── Audio data ───────────────────────────────────────
        self.processor    = FilterProcessor()
        self.current_path = ''
        self.sample_rate  = 44100

        # ── Settings ─────────────────────────────────────────
        self._settings = QSettings('FrequencyCleaner', 'FrequencyCleaner')
        if not self._settings.allKeys():
            old = QSettings('HighFrequencyCleaner', 'HighFrequencyCleaner')
            for k in old.allKeys():
                self._settings.setValue(k, old.value(k))
        self.name_suffix    = self._settings.value('name_suffix', self.DEFAULT_SUFFIX, type=str)
        self.export_format  = self._settings.value('export_format', 'wav24', type=str)
        if self.export_format not in EXPORT_FORMATS:
            self.export_format = 'wav24'
        self.detect_on_load = self._settings.value('detect_on_load_v2', False, type=bool)
        self.batch_detect   = self._settings.value('batch_detect_v2', False, type=bool)
        GPU.enabled = self._settings.value('use_gpu', True, type=bool)
        self.fs = FilterSettings.from_json(self._settings.value('filter_settings_v2', '', type=str))
        last_notches = self._settings.value('last_notches', '', type=str)
        self._have_last_notches = bool(last_notches)   # False only before the first result ever
        if last_notches:
            self.fs.notches = FilterSettings.from_dict({'notches': read_json_text(last_notches, [])}).notches
        self._manual_notches: dict = {}   # path → hand-edited tone list
        self._detect_cache:   dict = {}   # (path, detect settings) → detected tone list

        # ── History / user presets ───────────────────────────
        self._history: list = self._load_history()
        self._hist_current: int | None = None   # id of the entry the shown result belongs to
        self._hist_next_id = max((e['id'] for e in self._history), default=0) + 1
        self._user_preset_map: dict = {}         # name → (file path, settings dict)
        self._ref_id: int | None = None          # history entry playing as A
        self._ref_worker: RenderWorker | None = None
        self._ref_generation = 0

        # ── Callback-shared state  (CPython GIL → atomic reads/writes) ──
        self._audio_orig: np.ndarray | None = None
        self._audio_proc: np.ndarray | None = None
        self._cb_frame:   int   = 0
        self._cb_playing: bool  = False
        self._ab_mode:    int   = 1        # 0=original  1=cleaned (start on B: hear the result)
        self._solo_residual: bool = False
        self._cb_volume:  float = 0.85
        self._residual_boost: float = 1.0
        self._audio_ref:  np.ndarray | None = None   # history version played as A

        # ── Playback ─────────────────────────────────────────
        self._stream: sd.OutputStream | None = None
        self.is_playing = False

        # ── Workers ──────────────────────────────────────────
        self._load_worker:  LoadWorker  | None = None
        self._clean_worker: CleanWorker | None = None
        self._batch_worker: BatchWorker | None = None
        self._old_workers: list = []
        self._proc_generation = 0
        self._load_generation = 0
        self._pending_detect  = False
        self._updating  = False    # True while controls are set from code
        self._table_updating = False

        # ── Build ────────────────────────────────────────────
        self._build_menu()
        self._build_ui()
        self.setStyleSheet(DARK_QSS)
        self._settings_to_controls()
        self._connect_signals()
        self._refresh_history_list()
        self._load_input_folder()
        self.setAcceptDrops(True)

        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(40)
        self._pos_timer.timeout.connect(self._poll_position)

        self._clean_debounce = QTimer(self)
        self._clean_debounce.setSingleShot(True)
        self._clean_debounce.timeout.connect(self._run_clean)

        QShortcut(QKeySequence('Space'), self).activated.connect(self._toggle_play)
        QShortcut(QKeySequence('Tab'),   self).activated.connect(self._toggle_ab)
        QShortcut(QKeySequence('r'),     self).activated.connect(self._toggle_solo_residual)
        QShortcut(QKeySequence('Left'),  self).activated.connect(lambda: self._seek_rel(-5))
        QShortcut(QKeySequence('Right'), self).activated.connect(lambda: self._seek_rel(5))
        QShortcut(QKeySequence('Ctrl+Z'), self).activated.connect(lambda: self._history_step(-1))
        QShortcut(QKeySequence('Ctrl+Y'), self).activated.connect(lambda: self._history_step(1))
        QShortcut(QKeySequence('Ctrl+Shift+Z'), self).activated.connect(lambda: self._history_step(1))

        self._update_steps(1)

        self.gpu_ready.connect(self._update_device_label)
        threading.Thread(target=lambda: (GPU.warm_up(), self.gpu_ready.emit()), daemon=True).start()

    def _update_device_label(self):
        text = GPU.label()
        on_gpu = text.startswith('GPU')
        if on_gpu and GPU.last_error:
            text += '  ·  last run fell back to CPU'
        col = C['proc'] if on_gpu and not GPU.last_error else C['res']
        self.lbl_device.setText(text)
        self.lbl_device.setStyleSheet(f'color: {col}; padding: 0 8px;')
        self.lbl_device.setToolTip(f'GPU error: {GPU.last_error}' if GPU.last_error else
                                   'Where the FFT work runs. Change it in Settings.')
        self.lbl_device_adv.setText(text)

    # ── UI Construction ───────────────────────────────────────

    def _build_menu(self):
        menu = self.menuBar().addMenu('&Settings')
        menu.addAction('Settings…').triggered.connect(self._open_settings)

    def _open_settings(self):
        dlg = SettingsDialog(self, self.name_suffix, self.export_format,
                             self.detect_on_load, GPU.enabled)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.use_gpu() is not None:
                GPU.enabled = dlg.use_gpu()
                GPU.last_error = ''
                self._settings.setValue('use_gpu', GPU.enabled)
                self._update_device_label()
            self.name_suffix    = dlg.suffix()
            self.export_format  = dlg.fmt_key()
            self.detect_on_load = dlg.detect_on_load()
            self._settings.setValue('name_suffix', self.name_suffix)
            self._settings.setValue('export_format', self.export_format)
            self._settings.setValue('detect_on_load_v2', self.detect_on_load)
            self._update_export_label()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QHBoxLayout(root)
        rl.setContentsMargins(10, 10, 10, 6)
        rl.setSpacing(10)

        rl.addWidget(self._build_tracks_panel())

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        self.lbl_steps = QLabel()
        self.lbl_steps.setObjectName('lbl_steps')
        self.lbl_steps.setTextFormat(Qt.TextFormat.RichText)
        cl.addWidget(self.lbl_steps)

        self.view_frame = QFrame()
        self.view_frame.setObjectName('view_frame')
        vf = QVBoxLayout(self.view_frame)
        vf.setContentsMargins(3, 3, 3, 3)
        vf.setSpacing(2)

        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(6, 2, 6, 0)
        zoom_row.addWidget(_hint('Spectrum view:', wrap=False))
        self.btn_zoom_full = QPushButton('Full range')
        self.btn_zoom_band = QPushButton('AI tone band')
        for b in (self.btn_zoom_full, self.btn_zoom_band):
            b.setObjectName('btn_small')
            zoom_row.addWidget(b)
        zoom_row.addStretch()
        legend = _hint('Blue = original · Green = cleaned · Red = removed · '
                       'small red ticks = tones found', wrap=False)
        legend.setStyleSheet('font-size: 12px;')
        zoom_row.addWidget(legend)
        vf.addLayout(zoom_row)

        spec_pane = QWidget()
        sp = QVBoxLayout(spec_pane)
        sp.setContentsMargins(0, 0, 0, 0)
        sp.setSpacing(2)
        self.spectrum = SpectrumCanvas()
        sp.addWidget(self.spectrum, stretch=1)
        pan_row = QHBoxLayout()
        pan_row.setContentsMargins(6, 0, 6, 2)
        lbl_pan = _hint('Pan', wrap=False)
        lbl_pan.setFixedWidth(26)
        pan_row.addWidget(lbl_pan)
        self.slider_pan = QSlider(Qt.Orientation.Horizontal)
        self.slider_pan.setRange(0, 10000)
        self.slider_pan.setEnabled(False)
        self.slider_pan.setToolTip('Pan the spectrum view (active when zoomed in)')
        pan_row.addWidget(self.slider_pan)
        sp.addLayout(pan_row)

        lower = QWidget()
        lw = QVBoxLayout(lower)
        lw.setContentsMargins(0, 2, 0, 0)
        lw.setSpacing(2)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(6, 0, 6, 0)
        view_row.addWidget(_hint('Lower view:', wrap=False))
        self.btn_view_close = QPushButton('Tone close-up')
        self.btn_view_lines = QPushButton('Tone lines over time')
        self.view_group = QButtonGroup(self)
        for b in (self.btn_view_close, self.btn_view_lines):
            b.setObjectName('btn_small')
            b.setCheckable(True)
            self.view_group.addButton(b)
            view_row.addWidget(b)
        self.btn_view_close.setChecked(True)
        view_row.addStretch()
        lw.addLayout(view_row)
        self.lower_stack = QStackedWidget()
        self.closeup = ToneCloseupCanvas()
        self.tone_view = ToneLinesCanvas()
        self.lower_stack.addWidget(self.closeup)
        self.lower_stack.addWidget(self.tone_view)
        lw.addWidget(self.lower_stack, stretch=1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(spec_pane)
        splitter.addWidget(lower)
        splitter.setSizes([400, 280])
        splitter.setChildrenCollapsible(False)
        vf.addWidget(splitter)
        cl.addWidget(self.view_frame, stretch=1)

        self.lbl_busy = QLabel(self.view_frame)
        self.lbl_busy.setObjectName('lbl_busy')
        self.lbl_busy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_busy.hide()
        self.view_frame.installEventFilter(self)
        self._update_view_mode()

        cl.addWidget(self._build_transport())
        rl.addWidget(center, stretch=1)
        rl.addWidget(self._build_controls_panel())

        self.lbl_device = QLabel('Checking for GPU …')
        self.lbl_device.setStyleSheet(f'color: {C["off"]}; padding: 0 8px;')
        self.statusBar().addPermanentWidget(self.lbl_device)
        self.statusBar().showMessage('Load a track to begin.')

    def _build_tracks_panel(self) -> QWidget:
        left = QWidget()
        left.setFixedWidth(230)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        hdr = QLabel('TRACKS')
        hdr.setObjectName('lbl_section')
        ll.addWidget(hdr)

        self.file_list = QListWidget()
        self.file_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ll.addWidget(self.file_list, stretch=1)

        self.lbl_track = QLabel('No track loaded')
        self.lbl_track.setWordWrap(True)
        self.lbl_track.setStyleSheet('color: #a6adc8; font-size: 11px; padding: 4px;')
        ll.addWidget(self.lbl_track)

        btn_add = QPushButton('+ Add Files…')
        btn_add.clicked.connect(self._browse_files)
        ll.addWidget(btn_add)
        ll.addWidget(_hint('Or drag & drop audio files onto the window.'))
        return left

    def _build_transport(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName('transport')
        pb = QVBoxLayout(frame)
        pb.setContentsMargins(10, 8, 10, 8)
        pb.setSpacing(6)

        tl = QHBoxLayout()
        self.lbl_time_cur = QLabel('0:00')
        self.lbl_time_cur.setFixedWidth(40)
        self.lbl_time_cur.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tl.addWidget(self.lbl_time_cur)
        self.slider_pos = ClickSlider(Qt.Orientation.Horizontal)
        self.slider_pos.setRange(0, 10000)
        tl.addWidget(self.slider_pos)
        self.lbl_time_tot = QLabel('0:00')
        self.lbl_time_tot.setFixedWidth(40)
        tl.addWidget(self.lbl_time_tot)
        pb.addLayout(tl)

        br = QHBoxLayout()
        br.setSpacing(8)
        self._icon_play  = _transport_icon('play', '#1e1e2e')
        self._icon_pause = _transport_icon('pause', '#1e1e2e')
        self.btn_play = QPushButton(' Play')
        self.btn_play.setIcon(self._icon_play)
        self.btn_play.setObjectName('btn_primary')
        self.btn_play.setMinimumWidth(96)
        self.btn_play.setEnabled(False)
        self.btn_play.setToolTip('Space')
        br.addWidget(self.btn_play)

        self.btn_stop = QPushButton(' Stop')
        self.btn_stop.setIcon(_transport_icon('stop', '#cdd6f4'))
        self.btn_stop.setEnabled(False)
        br.addWidget(self.btn_stop)

        br.addSpacing(12)
        label, obj = self._AB_LABELS[self._ab_mode]
        self.btn_ab = QPushButton(label)
        self.btn_ab.setObjectName(obj)
        self.btn_ab.setMinimumWidth(120)
        self.btn_ab.setEnabled(False)
        self.btn_ab.setToolTip('Tab — switch between the original and the cleaned version')
        br.addWidget(self.btn_ab)

        self.btn_solo = QPushButton('Hear Removed')
        self.btn_solo.setObjectName('btn_ab_c')
        self.btn_solo.setEnabled(False)
        self.btn_solo.setCheckable(True)
        self.btn_solo.setToolTip('R — hear only what the cleaner takes out (level-boosted). '
                                 'This should sound like the whistle/sizzle, not like music.')
        br.addWidget(self.btn_solo)

        br.addStretch()
        br.addWidget(_hint('Space ▶ · Tab A/B · R removed · ◀/▶ 5 s', wrap=False))
        br.addSpacing(12)
        br.addWidget(QLabel('Vol'))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(85)
        self.slider_vol.setFixedWidth(100)
        br.addWidget(self.slider_vol)
        pb.addLayout(br)
        return frame

    def _slider_row(self, grid: QGridLayout, row: int, label: str, lo: int, hi: int,
                    tooltip: str, color: str | None = None) -> tuple[QSlider, QLabel]:
        lbl = QLabel(label)
        lbl.setToolTip(tooltip)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setToolTip(tooltip)
        val = QLabel()
        val.setFixedWidth(62)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if color:
            val.setStyleSheet(f'color: {color};')
        grid.addWidget(lbl, row, 0)
        grid.addWidget(sl, row, 1)
        grid.addWidget(val, row, 2)
        return sl, val

    def _card(self, title: str, key: str) -> Card:
        """Collapsible section; its open / closed state is remembered."""
        card = Card(title)
        card.set_collapsed(self._settings.value(f'card_collapsed/{key}', False, type=bool))
        card.collapsed_changed.connect(
            lambda on: self._settings.setValue(f'card_collapsed/{key}', bool(on)))
        return card

    def _build_controls_panel(self) -> QScrollArea:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(4)

        # ── Strength presets ─────────────────────────────────
        box_p = self._card('Cleaning strength', 'strength')
        bp = QVBoxLayout(box_p.body)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.preset_group = QButtonGroup(self)
        self.preset_buttons = {}
        for key, p in PRESETS.items():
            b = QPushButton(p['label'])
            b.setObjectName('btn_preset')
            b.setCheckable(True)
            b.setToolTip(p['text'])
            self.preset_group.addButton(b)
            self.preset_buttons[key] = b
            row.addWidget(b)
        bp.addLayout(row)
        self.lbl_preset = _hint('')
        bp.addWidget(self.lbl_preset)
        ur = QHBoxLayout()
        ur.setSpacing(4)
        self.combo_user_presets = QComboBox()
        self.combo_user_presets.setToolTip('Your saved presets — pick one to load it.')
        ur.addWidget(self.combo_user_presets, stretch=1)
        self.btn_preset_save = QPushButton('Save as…')
        self.btn_preset_save.setToolTip('Save all current settings as a named preset')
        self.btn_preset_del = QPushButton('Delete')
        self.btn_preset_del.setToolTip('Delete the preset picked on the left')
        self.btn_preset_del.setEnabled(False)
        for b in (self.btn_preset_save, self.btn_preset_del):
            b.setObjectName('btn_small')
            ur.addWidget(b)
        bp.addLayout(ur)
        lay.addWidget(box_p)

        # ── Result ───────────────────────────────────────────
        box_q = self._card('Result', 'result')
        bq = QGridLayout(box_q.body)
        bq.setVerticalSpacing(4)
        self.lbl_tones_big = QLabel('—')
        self.lbl_tones_big.setObjectName('lbl_big')
        bq.addWidget(self.lbl_tones_big, 0, 0, 1, 2)
        self.lbl_tones_sub = _hint('AI tones found → still audible after cleaning')
        bq.addWidget(self.lbl_tones_sub, 1, 0, 1, 2)
        bq.addWidget(QLabel('Energy removed'), 2, 0)
        self.lbl_energy = QLabel('—')
        self.lbl_energy.setObjectName('lbl_metric')
        bq.addWidget(self.lbl_energy, 2, 1, alignment=Qt.AlignmentFlag.AlignRight)
        bq.addWidget(QLabel('Removed sound is'), 3, 0)
        self.lbl_flat = QLabel('—')
        self.lbl_flat.setObjectName('lbl_metric')
        self.lbl_flat.setToolTip('Spectral flatness of the removed part. Near 0 = tones (good), '
                                 'near 1 = broadband music (too much).')
        bq.addWidget(self.lbl_flat, 3, 1, alignment=Qt.AlignmentFlag.AlignRight)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        bq.addWidget(self.progress, 4, 0, 1, 2)
        lay.addWidget(box_q)

        # ── History ──────────────────────────────────────────
        box_h = self._card('History', 'history')
        bh = QVBoxLayout(box_h.body)
        bh.setSpacing(4)
        self.list_history = WheelTrapList()
        self.list_history.setMinimumHeight(190)
        self.list_history.setWordWrap(True)
        self.list_history.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.list_history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_history.setToolTip('Every result is kept here, newest on top. Double-click to go back to it.')
        bh.addWidget(self.list_history)
        hr = QHBoxLayout()
        hr.setSpacing(4)
        self.btn_hist_restore = QPushButton('Restore')
        self.btn_hist_restore.setToolTip('Go back to the selected version (double-click does the same)')
        self.btn_hist_ref = QPushButton('Compare as A')
        self.btn_hist_ref.setToolTip('Play the selected version on A, so Tab switches between it and the current result')
        self.btn_hist_ref_clear = QPushButton('A = Original')
        self.btn_hist_ref_clear.setToolTip('Play the untouched original on A again')
        for b in (self.btn_hist_restore, self.btn_hist_ref, self.btn_hist_ref_clear):
            b.setObjectName('btn_small')
            b.setEnabled(False)
            hr.addWidget(b)
        bh.addLayout(hr)
        hr2 = QHBoxLayout()
        self.chk_hist_track = QCheckBox('This track only')
        self.chk_hist_track.setChecked(self._settings.value('history_track_only', True, type=bool))
        hr2.addWidget(self.chk_hist_track)
        hr2.addStretch()
        self.btn_hist_clear = QPushButton('Clear')
        self.btn_hist_clear.setObjectName('btn_small')
        self.btn_hist_clear.setToolTip('Delete the history entries shown in the list')
        hr2.addWidget(self.btn_hist_clear)
        bh.addLayout(hr2)
        bh.addWidget(_hint('Ctrl+Z / Ctrl+Y step back / forward through the list.'))
        lay.addWidget(box_h)

        # ── 1 · AI tones ─────────────────────────────────────
        self.box_tones = self._card('1 · Remove AI tones', 'tones')
        self.box_tones.setCheckable(True)
        self.box_tones.setToolTip('Steady narrow tones (whistles) that AI generators leave in the '
                                  'high frequencies. Removed with very narrow notches.')
        bt = QVBoxLayout(self.box_tones.body)
        bt.setSpacing(6)
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        self.sl_sens, self.lbl_sens = self._slider_row(
            g, 0, 'Sensitivity', 2, 10,
            'How far above its surroundings a peak must stand to count as a tone. '
            'Further right = catches fainter tones.')
        self.sl_tdepth, self.lbl_tdepth = self._slider_row(
            g, 1, 'Reduction', -80, 0, 'How much each tone is reduced.', C['fade'])
        g.addWidget(QLabel('Search above'), 2, 0)
        self.spin_minhz = QSpinBox()
        self.spin_minhz.setRange(1000, 20000)
        self.spin_minhz.setSingleStep(500)
        self.spin_minhz.setSuffix(' Hz')
        self.spin_minhz.setKeyboardTracking(False)
        g.addWidget(self.spin_minhz, 2, 1, 1, 2)
        bt.addLayout(g)
        self.chk_pass2 = QCheckBox('Second pass (catch leftovers)')
        bt.addWidget(self.chk_pass2)

        tr = QHBoxLayout()
        self.btn_detect = QPushButton('Auto-detect tones')
        self.btn_detect.setToolTip('Find the AI tones in this track and replace the tone list with them')
        self.btn_detect.setObjectName('btn_small')
        self.btn_detect.setEnabled(False)
        tr.addWidget(self.btn_detect)
        self.btn_show_list = QPushButton('Show tone list ▸')
        self.btn_show_list.setObjectName('btn_link')
        self.btn_show_list.setCheckable(True)
        tr.addWidget(self.btn_show_list)
        tr.addStretch()
        bt.addLayout(tr)

        self.list_box = QWidget()
        lb = QVBoxLayout(self.list_box)
        lb.setContentsMargins(0, 0, 0, 0)
        self.tbl_notches = WheelTrapTable(0, 4)
        self.tbl_notches.setHorizontalHeaderLabels(['', 'Freq Hz', 'Width Hz', 'Stands out'])
        hh = self.tbl_notches.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.tbl_notches.setColumnWidth(0, 26)
        self.tbl_notches.verticalHeader().setVisible(False)
        self.tbl_notches.verticalHeader().setDefaultSectionSize(22)
        self.tbl_notches.setAlternatingRowColors(True)
        self.tbl_notches.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_notches.setMinimumHeight(170)
        self.tbl_notches.setToolTip('Double-click a frequency or width to edit. Untick to keep a tone.')
        lb.addWidget(self.tbl_notches)
        nr = QHBoxLayout()
        nr.setSpacing(4)
        self.btn_notch_add = QPushButton('Add')
        self.btn_notch_del = QPushButton('Remove')
        self.btn_notch_clear = QPushButton('Clear')
        for b in (self.btn_notch_add, self.btn_notch_del, self.btn_notch_clear):
            b.setObjectName('btn_small')
            nr.addWidget(b)
        lb.addLayout(nr)
        lb.addWidget(_hint('Shift+click the spectrum or right-click a line in the tone view to add one.'))
        self.list_box.hide()
        bt.addWidget(self.list_box)
        lay.addWidget(self.box_tones)

        # ── 2 · High-band softening ──────────────────────────
        self.box_shelf = self._card('2 · Soften the harsh top band', 'shelf')
        self.box_shelf.setCheckable(True)
        self.box_shelf.setToolTip('Turns the top band down (the "sizzle"). Drag the orange / red lines '
                                  'on the spectrum with left / right mouse button.')
        bs = QGridLayout(self.box_shelf.body)
        bs.setHorizontalSpacing(8)
        self.sl_sdepth, self.lbl_sdepth = self._slider_row(
            bs, 0, 'Reduction', -48, 0, 'How much the band above the red line is turned down.', C['fade'])
        bs.addWidget(QLabel('Fade from'), 1, 0)
        self.spin_fade = QDoubleSpinBox()
        bs.addWidget(self.spin_fade, 1, 1, 1, 2)
        bs.addWidget(QLabel('Full from'), 2, 0)
        self.spin_cut = QDoubleSpinBox()
        bs.addWidget(self.spin_cut, 2, 1, 1, 2)
        for sb, col in ((self.spin_fade, C['fade']), (self.spin_cut, C['cut'])):
            sb.setRange(20.0, 96000.0)
            sb.setDecimals(0)
            sb.setSingleStep(250.0)
            sb.setSuffix(' Hz')
            sb.setKeyboardTracking(False)
            sb.setStyleSheet(f'color: {col};')
        bs.addWidget(QLabel('Curve'), 3, 0)
        self.combo_curve = QComboBox()
        for label, key in self._CURVES:
            self.combo_curve.addItem(label, key)
        bs.addWidget(self.combo_curve, 3, 1, 1, 2)
        bs.addWidget(_hint('Or drag the orange / red lines on the spectrum (left / right mouse).'),
                     4, 0, 1, 3)
        lay.addWidget(self.box_shelf)

        # ── Export ───────────────────────────────────────────
        box_e = self._card('Save', 'save')
        be = QVBoxLayout(box_e.body)
        er = QHBoxLayout()
        self.btn_export = QPushButton('Save')
        self.btn_export.setObjectName('btn_primary')
        self.btn_export.setEnabled(False)
        self.btn_export_diff = QPushButton('Save Removed')
        self.btn_export_diff.setEnabled(False)
        self.btn_export_diff.setToolTip('Save only the part that was removed (original minus cleaned)')
        self.btn_batch = QPushButton('Save All…')
        self.btn_batch.setEnabled(False)
        self.btn_batch.setToolTip('Clean every track in the list — choose the settings in the next window')
        for b in (self.btn_export, self.btn_export_diff, self.btn_batch):
            er.addWidget(b)
        be.addLayout(er)
        self.lbl_export = _hint('')
        be.addWidget(self.lbl_export)
        lay.addWidget(box_e)

        # ── Advanced ─────────────────────────────────────────
        self.box_adv = self._card('Advanced', 'advanced')
        ba = QGridLayout(self.box_adv.body)
        ba.setHorizontalSpacing(8)
        ba.setVerticalSpacing(6)
        ba.addWidget(QLabel('Method'), 0, 0)
        self.combo_method = QComboBox()
        for label, key in self._METHODS:
            self.combo_method.addItem(label, key)
        self.combo_method.setToolTip(
            'Static reduces the targeted bins by a fixed amount. Adaptive only removes the steady '
            'part and lets loud moments through.')
        ba.addWidget(self.combo_method, 0, 1, 1, 3)

        self.adaptive_box = QWidget()
        ag = QGridLayout(self.adaptive_box)
        ag.setContentsMargins(0, 0, 0, 0)
        ag.setHorizontalSpacing(8)
        self.spin_oversub = QDoubleSpinBox()
        self.spin_oversub.setRange(0.5, 5.0)
        self.spin_oversub.setSingleStep(0.1)
        self.spin_oversub.setDecimals(1)
        self.spin_oversub.setSuffix(' ×')
        self.spin_oversub.setToolTip('Oversubtraction: how hard the noise profile is subtracted.')
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(1, 90)
        self.spin_pct.setSuffix(' %')
        self.spin_pct.setToolTip('Noise profile = this percentile of each bin over time.')
        self.spin_attack = QSpinBox()
        self.spin_attack.setRange(0, 500)
        self.spin_attack.setSuffix(' ms')
        self.spin_attack.setToolTip('How fast the gain opens when music arrives in the bin.')
        self.spin_release = QSpinBox()
        self.spin_release.setRange(0, 2000)
        self.spin_release.setSuffix(' ms')
        self.spin_release.setToolTip('How fast the gain closes again afterwards.')
        for sb in (self.spin_oversub, self.spin_pct, self.spin_attack, self.spin_release):
            sb.setKeyboardTracking(False)
        ag.addWidget(QLabel('Oversub'), 0, 0)
        ag.addWidget(self.spin_oversub, 0, 1)
        ag.addWidget(QLabel('Percentile'), 0, 2)
        ag.addWidget(self.spin_pct, 0, 3)
        ag.addWidget(QLabel('Attack'), 1, 0)
        ag.addWidget(self.spin_attack, 1, 1)
        ag.addWidget(QLabel('Release'), 1, 2)
        ag.addWidget(self.spin_release, 1, 3)
        self.chk_smooth = QCheckBox('Smooth gain (3×3 median)')
        ag.addWidget(self.chk_smooth, 2, 0, 1, 4)
        prof = QHBoxLayout()
        prof.addWidget(QLabel('Noise profile:'))
        self.lbl_profile = QLabel('Whole track')
        self.lbl_profile.setStyleSheet(f'color: {C["proc"]};')
        prof.addWidget(self.lbl_profile, stretch=1)
        self.btn_profile_clear = QPushButton('Whole track')
        self.btn_profile_clear.setObjectName('btn_small')
        self.btn_profile_clear.setEnabled(False)
        prof.addWidget(self.btn_profile_clear)
        ag.addLayout(prof, 3, 0, 1, 4)
        ag.addWidget(_hint('Drag across a quiet part of the tone view to take the profile from there.'),
                     4, 0, 1, 4)
        ba.addWidget(self.adaptive_box, 1, 0, 1, 4)

        ba.addWidget(QLabel('Stereo'), 2, 0)
        self.combo_stereo = QComboBox()
        for label, key in self._STEREO:
            self.combo_stereo.addItem(label, key)
        self.combo_stereo.setToolTip('The whistle is usually dead-centre while the music is wide. '
                                     'Mid-only or coherence scaling keeps the stereo air intact.')
        ba.addWidget(self.combo_stereo, 2, 1, 1, 3)
        ba.addWidget(QLabel('FFT size'), 3, 0)
        self.combo_res = QComboBox()
        for label, key in self._RESOLUTIONS:
            self.combo_res.addItem(label, key)
        ba.addWidget(self.combo_res, 3, 1, 1, 3)
        self.lbl_res = _hint('')
        ba.addWidget(self.lbl_res, 4, 0, 1, 4)
        ba.addWidget(QLabel('Runs on'), 5, 0)
        self.lbl_device_adv = QLabel('…')
        ba.addWidget(self.lbl_device_adv, 5, 1, 1, 3)
        lay.addWidget(self.box_adv)
        lay.addStretch()

        for card in panel.findChildren(Card):
            card.body.layout().setContentsMargins(0, 0, 0, 0)
        for combo in panel.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(390)
        return scroll

    # ── Busy indicator / steps ───────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.view_frame and event.type() == QEvent.Type.Resize:
            self._place_busy()
        return super().eventFilter(obj, event)

    def _place_busy(self):
        self.lbl_busy.adjustSize()
        w = self.lbl_busy.width()
        self.lbl_busy.move(max((self.view_frame.width() - w) // 2, 0), 44)

    def _set_busy(self, msg: str | None):
        busy = msg is not None
        if busy:
            self.lbl_busy.setText(f'⏳  {msg}')
            self._place_busy()
            self.lbl_busy.raise_()
            self._set_status(msg)
            self._update_steps(2)
        self.lbl_busy.setVisible(busy)
        self.progress.setVisible(busy)
        ready = not busy and self._audio_proc is not None
        for b in (self.btn_export, self.btn_export_diff):
            b.setEnabled(ready)

    def _update_steps(self, current: int):
        parts = []
        for i, text in enumerate(self._STEPS, start=1):
            if i < current:
                parts.append(f'<span style="color:#a6e3a1;">✔ {text}</span>')
            elif i == current:
                parts.append(f'<span style="color:#f9e2af; font-weight:bold;">{i} · {text}</span>')
            else:
                parts.append(f'<span style="color:#6c7086;">{i} · {text}</span>')
        self.lbl_steps.setText('&nbsp;&nbsp;→&nbsp;&nbsp;'.join(parts))

    # ── Settings ↔ controls ──────────────────────────────────

    @staticmethod
    def _select_data(combo: QComboBox, value):
        combo.setCurrentIndex(max(combo.findData(value), 0))

    def _settings_to_controls(self):
        self._updating = True
        s = self.fs
        if s.preset in PRESETS:
            self.preset_buttons[s.preset].setChecked(True)
            self.lbl_preset.setText(PRESETS[s.preset]['text'])
        elif s.preset.startswith('user:'):
            self._uncheck_preset_buttons()
            self.lbl_preset.setText(f'Your preset: {s.preset[5:]}')
        else:
            self._show_custom_preset()
        self._refresh_preset_combo()

        self.box_tones.setChecked(s.tones_on)
        self.sl_sens.setValue(int(round(12 - s.threshold_db)))
        self.lbl_sens.setText(f'{s.threshold_db:.0f} dB')
        self.sl_tdepth.setValue(int(round(s.tone_depth_db)))
        self.lbl_tdepth.setText(f'{s.tone_depth_db:+.0f} dB')
        self.spin_minhz.setValue(int(s.min_hz))
        self.chk_pass2.setChecked(s.second_pass)

        self.box_shelf.setChecked(s.shelf_on)
        self.sl_sdepth.setValue(int(round(s.shelf_depth_db)))
        self.lbl_sdepth.setText(f'{s.shelf_depth_db:+.0f} dB')
        self.spin_fade.setValue(s.fade_start)
        self.spin_cut.setValue(s.cutoff)
        self._select_data(self.combo_curve, s.curve)

        self._select_data(self.combo_method, s.method)
        self.adaptive_box.setVisible(s.method == 'adaptive')
        self.spin_oversub.setValue(s.oversub)
        self.spin_pct.setValue(int(s.percentile))
        self.spin_attack.setValue(int(s.attack_ms))
        self.spin_release.setValue(int(s.release_ms))
        self.chk_smooth.setChecked(s.smoothing)
        self._select_data(self.combo_stereo, s.stereo)
        self._select_data(self.combo_res, s.resolution)
        self._updating = False

        self.spectrum.set_shelf(s.shelf_on, s.fade_start, s.cutoff)
        self._show_profile_region()
        self._update_resolution_label()
        self._update_export_label()
        self._refresh_notch_views()

    def _save_filter_settings(self):
        """Knobs and the tone list — the next track and the next app start use both."""
        self._settings.setValue('filter_settings_v2', self.fs.to_json())
        self._settings.setValue('last_notches', json.dumps(self.fs.notches))

    # ── Signals ───────────────────────────────────────────────

    def _connect_signals(self):
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)

        self.spectrum.shelf_changed.connect(self._on_canvas_shelf)
        self.spectrum.notches_edited.connect(lambda n: self._set_notches(n, manual=True))
        self.spectrum.view_changed.connect(self._sync_pan_slider)
        self.slider_pan.sliderMoved.connect(self._on_pan_slider_moved)
        self.btn_zoom_full.clicked.connect(lambda: self.spectrum.set_view(*self.spectrum._zoom_xlim_full))
        self.btn_zoom_band.clicked.connect(self._zoom_tone_band)

        self.tone_view.region_selected.connect(self._on_region_selected)
        self.closeup.add_notch_requested.connect(self._add_notch_at)
        self.btn_view_close.toggled.connect(lambda on: self.lower_stack.setCurrentIndex(0 if on else 1))
        self.tone_view.add_notch_requested.connect(self._add_notch_at)
        self.tone_view.seek_requested.connect(
            lambda t: self._seek_to(int(t * self.sample_rate)) if self._audio_orig is not None else None)

        self.preset_group.buttonClicked.connect(self._on_preset_clicked)
        self.combo_user_presets.activated.connect(self._on_user_preset_activated)
        self.combo_user_presets.currentIndexChanged.connect(
            lambda i: self.btn_preset_del.setEnabled(i > 0))
        self.btn_preset_save.clicked.connect(self._save_user_preset)
        self.btn_preset_del.clicked.connect(self._delete_user_preset)

        self.list_history.itemSelectionChanged.connect(self._update_history_buttons)
        self.list_history.itemDoubleClicked.connect(lambda _item: self._restore_selected_history())
        self.btn_hist_restore.clicked.connect(self._restore_selected_history)
        self.btn_hist_ref.clicked.connect(self._compare_history_as_a)
        self.btn_hist_ref_clear.clicked.connect(self._clear_ref)
        self.btn_hist_clear.clicked.connect(self._clear_history)
        self.chk_hist_track.toggled.connect(self._on_hist_filter_toggled)

        self.box_tones.toggled.connect(lambda v: self._on_control('tones_on', bool(v)))
        self.sl_sens.valueChanged.connect(self._on_sens_changed)
        self.sl_tdepth.valueChanged.connect(self._on_tdepth_changed)
        self.spin_minhz.valueChanged.connect(lambda v: self._on_control('min_hz', float(v)))
        self.chk_pass2.toggled.connect(lambda v: self._on_control('second_pass', bool(v)))
        self.btn_detect.clicked.connect(lambda: self._request_clean(detect=True, delay=0, force=True))
        self.btn_show_list.toggled.connect(self._on_show_list)
        self.tbl_notches.itemChanged.connect(self._on_notch_item_changed)
        self.btn_notch_add.clicked.connect(lambda: self._add_notch_at(self._default_new_notch_freq()))
        self.btn_notch_del.clicked.connect(self._remove_selected_notches)
        self.btn_notch_clear.clicked.connect(lambda: self._set_notches([], manual=True))

        self.box_shelf.toggled.connect(self._on_shelf_toggled)
        self.sl_sdepth.valueChanged.connect(self._on_sdepth_changed)
        self.spin_fade.valueChanged.connect(self._on_spin_shelf)
        self.spin_cut.valueChanged.connect(self._on_spin_shelf)
        self.combo_curve.currentIndexChanged.connect(
            lambda _: self._on_control('curve', self.combo_curve.currentData()))

        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.spin_oversub.valueChanged.connect(lambda v: self._on_control('oversub', float(v)))
        self.spin_pct.valueChanged.connect(lambda v: self._on_control('percentile', float(v)))
        self.spin_attack.valueChanged.connect(lambda v: self._on_control('attack_ms', float(v)))
        self.spin_release.valueChanged.connect(lambda v: self._on_control('release_ms', float(v)))
        self.chk_smooth.toggled.connect(lambda v: self._on_control('smoothing', bool(v)))
        self.btn_profile_clear.clicked.connect(lambda: self._on_region_selected(None, None))
        self.combo_stereo.currentIndexChanged.connect(
            lambda _: self._on_control('stereo', self.combo_stereo.currentData()))
        self.combo_res.currentIndexChanged.connect(self._on_resolution_changed)

        self.slider_vol.valueChanged.connect(lambda v: setattr(self, '_cb_volume', v / 100.0))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_ab.clicked.connect(self._toggle_ab)
        self.btn_solo.clicked.connect(self._toggle_solo_residual)
        self.btn_export.clicked.connect(self._export_single)
        self.btn_export_diff.clicked.connect(self._export_diff)
        self.btn_batch.clicked.connect(self._export_batch)

        self.slider_pos.sliderPressed.connect(self._timeline_pressed)
        self.slider_pos.sliderReleased.connect(self._timeline_released)

    # ── Control handlers ─────────────────────────────────────

    def _on_control(self, name: str, value, detect: bool | None = None):
        if self._updating or getattr(self.fs, name) == value:
            return
        setattr(self.fs, name, value)
        if ((name in PRESET_FIELDS or self.fs.preset.startswith('user:'))
                and self.fs.preset != 'custom'):
            self.fs.preset = 'custom'
            self._show_custom_preset()
        self._save_filter_settings()
        self._request_clean(detect=(name in DETECT_FIELDS) if detect is None else detect)

    def _uncheck_preset_buttons(self):
        self.preset_group.setExclusive(False)
        for b in self.preset_buttons.values():
            b.setChecked(False)
        self.preset_group.setExclusive(True)

    def _show_custom_preset(self):
        self._uncheck_preset_buttons()
        self.lbl_preset.setText('Custom settings (you changed something below). '
                                'Click a strength or load a preset to go back.')
        self.combo_user_presets.blockSignals(True)
        self.combo_user_presets.setCurrentIndex(0)
        self.combo_user_presets.blockSignals(False)
        self.btn_preset_del.setEnabled(False)

    # ── User presets (one JSON file each in presets/) ────────

    def _refresh_preset_combo(self):
        found = {}
        if self.PRESET_DIR.is_dir():
            for p in sorted(self.PRESET_DIR.glob('*.json')):
                data = read_json(p, None)
                if isinstance(data, dict) and isinstance(data.get('settings'), dict):
                    found[str(data.get('name') or p.stem)] = (p, data['settings'])
        self._user_preset_map = found
        combo = self.combo_user_presets
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('My presets…' if found else 'No saved presets yet', None)
        for name in sorted(found, key=str.lower):
            combo.addItem(name, name)
        current = self.fs.preset[5:] if self.fs.preset.startswith('user:') else None
        combo.setCurrentIndex(max(combo.findData(current), 0) if current else 0)
        combo.blockSignals(False)
        self.btn_preset_del.setEnabled(combo.currentIndex() > 0)

    @staticmethod
    def _preset_filename(name: str) -> str:
        return re.sub(r'[^\w\- ]+', '_', name).strip() or 'preset'

    def _save_user_preset(self):
        current = self.fs.preset[5:] if self.fs.preset.startswith('user:') else ''
        name, ok = QInputDialog.getText(self, 'Save preset', 'Preset name:', text=current)
        name = name.strip()
        if not ok or not name:
            return
        existing = self._user_preset_map.get(name)
        if existing:
            if QMessageBox.question(self, 'Save preset', f'Replace the preset "{name}"?') \
                    != QMessageBox.StandardButton.Yes:
                return
            path = existing[0]
        else:
            stem = self._preset_filename(name)
            path, n = self.PRESET_DIR / f'{stem}.json', 2
            while path.exists():
                path, n = self.PRESET_DIR / f'{stem} ({n}).json', n + 1
        s = self.fs.copy()
        s.preset = f'user:{name}'
        err = write_json(path, {'name': name, 'settings': s.to_dict(notches=False)})
        if err:
            QMessageBox.critical(self, 'Save preset', f'Could not save the preset:\n{err}')
            return
        self.fs.preset = s.preset
        self._save_filter_settings()
        self._uncheck_preset_buttons()
        self.lbl_preset.setText(f'Your preset: {name}')
        self._refresh_preset_combo()
        self._set_status(f'Preset saved: {name}  ({path})')

    def _on_user_preset_activated(self, idx: int):
        name = self.combo_user_presets.itemData(idx)
        if name is None:
            return
        entry = self._user_preset_map.get(name)
        if entry is None:
            self._refresh_preset_combo()
            return
        s = FilterSettings.from_dict(
            {k: v for k, v in entry[1].items() if k not in FilterSettings._TRANSIENT})
        s.preset = f'user:{name}'
        s.notches = [dict(n) for n in self.fs.notches]
        s.profile_region = self.fs.profile_region
        self.fs = s
        self._settings_to_controls()
        self._save_filter_settings()
        self._request_clean(detect=False, delay=0)
        self._set_status(f'Preset loaded: {name}')

    def _delete_user_preset(self):
        name = self.combo_user_presets.currentData()
        entry = self._user_preset_map.get(name) if name is not None else None
        if entry is None:
            return
        if QMessageBox.question(self, 'Delete preset', f'Delete the preset "{name}"?') \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            entry[0].unlink()
        except OSError as e:
            QMessageBox.critical(self, 'Delete preset', f'Could not delete the preset:\n{e}')
            return
        if self.fs.preset == f'user:{name}':
            self.fs.preset = 'custom'
            self._show_custom_preset()
            self._save_filter_settings()
        self._refresh_preset_combo()
        self._set_status(f'Preset deleted: {name}')

    def _on_preset_clicked(self, button):
        key = next(k for k, b in self.preset_buttons.items() if b is button)
        self.fs.apply_preset(key)
        self._settings_to_controls()
        self._save_filter_settings()
        self._request_clean(detect=True, delay=0)

    def _on_sens_changed(self, v: int):
        thr = float(12 - v)
        self.lbl_sens.setText(f'{thr:.0f} dB')
        self.closeup.set_threshold(thr)
        self._on_control('threshold_db', thr)

    def _on_tdepth_changed(self, v: int):
        self.lbl_tdepth.setText(f'{v:+d} dB')
        self._on_control('tone_depth_db', float(v))

    def _on_sdepth_changed(self, v: int):
        self.lbl_sdepth.setText(f'{v:+d} dB')
        self._on_control('shelf_depth_db', float(v))

    def _on_shelf_toggled(self, on: bool):
        if not self._updating:
            self._set_shelf(bool(on), self.fs.fade_start, self.fs.cutoff)

    def _on_method_changed(self, _idx: int):
        self.adaptive_box.setVisible(self.combo_method.currentData() == 'adaptive')
        self._on_control('method', self.combo_method.currentData())

    def _on_resolution_changed(self, _idx: int):
        self._on_control('resolution', int(self.combo_res.currentData()), detect=False)
        self._update_resolution_label()

    def _on_show_list(self, show: bool):
        self.list_box.setVisible(show)
        self._update_list_button()

    def _update_list_button(self):
        arrow = '▾' if self.btn_show_list.isChecked() else '▸'
        self.btn_show_list.setText(f'{"Hide" if self.btn_show_list.isChecked() else "Show"} '
                                   f'tone list ({len(self.fs.notches)}) {arrow}')

    def _update_resolution_label(self):
        sr = self.sample_rate
        nt = self.processor.nperseg_tones(self.fs, sr)
        ns = self.processor.nperseg_shelf(self.fs, sr)
        self.lbl_res.setText(f'Tones: {nt} pts ({sr / nt:.1f} Hz/bin) · '
                             f'top band: {ns} pts ({sr / ns:.1f} Hz/bin) @ {sr} Hz')

    def _update_export_label(self):
        label = EXPORT_FORMATS.get(self.export_format, EXPORT_FORMATS['wav24'])[0]
        suffix = self.name_suffix or '(none)'
        self.lbl_export.setText(f'{label} · suffix {suffix} · to output/ · change in Settings menu')

    def _zoom_tone_band(self):
        lo = max(self.fs.min_hz * 0.8, 1000.0)
        hi = (self.processor.bandwidth * 1.08) if self.processor.loaded else self.sample_rate / 2.0
        self.spectrum.set_view(lo, max(hi, lo * 2))

    # ── Shelf ────────────────────────────────────────────────

    def _set_shelf(self, on: bool, fade_start: float, cutoff: float):
        """Shared by the canvas lines and the spin boxes."""
        s = self.fs
        if (s.shelf_on, s.fade_start, s.cutoff) == (on, fade_start, cutoff):
            return
        s.shelf_on, s.fade_start, s.cutoff = on, fade_start, cutoff
        self._updating = True
        self.box_shelf.setChecked(on)
        self.spin_fade.setValue(fade_start)
        self.spin_cut.setValue(cutoff)
        self._updating = False
        self.spectrum.set_shelf(on, fade_start, cutoff)
        self._refresh_notch_views()
        if s.preset != 'custom':
            s.preset = 'custom'
            self._show_custom_preset()
        self._save_filter_settings()
        self._request_clean(detect=False)

    def _on_canvas_shelf(self, fade_start: float, cutoff: float):
        self._set_shelf(True, float(round(fade_start)), float(round(cutoff)))

    def _on_spin_shelf(self, _v):
        if self._updating:
            return
        fade, cut = self.spin_fade.value(), self.spin_cut.value()
        if cut < fade + SpectrumCanvas.MIN_SEP:
            if self.sender() is self.spin_fade:
                fade = cut - SpectrumCanvas.MIN_SEP
            else:
                cut = fade + SpectrumCanvas.MIN_SEP
        self._set_shelf(self.fs.shelf_on, fade, cut)

    # ── Tones / notches ──────────────────────────────────────

    def _set_notches(self, notches: list, manual: bool, reprocess: bool = True):
        nyq = self.sample_rate / 2.0
        clean = [{**n, 'freq': round(float(np.clip(float(n['freq']), 20.0, nyq - 1.0)), 1)}
                 for n in notches]
        self.fs.notches = sorted(clean, key=lambda n: n['freq'])
        if manual and self.current_path:
            self._manual_notches[self.current_path] = [dict(n) for n in self.fs.notches]
        self._save_filter_settings()
        self._refresh_notch_views()
        if reprocess and self._audio_orig is not None:
            self._request_clean(detect=False)

    def _refresh_notch_views(self):
        self.spectrum.set_notches(self.fs.notches)
        self.closeup.set_notches(self.fs.notches)
        self.closeup.set_threshold(self.fs.threshold_db)
        self.tone_view.set_overlay(self.fs.notches, self.fs.shelf_on, self.fs.fade_start, self.fs.cutoff)
        self._refresh_notch_table()
        self._update_list_button()

    def _default_new_notch_freq(self) -> float:
        c = self.spectrum.view_center()
        return c if c >= self.fs.min_hz else max(self.fs.min_hz, 8000.0)

    def _add_notch_at(self, freq: float):
        if not self.fs.tones_on:
            self.box_tones.setChecked(True)
        self._set_notches(self.fs.notches + [
            {'freq': round(float(freq), 1), 'width': SpectrumCanvas.DEFAULT_WIDTH, 'on': True}], manual=True)

    def _remove_selected_notches(self):
        rows = {i.row() for i in self.tbl_notches.selectedIndexes()}
        if rows:
            self._set_notches([n for i, n in enumerate(self.fs.notches) if i not in rows], manual=True)

    def _refresh_notch_table(self):
        self._table_updating = True
        t = self.tbl_notches
        t.setRowCount(len(self.fs.notches))
        editable = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        for r, n in enumerate(self.fs.notches):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable |
                         Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked if n.get('on', True) else Qt.CheckState.Unchecked)
            t.setItem(r, 0, chk)
            fi = QTableWidgetItem(f"{n['freq']:.1f}")
            fi.setFlags(editable)
            t.setItem(r, 1, fi)
            wi = QTableWidgetItem(f"{n['width']:.1f}")
            wi.setFlags(editable)
            t.setItem(r, 2, wi)
            ii = QTableWidgetItem(f"+{n['prom']:.1f} dB" if 'prom' in n else 'added by hand')
            ii.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            t.setItem(r, 3, ii)
        self._table_updating = False

    def _on_notch_item_changed(self, item: QTableWidgetItem):
        if self._table_updating:
            return
        r, c = item.row(), item.column()
        if r >= len(self.fs.notches):
            return
        notches = [dict(n) for n in self.fs.notches]
        n = notches[r]
        if c == 0:
            n['on'] = item.checkState() == Qt.CheckState.Checked
        elif c in (1, 2):
            try:
                v = float(item.text().replace(',', '.').replace('Hz', '').strip())
            except ValueError:
                v = None
            if c == 1 and v is not None and 20.0 <= v < self.sample_rate / 2.0:
                n['freq'] = v
            elif c == 2 and v is not None and 0.5 <= v <= 5000.0:
                n['width'] = v
        QTimer.singleShot(0, lambda: self._set_notches(notches, manual=True))

    # ── Noise-profile region (adaptive) ──────────────────────

    def _on_region_selected(self, t0, t1, reprocess: bool = True):
        if t0 is None:
            self.fs.profile_region = None
        else:
            self.fs.profile_region = (float(t0), float(t1))
            if self.fs.method != 'adaptive':
                self._select_data(self.combo_method, 'adaptive')
        self._show_profile_region()
        if reprocess and self._audio_orig is not None and self.fs.method == 'adaptive':
            self._request_clean(detect=False)

    def _show_profile_region(self):
        r = self.fs.profile_region
        self.lbl_profile.setText(f'{fmt_time(r[0], True)} – {fmt_time(r[1], True)}' if r else 'Whole track')
        self.btn_profile_clear.setEnabled(bool(r))
        self.tone_view.set_region(r)

    # ── File Loading ──────────────────────────────────────────

    def _add_paths(self, paths):
        existing = {self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.file_list.count())}
        for p in paths:
            p = str(p)
            if Path(p).suffix.lower() in AUDIO_EXTS and p not in existing:
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setToolTip(p)
                self.file_list.addItem(item)
                existing.add(p)
        self.btn_batch.setEnabled(self.file_list.count() > 0)

    def _load_input_folder(self):
        if self.INPUT_DIR.exists():
            self._add_paths(sorted(self.INPUT_DIR.iterdir()))

    def _browse_files(self):
        exts = ' '.join(f'*{e}' for e in sorted(AUDIO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add Audio Files', '', f'Audio Files ({exts})')
        self._add_paths(paths)

    def _on_file_selected(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        self._switch_was_playing = self.is_playing
        self._stop()
        self._disable_playback_ui()
        self._proc_generation += 1       # drop any running clean for the old track
        self._clean_debounce.stop()
        self._set_busy(f'Loading {Path(path).name} …')

        self._load_generation += 1
        if self._load_worker and self._load_worker.isRunning():
            self._old_workers.append(self._load_worker)
        self._load_worker = LoadWorker(path, self._load_generation)
        self._load_worker.done.connect(self._on_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_error(self, msg: str, generation: int):
        if generation == self._load_generation:
            self._set_busy(None)
            self._set_status(f'Could not open the file: {msg}')

    def _on_loaded(self, payload: dict):
        if payload['generation'] != self._load_generation:
            return
        self._proc_generation += 1

        self.processor    = payload['processor']
        self._audio_orig  = self.processor.audio
        self._audio_proc  = None
        self.sample_rate  = payload['sr']
        self.current_path = payload['path']
        self._cb_frame    = 0
        self._residual_boost = 1.0

        dur, n_ch = self.processor.duration, self.processor.n_ch
        self.lbl_track.setText(
            f"{Path(self.current_path).name}\n{fmt_time(dur)} · {self.sample_rate} Hz · {n_ch} ch · "
            f"content up to {self.processor.bandwidth / 1000:.1f} kHz")
        self._update_total_time()
        self._update_resolution_label()

        self.spectrum.set_nyquist(self.sample_rate / 2.0)
        self.spectrum.plot_original(payload['spec_f'], payload['spec_p'])
        self.tone_view.set_original(self.processor.tone_view(), self.fs.min_hz - 1000.0,
                                    self.processor.bandwidth + 300.0)
        self.closeup.set_original(payload['spec_f'], self.processor.tone_prominence(),
                                  max(self.fs.min_hz - 1000.0, 1000.0), self.processor.bandwidth + 300.0)
        self._on_region_selected(None, None, reprocess=False)
        self._clear_result()
        self._clear_ref()
        self._refresh_history_list()

        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.combo_stereo.setEnabled(n_ch == 2)

        # Keep the last used tone list; only detect when asked to (or nothing was ever found).
        if self.current_path in self._manual_notches:
            self._set_notches(self._manual_notches[self.current_path], manual=False, reprocess=False)
            self._request_clean(detect=False, delay=0)
        elif self.detect_on_load or not self._have_last_notches:
            self._request_clean(detect=True, delay=0)
        else:
            self._set_notches(self.fs.notches, manual=False, reprocess=False)
            self._request_clean(detect=False, delay=0)

        if getattr(self, '_switch_was_playing', False):
            self._play()

    # ── Cleaning pipeline ────────────────────────────────────

    def _detect_key(self) -> tuple:
        s = self.fs
        return (self.current_path, s.threshold_db, s.min_hz, s.second_pass,
                s.tone_depth_db if s.second_pass else None, s.method, s.stereo, s.resolution)

    def _request_clean(self, detect: bool, delay: int = 350, force: bool = False):
        if self._audio_orig is None:
            return
        if detect and force:
            self._detect_cache.pop(self._detect_key(), None)
            self._manual_notches.pop(self.current_path, None)
        if detect:
            self._manual_notches.pop(self.current_path, None)
        self._pending_detect = self._pending_detect or detect
        self._set_busy('Waiting for you to finish adjusting …' if delay else 'Starting …')
        self._clean_debounce.start(delay)

    def _run_clean(self):
        if self._audio_orig is None:
            return
        if self._clean_worker and self._clean_worker.isRunning():
            self._clean_debounce.start(150)
            return
        detect = self._pending_detect and self.fs.tones_on
        self._pending_detect = False
        s = self.fs.copy()
        if detect:
            cached = self._detect_cache.get(self._detect_key())
            if cached is not None:
                s.notches = [dict(n) for n in cached]
                self.fs.notches = [dict(n) for n in cached]
                self._refresh_notch_views()
                detect = False
        self._proc_generation += 1
        self._clean_worker = CleanWorker(self.processor, s, detect, self._proc_generation)
        self._clean_worker.progress.connect(self._on_clean_progress)
        self._clean_worker.done.connect(self._on_cleaned)
        self._clean_worker.error.connect(self._on_clean_error)
        self._clean_worker.start()
        self._set_busy('Starting …')

    def _on_clean_progress(self, msg: str, generation: int):
        if generation == self._proc_generation:
            self._set_busy(msg)

    def _on_clean_error(self, msg: str, generation: int):
        if generation == self._proc_generation:
            self._set_busy(None)
            self._set_status(f'Cleaning failed: {msg}')

    def _on_cleaned(self, payload: dict):
        if payload['generation'] != self._proc_generation or self._audio_orig is None:
            return
        if self._clean_debounce.isActive():   # newer changes are queued — keep the spinner
            return
        proc = payload['audio']
        if proc.shape != self._audio_orig.shape:
            return
        if payload['detected']:
            self._detect_cache[self._detect_key()] = [dict(n) for n in payload['notches']]
            self._set_notches(payload['notches'], manual=False, reprocess=False)

        self._audio_proc = proc
        self._residual_boost = payload['boost']
        self._set_busy(None)

        self.spectrum.plot_processed(payload['psd_f'], payload['psd_proc'])
        self.tone_view.set_processed(payload['view_proc'], payload['view_res'])
        self.closeup.set_processed(payload['prom_proc'])
        self._show_result(payload)
        for b in (self.btn_ab, self.btn_solo):
            b.setEnabled(True)
        self._update_steps(3)
        self._have_last_notches = True
        self._save_filter_settings()
        self._record_history(payload)

        s = payload['settings']
        parts = []
        if s.tones_on:
            parts.append(f'{sum(1 for n in s.notches if n.get("on", True))} tones notched')
        if s.shelf_on:
            parts.append(f'top band {s.shelf_depth_db:+.0f} dB from {s.fade_start / 1000:.1f} kHz')
        self._update_device_label()
        self._set_status('Done — ' + (', '.join(parts) if parts else 'nothing enabled') +
                         f' ({payload["seconds"]:.1f} s on {payload["device"]}).'
                         '   Press Tab to compare, R to hear what was removed.')

    def _clear_result(self):
        self.lbl_tones_big.setText('—')
        self.lbl_tones_big.setStyleSheet('')
        for lbl in (self.lbl_energy, self.lbl_flat):
            lbl.setText('—')
            lbl.setStyleSheet('')

    @staticmethod
    def _fmt_pct(pct: float) -> str:
        if pct == 0.0:
            return '0 %'
        return f'{pct:.3f} %' if pct >= 0.001 else f'{pct:.1e} %'

    def _show_result(self, r: dict):
        good, warn, bad = C['proc'], C['res'], C['cut']
        before, after = r['tones_before'], r['tones_after']
        if self.fs.tones_on:
            col = good if after <= max(1, before // 10) else warn if after <= before // 3 else bad
            self.lbl_tones_big.setText(f'{before} AI tones  →  {after} left')
        else:
            col = C['off']
            self.lbl_tones_big.setText(f'{before} AI tones (removal off)')
        self.lbl_tones_big.setStyleSheet(f'color: {col};')

        pct = r['energy_pct']
        col = good if pct < 1.0 else warn if pct < 5.0 else bad
        self.lbl_energy.setText(self._fmt_pct(pct))
        self.lbl_energy.setStyleSheet(f'color: {col};')
        flat = r['flatness']
        if flat is None:
            self.lbl_flat.setText('—')
            self.lbl_flat.setStyleSheet('')
        else:
            col, word = self._flatness_verdict(flat)
            self.lbl_flat.setText(word)
            self.lbl_flat.setStyleSheet(f'color: {col};')

    @staticmethod
    def _flatness_verdict(flat: float) -> tuple[str, str]:
        return ((C['proc'], 'mostly tones ✓') if flat < 0.2 else
                (C['res'], 'tones + some music') if flat < 0.5 else
                (C['cut'], 'mostly music — too much'))

    # ── History ───────────────────────────────────────────────

    def _load_history(self) -> list:
        data = read_json(self.HISTORY_FILE, [])
        if not isinstance(data, list):
            return []
        return [e for e in data if isinstance(e, dict) and isinstance(e.get('id'), int)
                and isinstance(e.get('settings'), dict) and isinstance(e.get('path'), str)]

    def _save_history(self):
        err = write_json(self.HISTORY_FILE, self._history)
        if err:
            self._set_status(f'Could not save history: {err}')

    @staticmethod
    def _notch_sig(notches) -> list:
        return [(float(n['freq']), float(n['width']), bool(n.get('on', True)))
                for n in (notches or []) if isinstance(n, dict) and 'freq' in n and 'width' in n]

    def _hist_key(self, path: str, d: dict) -> str:
        """Identity of a version: track + every setting + tone list (preset name ignored)."""
        core = {k: v for k, v in d.items() if k not in ('preset', 'notches')}
        return path + '|' + json.dumps([core, self._notch_sig(d.get('notches'))], sort_keys=True)

    def _hist_entry(self, entry_id) -> dict | None:
        return next((e for e in self._history if e['id'] == entry_id), None)

    @staticmethod
    def _preset_name(p) -> str:
        if p in PRESETS:
            return PRESETS[p]['label']
        if isinstance(p, str) and p.startswith('user:'):
            return p[5:]
        return 'Custom'

    def _fmt_value(self, k: str, v) -> str:
        if v is None:
            return 'whole track' if k == 'profile_region' else '—'
        if isinstance(v, bool):
            return 'on' if v else 'off'
        if k == 'profile_region':
            try:
                return f'{fmt_time(v[0], True)}–{fmt_time(v[1], True)}'
            except (TypeError, IndexError):
                return str(v)
        if k == 'resolution':
            return str(v) if v else 'auto'
        if isinstance(v, float):
            return f'{v:g}{self._FIELD_UNITS.get(k, "")}'
        return f'{v}{self._FIELD_UNITS.get(k, "")}'

    def _describe_changes(self, prev: dict | None, path: str, d: dict) -> tuple[str, list]:
        if prev is None:
            return 'First result', []
        changes = []
        if prev['path'] != path:
            changes.append(f'Track: {Path(path).name}')
        pd = prev['settings']
        if d.get('preset') != 'custom' and pd.get('preset') != d.get('preset'):
            changes.append(f'Preset {self._preset_name(pd.get("preset"))} → {self._preset_name(d.get("preset"))}')
        for k, label in self._FIELD_LABELS.items():
            a, b = pd.get(k), d.get(k)
            if a != b:
                changes.append(f'{label} {self._fmt_value(k, a)} → {self._fmt_value(k, b)}')
        na, nb = self._notch_sig(pd.get('notches')), self._notch_sig(d.get('notches'))
        if na != nb:
            changes.append(f'Tone list {len(na)} → {len(nb)}' if len(na) != len(nb) else 'Tone list edited')
        if not changes:
            changes = ['Same settings']
        short = ' · '.join(changes[:3]) + (f' · +{len(changes) - 3} more' if len(changes) > 3 else '')
        return short, changes

    def _record_history(self, r: dict):
        if not self.current_path:
            return
        d = r['settings'].to_dict()
        key = self._hist_key(self.current_path, d)
        for e in self._history:
            if self._hist_key(e['path'], e['settings']) == key:
                self._hist_current = e['id']
                self._refresh_history_list()
                return
        prev = self._hist_entry(self._hist_current) or (self._history[-1] if self._history else None)
        short, full = self._describe_changes(prev, self.current_path, d)
        entry = {
            'id':       self._hist_next_id,
            'time':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'path':     self.current_path,
            'manual':   self.current_path in self._manual_notches,
            'settings': d,
            'changes':  short,
            'changes_full': full,
            'metrics':  {'tones_on': bool(d.get('tones_on')), 'tones_before': int(r['tones_before']),
                         'tones_after': int(r['tones_after']), 'energy_pct': float(r['energy_pct']),
                         'flatness': None if r['flatness'] is None else float(r['flatness'])},
        }
        self._hist_next_id += 1
        self._history.append(entry)
        del self._history[:-self.HISTORY_MAX]
        self._hist_current = entry['id']
        self._save_history()
        self._refresh_history_list()

    def _visible_history(self) -> list:
        """Oldest → newest."""
        if self.chk_hist_track.isChecked() and self.current_path:
            return [e for e in self._history if e['path'] == self.current_path]
        return list(self._history)

    def _history_text(self, e: dict) -> str:
        t = e.get('time', '')
        today = datetime.now().strftime('%Y-%m-%d')
        when = t[11:] if t.startswith(today) else t[5:16]
        head = f'#{e["id"]}  {when}  · {self._preset_name(e["settings"].get("preset"))}'
        if not self.chk_hist_track.isChecked():
            head += f'  · {Path(e["path"]).name}'
        m = e.get('metrics') or {}
        if m.get('tones_on', True):
            res = f'{m.get("tones_before", "?")} → {m.get("tones_after", "?")} tones'
        else:
            res = 'tone removal off'
        if 'energy_pct' in m:
            res += f' · {self._fmt_pct(m["energy_pct"])} removed'
        if m.get('flatness') is not None:
            res += f' · {self._flatness_verdict(m["flatness"])[1]}'
        return f'{head}\n{e.get("changes", "")}\n{res}'

    def _refresh_history_list(self):
        selected = self._selected_history_id()
        lst = self.list_history
        lst.blockSignals(True)
        lst.clear()
        for e in reversed(self._visible_history()):
            marks = []
            if e['id'] == self._hist_current:
                marks.append('▶ current')
            if e['id'] == self._ref_id:
                marks.append('playing as A')
            text = self._history_text(e)
            if marks:
                text = f'[{" · ".join(marks)}]  {text}'
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, e['id'])
            item.setToolTip('\n'.join(e.get('changes_full') or [e.get('changes', '')]) +
                            f'\n\n{e["path"]}')
            if e['id'] == self._hist_current:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QColor(C['proc']))
            lst.addItem(item)
            if e['id'] == selected:
                item.setSelected(True)
        lst.blockSignals(False)
        self._update_history_buttons()

    def _selected_history_id(self):
        items = self.list_history.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _update_history_buttons(self):
        has = self._selected_history_id() is not None
        self.btn_hist_restore.setEnabled(has)
        self.btn_hist_ref.setEnabled(has and self._audio_orig is not None)
        self.btn_hist_ref_clear.setEnabled(self._ref_id is not None)
        self.btn_hist_clear.setEnabled(bool(self._visible_history()))

    def _on_hist_filter_toggled(self, on: bool):
        self._settings.setValue('history_track_only', bool(on))
        self._refresh_history_list()

    def _restore_selected_history(self):
        e = self._hist_entry(self._selected_history_id())
        if e is not None:
            self._restore_history(e)

    def _restore_history(self, entry: dict):
        """Bring back every setting and the tone list of a stored version."""
        s = FilterSettings.from_dict(entry['settings'])
        same = entry['path'] == self.current_path
        notches = s.notches
        s.notches = []
        if not same:
            s.profile_region = None
        self._hist_current = entry['id']
        self.fs = s
        self._settings_to_controls()
        self._set_notches(notches, manual=same and bool(entry.get('manual')), reprocess=False)
        if not (same and entry.get('manual')):
            self._manual_notches.pop(self.current_path, None)
        self._save_filter_settings()
        self._request_clean(detect=False, delay=0)
        self._refresh_history_list()
        self._set_status(f'Restored version #{entry["id"]}.')

    def _history_step(self, delta: int):
        entries = self._visible_history()
        if not entries:
            return
        ids = [e['id'] for e in entries]
        i = ids.index(self._hist_current) + delta if self._hist_current in ids else len(ids) - 1
        if 0 <= i < len(ids) and ids[i] != self._hist_current:
            self._restore_history(entries[i])

    def _clear_history(self):
        vis = {e['id'] for e in self._visible_history()}
        if not vis:
            return
        if QMessageBox.question(self, 'Clear history', f'Delete {len(vis)} history entries?') \
                != QMessageBox.StandardButton.Yes:
            return
        self._history = [e for e in self._history if e['id'] not in vis]
        if self._hist_current in vis:
            self._hist_current = None
        self._save_history()
        self._refresh_history_list()

    # ── Compare a stored version on A ────────────────────────

    def _compare_history_as_a(self):
        e = self._hist_entry(self._selected_history_id())
        if e is None or self._audio_orig is None:
            return
        s = FilterSettings.from_dict(e['settings'])
        if e['path'] != self.current_path:
            s.profile_region = None
        self._ref_generation += 1
        if self._ref_worker and self._ref_worker.isRunning():
            self._old_workers.append(self._ref_worker)
        self._ref_worker = RenderWorker(self.processor, s, self._ref_generation, e['id'])
        self._ref_worker.done.connect(self._on_ref_ready)
        self._ref_worker.error.connect(self._on_ref_error)
        self._ref_worker.start()
        self._set_status(f'Preparing version #{e["id"]} for A …')

    def _on_ref_ready(self, payload: dict):
        if payload['generation'] != self._ref_generation or self._audio_orig is None:
            return
        if payload['audio'].shape != self._audio_orig.shape:
            return
        self._audio_ref = payload['audio']
        self._ref_id = payload['tag']
        self._update_ab_button()
        self._refresh_history_list()
        self._set_status(f'A is now version #{self._ref_id}. Press Tab to switch between it and the current result.')

    def _on_ref_error(self, msg: str, generation: int):
        if generation == self._ref_generation:
            self._set_status(f'Could not prepare that version: {msg}')

    def _clear_ref(self):
        self._ref_generation += 1
        self._audio_ref = None
        self._ref_id = None
        self._update_ab_button()
        self._refresh_history_list()

    def _update_ab_button(self):
        label, obj = self._AB_LABELS[self._ab_mode]
        if self._ab_mode == 0 and self._ref_id is not None:
            label = f'A  Version #{self._ref_id}'
        self.btn_ab.setText(label)
        self.btn_ab.setObjectName(obj)
        self.style().unpolish(self.btn_ab)
        self.style().polish(self.btn_ab)

    # ── Audio callback ────────────────────────────────────────

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time, _status):
        orig  = self._audio_orig
        proc  = self._audio_proc
        ref   = self._audio_ref
        mode  = self._ab_mode
        solo  = self._solo_residual
        vol   = self._cb_volume

        if orig is None:
            outdata.fill(0.0)
            return

        frame = self._cb_frame
        end   = min(frame + frames, orig.shape[0])
        avail = end - frame

        if avail <= 0:
            outdata.fill(0.0)
            self._cb_playing = False
            return

        o = orig[frame:end]
        if solo and proc is not None and proc.shape == orig.shape:
            chunk = (o - proc[frame:end]) * (self._residual_boost * vol)
        elif mode == 1 and proc is not None and proc.shape == orig.shape:
            chunk = proc[frame:end] * vol
        elif ref is not None and ref.shape == orig.shape:
            chunk = ref[frame:end] * vol
        else:
            chunk = o * vol

        if avail < frames:
            outdata[:avail] = chunk
            outdata[avail:].fill(0.0)
            self._cb_playing = False
        else:
            outdata[:] = chunk
        self._cb_frame = end

    # ── Playback control ──────────────────────────────────────

    def _open_stream(self):
        if self._audio_orig is None:
            return
        self._close_stream()
        self._cb_playing = True
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self._audio_orig.shape[1],
                blocksize=1024,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            self._set_status(f'Audio device error: {e}')
            return
        self._pos_timer.start()
        self.is_playing = True
        self._show_play_state(True)

    def _show_play_state(self, playing: bool):
        self.btn_play.setIcon(self._icon_pause if playing else self._icon_play)
        self.btn_play.setText(' Pause' if playing else ' Play')

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _toggle_play(self):
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self._open_stream()

    def _pause(self):
        self._close_stream()
        self._pos_timer.stop()
        self.is_playing = False
        self._show_play_state(False)

    def _stop(self):
        self._close_stream()
        self._pos_timer.stop()
        self.is_playing = False
        self._cb_frame  = 0
        self._show_play_state(False)
        self.slider_pos.setValue(0)
        self._update_time_labels(0.0)
        self.tone_view.set_playhead(0.0)

    def _toggle_ab(self):
        """Toggle A ↔ B. If solo is active, Tab exits solo first. Stream keeps running."""
        if self._audio_proc is None:
            return
        if self._solo_residual:
            self._solo_residual = False
            self.btn_solo.setChecked(False)
        else:
            self._ab_mode = 1 - self._ab_mode
            self._update_ab_button()
        self._update_view_mode()

    def _toggle_solo_residual(self):
        if self._audio_proc is None:
            return
        self._solo_residual = not self._solo_residual
        self.btn_solo.setChecked(self._solo_residual)
        self._update_view_mode()

    def _update_view_mode(self):
        """Frame colour + tone view follow what you are hearing."""
        mode = 'res' if self._solo_residual else ('proc' if self._ab_mode == 1 else 'orig')
        self.view_frame.setStyleSheet(
            f'QFrame#view_frame {{ border: 2px solid {self._MODE_COLORS[mode]}; border-radius: 8px; }}')
        self.tone_view.set_view(mode)
        if self._audio_proc is not None:
            self._update_steps(4 if mode != 'orig' or self._ab_mode == 0 else 3)

    # ── Pan slider sync ───────────────────────────────────────

    def _sync_pan_slider(self, lo: float, hi: float):
        full_lo = np.log10(self.spectrum._zoom_xlim_full[0])
        full_hi = np.log10(self.spectrum._zoom_xlim_full[1])
        if lo <= 0 or hi <= lo:
            return
        scrollable = (full_hi - full_lo) - (np.log10(hi) - np.log10(lo))
        self.slider_pan.blockSignals(True)
        if scrollable < 1e-6:
            self.slider_pan.setValue(0)
            self.slider_pan.setEnabled(False)
        else:
            pos = (np.log10(lo) - full_lo) / scrollable
            self.slider_pan.setEnabled(True)
            self.slider_pan.setValue(int(np.clip(pos * 10000, 0, 10000)))
        self.slider_pan.blockSignals(False)

    def _on_pan_slider_moved(self, value: int):
        ax = self.spectrum.ax
        lo, hi = ax.get_xlim()
        if lo <= 0 or hi <= lo:
            return
        full_lo = np.log10(self.spectrum._zoom_xlim_full[0])
        full_hi = np.log10(self.spectrum._zoom_xlim_full[1])
        cur_span = np.log10(hi) - np.log10(lo)
        scrollable = (full_hi - full_lo) - cur_span
        if scrollable < 1e-6:
            return
        new_lo = full_lo + (value / 10000.0) * scrollable
        ax.set_xlim(10 ** new_lo, 10 ** (new_lo + cur_span))
        self.spectrum._auto_ylim()
        self.spectrum.draw_idle()

    # ── Seeking / timeline ────────────────────────────────────

    def _seek_rel(self, seconds: float):
        if self._audio_orig is None:
            return
        total = self._audio_orig.shape[0]
        self._seek_to(int(np.clip(self._cb_frame + seconds * self.sample_rate, 0, total - 1)))

    def _seek_to(self, frame: int):
        if self._audio_orig is None:
            return
        frame = int(np.clip(frame, 0, self._audio_orig.shape[0] - 1))
        was_playing = self.is_playing
        self._close_stream()
        self._pos_timer.stop()
        self.is_playing = False
        self._cb_frame  = frame
        self._update_slider_from_frame()
        self._update_time_labels(frame / self.sample_rate)
        self.tone_view.set_playhead(frame / self.sample_rate)
        if was_playing:
            self._open_stream()

    def _timeline_pressed(self):
        self._scrub_was_playing = self.is_playing
        if self.is_playing:
            self._close_stream()
            self._pos_timer.stop()
            self.is_playing = False
            self._show_play_state(False)

    def _timeline_released(self):
        if self._audio_orig is None:
            return
        frame = int(self._audio_orig.shape[0] * self.slider_pos.value() / 10000.0)
        self._cb_frame = frame
        self._update_time_labels(frame / self.sample_rate)
        self.tone_view.set_playhead(frame / self.sample_rate)
        if getattr(self, '_scrub_was_playing', False):
            self._open_stream()

    def _poll_position(self):
        if not self._cb_playing:
            self._stop()
            return
        if not self.slider_pos.isSliderDown():
            self._update_slider_from_frame()
        t = self._cb_frame / self.sample_rate
        self._update_time_labels(t)
        self.tone_view.set_playhead(t)

    def _update_slider_from_frame(self):
        if self._audio_orig is None or self.slider_pos.isSliderDown():
            return
        total = self._audio_orig.shape[0]
        if total > 0:
            self.slider_pos.setValue(int(self._cb_frame / total * 10000))

    def _update_time_labels(self, cur: float):
        self.lbl_time_cur.setText(fmt_time(cur))
        self._update_total_time()

    def _update_total_time(self):
        orig = self._audio_orig
        tot = (orig.shape[0] / self.sample_rate) if orig is not None else 0.0
        self.lbl_time_tot.setText(fmt_time(tot))

    # ── Export ────────────────────────────────────────────────

    def _save(self, audio: np.ndarray, suffix: str):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out, fmt, sub = output_path(self.current_path, self.OUTPUT_DIR, suffix, self.export_format)
        try:
            write_audio(out, audio, self.sample_rate, fmt, sub)
            self._set_status(f'Saved: {out}')
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', str(e))

    def _export_single(self):
        if self._audio_proc is not None:
            self._save(self._audio_proc, self.name_suffix)

    def _export_diff(self):
        if self._audio_proc is not None and self._audio_orig is not None:
            self._save(self._audio_orig - self._audio_proc, self.DIFF_SUFFIX)

    def center_on_screen(self):
        """Fit the window to the screen under the mouse and centre it there."""
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = max(min(1600, avail.width() - 40), min(self.minimumWidth(), avail.width()))
        h = max(min(1000, avail.height() - 60), min(self.minimumHeight(), avail.height()))
        if self.minimumWidth() > avail.width() or self.minimumHeight() > avail.height():
            self.setMinimumSize(min(self.minimumWidth(), avail.width()),
                                min(self.minimumHeight(), avail.height()))
        self.resize(min(w, avail.width()), min(h, avail.height()))
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(max(frame.left(), avail.left()), max(frame.top(), avail.top()))

    def _current_settings_label(self) -> str:
        p = self.fs.preset
        if p in PRESETS:
            return f'{PRESETS[p]["label"]} strength, as set in the panel'
        if p.startswith('user:'):
            return f'Your preset "{p[5:]}", as set in the panel'
        return 'Custom — your tweaked settings in the panel'

    def _export_batch(self):
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.cancelled = True
            self.btn_batch.setEnabled(False)
            self.btn_batch.setText('Cancelling …')
            return
        if self.file_list.count() == 0:
            return
        tracks = [(self.file_list.item(i).data(Qt.ItemDataRole.UserRole), self.file_list.item(i).text())
                  for i in range(self.file_list.count())]
        manual = {k: v for k, v in self._manual_notches.items() if k in {p for p, _ in tracks}}
        prefs = read_json_text(self._settings.value('batch_prefs', '', type=str), {})
        if not isinstance(prefs, dict):
            prefs = {}
        prefs.setdefault('detect', self.batch_detect)
        out_dir = Path(self._settings.value('batch_dir', str(self.OUTPUT_DIR / 'batch'), type=str))
        dlg = BatchDialog(self, tracks, self._current_settings_label(), list(self._user_preset_map),
                          sum(1 for n in self.fs.notches if n.get('on', True)), len(manual),
                          out_dir, self.export_format, self.name_suffix, prefs)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.paths():
            return
        self._settings.setValue('batch_prefs', json.dumps(dlg.prefs()))
        self._settings.setValue('batch_dir', str(dlg.out_dir()))
        self.batch_detect = dlg.detect()
        self._settings.setValue('batch_detect_v2', self.batch_detect)

        kind, key = dlg.source()
        s = self.fs.copy()
        if kind == 'strength':
            s.apply_preset(key)
        elif kind == 'user' and key in self._user_preset_map:
            loaded = FilterSettings.from_dict(
                {k: v for k, v in self._user_preset_map[key][1].items()
                 if k not in FilterSettings._TRANSIENT})
            loaded.notches = s.notches
            s = loaded

        self.btn_batch.setText('Cancel batch')
        self.btn_batch.setToolTip('Stop after the track that is being cleaned now')
        self._batch_worker = BatchWorker(
            dlg.paths(), s,
            {k: [dict(n) for n in v] for k, v in manual.items()} if dlg.keep_manual() else {},
            dlg.detect(), dlg.out_dir(), dlg.suffix(), self.DIFF_SUFFIX, dlg.fmt_key(),
            save_removed=dlg.save_removed())
        self._batch_worker.progress.connect(self._set_status)
        self._batch_worker.finished_msg.connect(self._on_batch_done)
        self._batch_worker.start()

    def _on_batch_done(self, msg: str):
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText('Save All…')
        self.btn_batch.setToolTip('Clean every track in the list — choose the settings in the next window')
        self._set_status('Batch complete.')
        QMessageBox.information(self, 'Batch', msg)

    # ── Misc ─────────────────────────────────────────────────

    def _disable_playback_ui(self):
        for w in (self.btn_play, self.btn_stop, self.btn_ab, self.btn_solo,
                  self.btn_export, self.btn_export_diff, self.btn_detect):
            w.setEnabled(False)

    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def dragEnterEvent(self, event):
        event.accept() if event.mimeData().hasUrls() else event.ignore()

    def dropEvent(self, event):
        self._add_paths(url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile())
        event.accept()

    def closeEvent(self, event):
        self._close_stream()
        self._save_filter_settings()
        for w in [self._load_worker, self._clean_worker, self._batch_worker, self._ref_worker,
                  *self._old_workers]:
            if w and w.isRunning():
                w.wait(5000)
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app_icon = QIcon(str(Path(__file__).resolve().parent / 'assets' / 'frequency_cleaner.png'))
    app.setWindowIcon(app_icon)
    win = MainWindow()
    win.setWindowIcon(app_icon)
    win.center_on_screen()
    win.show()
    sys.exit(app.exec())
