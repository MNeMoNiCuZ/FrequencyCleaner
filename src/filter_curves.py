"""Editable, log-frequency EQ shapes shared by the display and audio processor.

Widths are octaves; plateau is the fraction of that width with full gain.
These are spectral EQ shapes, applied with the app's overlap-add STFT engine.
"""

import numpy as np


CURVE_TYPES = [
    ('Bell', 'bell'), ('Low shelf', 'low_shelf'), ('High shelf', 'high_shelf'),
    ('Low cut / high-pass', 'low_cut'), ('High cut / low-pass', 'high_cut'),
    ('Notch', 'notch'), ('Band pass', 'band_pass'), ('Tilt', 'tilt'),
]
CURVE_COLORS = ['#a8d779', '#89dceb', '#fab387', '#f9e2af', '#f38ba8', '#a6e3a1']


def normalize_curve(data):
    def number(key, default, lo, hi):
        try:
            value = float(data.get(key, default))
            return float(np.clip(value, lo, hi)) if np.isfinite(value) else default
        except (ValueError, TypeError):
            return default

    kind = data.get('type', 'bell')
    return dict(type=kind if kind in dict(CURVE_TYPES).values() else 'bell',
                freq=number('freq', 1000., 20., 96000.),
                gain=number('gain', -6., -60., 24.),
                width=number('width', 1., .02, 10.),
                plateau=number('plateau', 0., 0., .95),
                scale=number('scale', 1., .1, 3.), on=bool(data.get('on', True)),
                dynamic=bool(data.get('dynamic', True)),
                threshold=number('threshold', -48., -120., 0.),
                ratio=number('ratio', 4., 1., 20.),
                attack=number('attack', 10., 0., 500.),
                release=number('release', 150., 0., 3000.),
                auto=bool(data.get('auto', False)))


def curve_gain_db(freqs, curve):
    """Return one curve's dB response, including independent overall scaling."""
    c = normalize_curve(curve)
    if not c['on']:
        return np.zeros_like(freqs, dtype=float)
    x = np.log2(np.maximum(freqs, 1e-8) / c['freq'])
    width = c['width'] * c['scale']
    half = width / 2.
    flat = half * c['plateau']
    t = np.clip((np.abs(x) - flat) / max(half - flat, 1e-6), 0., 1.)
    bump = .5 + .5 * np.cos(np.pi * t)
    # Shelves transition smoothly across the complete specified width.
    ramp = np.clip((x + half) / width, 0., 1.)
    ramp = .5 - .5 * np.cos(np.pi * ramp)
    gain = c['gain'] * c['scale']
    kind = c['type']
    if kind == 'low_shelf':
        return gain * (1. - ramp)
    if kind == 'high_shelf':
        return gain * ramp
    if kind == 'low_cut':
        return -abs(gain) * (1. - ramp)
    if kind == 'high_cut':
        return -abs(gain) * ramp
    if kind == 'band_pass':
        return -abs(gain) * (1. - bump)
    if kind == 'tilt':
        return gain * (2. * ramp - 1.)
    return (-abs(gain) if kind == 'notch' else gain) * bump


def combined_gain_db(freqs, curves):
    result = np.zeros_like(freqs, dtype=float)
    for curve in curves:
        result += curve_gain_db(freqs, curve)
    return np.clip(result, -120., 36.)


def handle_gain(curve):
    """Handle ordinate, chosen so its gain remains editable on every shape."""
    gain = curve['gain'] * curve['scale']
    if curve['type'] in ('low_cut', 'high_cut', 'notch', 'band_pass'):
        gain = -abs(gain)
    return gain / 2. if curve['type'] in ('low_shelf', 'high_shelf', 'low_cut', 'high_cut') else gain
