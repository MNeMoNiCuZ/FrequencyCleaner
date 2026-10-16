"""Live analyzer and EQ editing layered over the legacy range canvas."""

import time
import numpy as np
from scipy import ndimage, fft as scipy_fft
from scipy.interpolate import PchipInterpolator
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor
from spectrum_painter import SpectrumPainterMixin

from filter_curves import CURVE_COLORS, curve_gain_db, handle_gain, normalize_curve


class LiveSpectrumMixin(SpectrumPainterMixin):
    def __init__(self, *args, **kwargs):
        self._live_ready = False
        super().__init__(*args, **kwargs)
        self.edit_mode = 'curves'
        self.eq_curves = []
        self.selected_curve = -1
        self.selected_curves = set()
        self._curve_drag = None
        self._peak_db = None
        self._current_db = None
        self._last_frame = None
        self._last_tick = None
        self._fft_cache = None
        self._fft_setup = None
        self._playing = False
        self.peak_decay_db = .5
        self.show_auto_cuts = False
        self.smoothing_octaves = 1 / 12
        self.gain_range = 24.
        self._auto_response = None
        self._instant_response = None
        self._response_artist = None
        self._eq_revision = 0
        self._response_freq = np.geomspace(20., self.max_freq, 1600)
        self._response_db = np.zeros(1600)
        self.eq_ax = self.ax.twinx()
        self.fig.set_layout_engine(None)
        self.fig.subplots_adjust(left=.045, right=.957, bottom=.065, top=.94)
        self.fig.set_facecolor('#141218')
        self.ax.set_facecolor('#141218')
        self.eq_ax.set_ylim(-self.gain_range, self.gain_range)
        self.eq_ax.set_ylabel('')
        self.eq_ax.yaxis.set_major_locator(FixedLocator([-24, -18, -12, -6, 0, 6, 12, 18, 24]))
        self.eq_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+g}' if v else '0'))
        self.eq_ax.tick_params(colors='#b6a266', labelsize=8, length=0, pad=9)
        for spine in self.eq_ax.spines.values():
            spine.set_visible(False)
        self.eq_ax.patch.set_visible(False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_ylabel('')
        self.ax.set_xlabel('')
        self.ax.set_ylim(-102, -6)
        self.ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]))
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'))
        self.ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        self.ax.xaxis.set_minor_formatter(NullFormatter())
        self.ax.yaxis.set_major_locator(FixedLocator([-96, -84, -72, -60, -48, -36, -24, -12]))
        self.ax.tick_params(axis='both', which='both', colors='#77727e', labelsize=8, length=0, pad=10)
        self.ax.grid(False, which='both')
        self.ax.grid(True, which='major', color='#353039', lw=.55, alpha=.8)
        self.ax.grid(True, axis='x', which='minor', color='#2b272f', lw=.45, alpha=.6)
        self.ax.set_title('')
        self.ax.set_title('', loc='left')
        self.ax.annotate('CURRENT', (0, 1), xycoords='axes fraction', xytext=(0, 17),
                         textcoords='offset points', color='#e9dff5', fontsize=8)
        self._peak_legend = self.ax.annotate('PEAK HOLD', (0, 1), xycoords='axes fraction', xytext=(80, 17),
                         textcoords='offset points', color='#80529e', fontsize=8)
        self.ax.annotate('EQ dB', (1, 1), xycoords='axes fraction', xytext=(10, 17),
                         textcoords='offset points', color='#a7986a', fontsize=8)
        self.ax.annotate('dBFS', (0, 1), xycoords='axes fraction', xytext=(-34, 17),
                         textcoords='offset points', color='#69636f', fontsize=7)
        self._current_line, = self.ax.plot([], [], color='#efe0ff', lw=1.45, zorder=3, animated=True,
                                           solid_joinstyle='round', solid_capstyle='round')
        self._peak_line, = self.ax.plot([], [], color='#80529e', lw=1.05, zorder=2, animated=True,
                                        solid_joinstyle='round')
        self._frame_label = self.ax.annotate('Load a track', (1, 1), xycoords='axes fraction',
                                            xytext=(-8, 17), textcoords='offset points',
                                            ha='right', color='#8a8191', fontsize=8)
        self._frame_label.set_animated(True)
        self.ax.set_autoscale_on(False)
        self.eq_ax.set_autoscale_on(False)
        self.mpl_connect('draw_event', self._on_live_draw)
        self.mpl_connect('resize_event', self._on_analyzer_resize)
        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(33)
        self._decay_timer.timeout.connect(self._decay_paused_peak)
        self._live_ready = True
        self._spectrum_accent = QColor('#68b7e8')
        self._spectrum_background = QColor('#181818')
        self._redraw_overlay()

    def set_colors(self, background, accent):
        self.fig.set_facecolor(background)
        self.ax.set_facecolor(background)
        peak = QColor(accent).darker(220)
        self._spectrum_accent = QColor(accent)
        self._spectrum_background = QColor(background)
        self._peak_line.set_color(peak.name())
        self._peak_legend.set_color(accent)
        self._current_line.set_color('#eeeeee')
        self.ax.grid(True, which='major', color='#363636', lw=.55, alpha=.8)
        self.ax.grid(True, axis='x', which='minor', color='#2b2b2b', lw=.45, alpha=.6)
        for text in self.ax.texts:
            text.set_fontsize(9)
            if text is not self._peak_legend:
                text.set_color('#bcbcbc')
        self.ax.tick_params(axis='both', which='both', colors='#999999', labelsize=9)
        self.draw_idle()

    def _on_analyzer_resize(self, event):
        # Pixel-sized gutters keep both scales readable in a narrow inspector layout.
        width, height = max(self.width(), 320), max(self.height(), 250)
        self.fig.subplots_adjust(left=58/width, right=1-56/width,
                                 bottom=48/height, top=1-46/height)

    def _display_handle_gain(self, curve):
        return float(np.clip(handle_gain(curve), -self.gain_range*.94, self.gain_range*.94))

    def _on_live_draw(self, event):
        # The Agg image contains axes and grid only. Qt paints every animated
        # layer afterward, so any axes redraw also preserves the spectrum.
        rgba = self.buffer_rgba()
        image = QImage(rgba, rgba.shape[1], rgba.shape[0], QImage.Format.Format_RGBA8888)
        image.setDevicePixelRatio(self.device_pixel_ratio)
        self._axes_pixmap = QPixmap.fromImage(image)

    def _paint_live(self, blit=True):
        if self._live_ready:
            self.update()

    def _decay_paused_peak(self):
        if self._playing or self._peak_db is None or not self.isVisible():
            return
        now = time.perf_counter()
        elapsed = max(0., now - (self._last_tick or now))
        self._last_tick = now
        if not self.peak_decay_db or np.all(self._peak_db <= self._current_db):
            return
        self._peak_db = np.maximum(self._current_db, self._peak_db - elapsed*self.peak_decay_db)
        self._peak_line.set_data(self._current_line.get_xdata(), self._peak_db)
        self._paint_live()

    def prepare_seek(self):
        # Preserve the long-lived envelope; the next frame snaps to the new
        # position instead of blending across unrelated parts of the track.
        self._current_db = None
        self._fft_cache = None

    def plot_original(self, freqs, pdb):
        # Whole-track statistics remain available in the diagnostic views.
        self.reset_peak()
        self._current_line.set_data([], [])

    def plot_processed(self, freqs, proc_db):
        pass

    def clear_processed(self, redraw=True):
        pass

    def _auto_ylim(self):
        # A stable dBFS scale prevents frame-to-frame vertical jumping.
        if self._live_ready:
            self.ax.set_ylim(-102, -6)

    def set_gain_range(self, value):
        self.gain_range = float(value)
        self.eq_ax.set_ylim(-self.gain_range, self.gain_range)
        self.eq_ax.yaxis.set_major_locator(FixedLocator(np.linspace(-self.gain_range, self.gain_range, 9)))
        self._redraw_overlay()
        self.draw_idle()

    def set_smoothing(self, octaves):
        self.smoothing_octaves = float(octaves)
        self.reset_peak()

    def set_auto_cuts_visible(self, visible):
        self.show_auto_cuts = bool(visible)
        self._redraw_overlay()

    def set_instant_response(self, freqs, db):
        self._instant_response = (freqs, db) if freqs is not None else None
        if self._response_artist is not None:
            if self._instant_response is not None:
                self._response_artist.set_data(freqs, db)
            else:
                # A new band can be edited before rendering finishes. Show its
                # maximum region, but never label it as an applied response.
                fixed = [c for c in self.eq_curves if not c.get('dynamic', True)]
                from filter_curves import combined_gain_db
                values = combined_gain_db(self._response_freq, fixed) if self.edit_mode == 'curves' else self._response_db
                self._response_artist.set_data(self._response_freq, values)

    def set_filter_response(self, settings, freqs, response):
        self.edit_mode = settings.edit_mode
        if self.eq_curves != settings.eq_curves:
            self._instant_response = None
        self.eq_curves = [dict(c) for c in settings.eq_curves]
        self._auto_response = getattr(settings, '_display_auto_response', None)
        self._response_freq, self._response_db = freqs, response
        if self.selected_curve >= len(self.eq_curves):
            self.selected_curve = len(self.eq_curves) - 1
        self.shelf_on = settings.shelf_on and settings.edit_mode == 'range'
        self.fade_start_freq, self.cutoff_freq = settings.fade_start, settings.cutoff
        self._redraw_overlay()

    def select_curve(self, index, additive=False):
        if index < 0:
            self.selected_curves.clear()
            self.selected_curve = -1
        elif additive:
            if index in self.selected_curves:
                self.selected_curves.remove(index)
            else:
                self.selected_curves.add(index)
            self.selected_curve = index if index in self.selected_curves else next(iter(self.selected_curves), -1)
        else:
            self.selected_curve = index
            self.selected_curves = {index}
        self._redraw_overlay()

    def select_all_curves(self):
        self.selected_curves = set(range(len(self.eq_curves)))
        self.selected_curve = min(self.selected_curves, default=-1)
        self._redraw_overlay()
        self.curve_selected.emit(self.selected_curve)

    def reset_peak(self):
        self._peak_db = None
        self._current_db = None
        self._last_frame = None
        self._last_tick = None
        self._fft_cache = None
        if self._live_ready:
            self._peak_line.set_data([], [])
            self._paint_live()

    def update_frame(self, audio, sr, frame, playing=False, residual=None, boost=1.):
        if audio is None or not self._live_ready:
            return
        n = min(32768, 2 ** int(np.floor(np.log2(max(len(audio), 256)))))
        frame = int(np.clip(frame, 0, len(audio) - 1))
        cache_key = (id(audio), sr, frame, id(residual), boost, self.smoothing_octaves)
        if self._fft_cache is not None and self._fft_cache[0] == cache_key:
            grid, values = self._fft_cache[1:]
        else:
            if self._fft_setup is None or self._fft_setup[0] != (n, sr):
                f = scipy_fft.rfftfreq(n, 1./sr)
                grid = np.geomspace(20., sr/2., 1100)
                edges = np.geomspace(20., sr/2., 1101)
                ids = np.clip(np.searchsorted(edges, f)-1, 0, len(grid)-1)
                valid = f >= 20.
                bins, starts = np.unique(ids[valid], return_index=True)
                self._fft_setup = ((n, sr), np.hanning(n).astype(np.float32),
                                   f, grid, bins, starts, valid, np.log2(np.maximum(f, 1.e-8)), np.log2(grid))
            _, window, f, grid, bins, starts, valid, log_f, log_grid = self._fft_setup
            # Centre the FFT at the playhead, padding only outside track boundaries.
            start, stop = frame - n // 2, frame + n // 2
            block = np.zeros((n, audio.shape[1]), dtype=np.float32)
            lo, hi = max(0, start), min(len(audio), stop)
            block[lo - start:hi - start] = audio[lo:hi]
            if residual is not None:
                block[lo - start:hi - start] -= residual[lo:hi]
                block *= boost
            fft = scipy_fft.rfft(block * window[:, None], axis=0)
            # Power-average channels so opposite-phase stereo does not disappear.
            amp = np.sqrt(np.mean(np.abs(fft) ** 2, axis=1)) * (2. / window.sum())
            db = 20. * np.log10(np.maximum(amp, 1e-8))
            pooled = np.full(len(grid), -160.)
            pooled[bins] = np.maximum.reduceat(db[valid], starts)
            # Shape-preserving cubic interpolation removes the straight segments
            # between sparse bass bins without inventing overshoots or new peaks.
            bass = (f > 0) & (f <= 650.)
            values = np.interp(log_grid, log_f[1:], db[1:])
            low = grid < 600.
            values[low] = PchipInterpolator(log_f[bass], db[bass], extrapolate=True)(log_grid[low])
            values = np.maximum(values, pooled)
            # Peak-preserving fractional-octave grouping, followed by smooth power
            # interpolation. This affects the analyzer only, never audio processing.
            sigma = max(self.smoothing_octaves / (np.log2(grid[-1] / grid[0]) / len(grid)) / 3., .65)
            power = ndimage.maximum_filter1d(10. ** (values / 10.), size=max(3, int(sigma * 3) | 1))
            values = 10. * np.log10(np.maximum(ndimage.gaussian_filter1d(power, sigma), 1e-16))
            self._fft_cache = (cache_key, grid, values)
        now = time.perf_counter()
        if playing and self._current_db is not None and len(self._current_db) == len(values):
            elapsed = min(now - (self._last_tick or now), .25)
            # Fast attack, gently falling release; independent of timer rate.
            seconds = max(elapsed, .0001)
            alpha = 1. - np.exp(-seconds / np.where(values > self._current_db, .018, .12))
            self._current_db += alpha * (values - self._current_db)
        else:
            self._current_db = values.copy()
        elapsed = max(0., now - (self._last_tick or now))
        self._peak_db = values.copy() if self._peak_db is None else np.maximum(
            self._peak_db - self.peak_decay_db*elapsed, np.maximum(values, self._current_db))
        self._peak_line.set_data(grid, self._peak_db)
        self._current_line.set_data(grid, self._current_db)
        self._playing = playing
        if not self._decay_timer.isActive():
            self._decay_timer.start()
        state = 'Playing' if playing else 'Current frame'
        self._frame_label.set_text(f'{state}  {int(frame / sr // 60)}:{frame / sr % 60:04.1f}')
        self._last_frame, self._last_tick = frame, now
        self._paint_live()

    def _redraw_overlay(self):
        if not self._live_ready:
            return super()._redraw_overlay()
        self.selected_curves = {i for i in self.selected_curves if i < len(self.eq_curves)}
        if self.selected_curve >= 0:
            self.selected_curves.add(self.selected_curve)
        self._native_bands = []
        self._eq_revision += 1
        if self.edit_mode == 'curves':
            for i, curve in enumerate(self.eq_curves):
                color = CURVE_COLORS[i % len(CURVE_COLORS)] if curve['on'] else '#6c7086'
                self._native_bands.append((i, curve, curve_gain_db(self._response_freq, curve), color))
        if self._response_artist is None:
            for artist in self._overlay:
                artist.remove()
            zero = self.eq_ax.axhline(0, animated=True)
            self._response_artist, = self.eq_ax.plot([], [], animated=True)
            self._overlay = [zero, self._response_artist]
        self.set_instant_response(*(self._instant_response or (None, None)))
        self._paint_live()

    def _normalize_event(self, event):
        if event.inaxes in (self.ax, self.eq_ax):
            event.inaxes = self.ax
            event.xdata, event.ydata = self.ax.transData.inverted().transform((event.x, event.y))

    def _curve_hit(self, event):
        best, distance = None, 16.
        for i, c in enumerate(self.eq_curves):
            point = self.eq_ax.transData.transform((c['freq'], self._display_handle_gain(c)))
            d = np.hypot(point[0] - event.x, point[1] - event.y)
            if d < distance:
                best, distance = i, d
        return best

    def _on_press(self, event):
        self._normalize_event(event)
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.edit_mode == 'range' or event.button == MouseButton.MIDDLE:
            return super()._on_press(event)
        if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
            return
        hit = self._curve_hit(event)
        if event.dblclick and event.button == MouseButton.LEFT and hit is None:
            self.add_curve_requested.emit(float(event.xdata))
            return
        if hit is None:
            self.select_curve(-1)
            self.curve_selected.emit(-1)
            self._redraw_overlay()
            return
        if event.key in ('control', 'ctrl', 'shift'):
            self.select_curve(hit, additive=True)
            self.curve_selected.emit(self.selected_curve)
            return
        if hit not in self.selected_curves:
            self.select_curve(hit)
        self.selected_curve = hit
        self.curve_selected.emit(hit)
        self._curve_drag = (event.button, event.x, event.y,
                            {i: dict(self.eq_curves[i]) for i in self.selected_curves})
        self._redraw_overlay()

    def _on_motion(self, event):
        self._normalize_event(event)
        if self._curve_drag is None:
            return super()._on_motion(event)
        if event.inaxes != self.ax or self.selected_curve < 0:
            return
        button, x0, y0, originals = self._curve_drag
        initial = originals[self.selected_curve]
        frequency_ratio = self._freq_at(event.xdata) / initial['freq']
        frequency_ratio = np.clip(frequency_ratio, 20/min(c['freq'] for c in originals.values()),
                                  self.max_freq/max(c['freq'] for c in originals.values()))
        y = self.eq_ax.transData.inverted().transform((event.x, event.y))[1]
        delta_gain = y - self._display_handle_gain(initial)
        for index, initial in originals.items():
            c = {**initial, 'auto': False}
            if button == MouseButton.RIGHT:
                c['width'] *= 2. ** ((event.x - x0) / 120.)
                c['plateau'] += (event.y - y0) / 200.
            else:
                c['freq'] *= frequency_ratio
                multiplier = 2. if c['type'] in ('low_shelf', 'high_shelf', 'low_cut', 'high_cut') else 1.
                c['gain'] += delta_gain * multiplier / c['scale']
                if c['type'] in ('notch', 'low_cut', 'high_cut', 'band_pass'):
                    c['gain'] = -abs(c['gain'])
            self.eq_curves[index] = normalize_curve(c)
        self._preview_response()

    def _preview_response(self):
        # Rebuild only the editable contribution while a gesture is underway.
        self.curves_preview.emit([dict(c) for c in self.eq_curves])

    def _on_release(self, event):
        if self._curve_drag is not None:
            self._curve_drag = None
            self.curves_edited.emit([dict(c) for c in self.eq_curves])
            return
        return super()._on_release(event)

    def _on_scroll(self, event):
        self._normalize_event(event)
        if event.inaxes != self.ax:
            return
        if self.edit_mode == 'curves' and event.key != 'control':
            hit = self._curve_hit(event)
            index = hit if hit is not None else self.selected_curve
            if 0 <= index < len(self.eq_curves):
                if index not in self.selected_curves:
                    self.select_curve(index)
                self.selected_curve = index
                factor = 2. ** (event.step / 6.)
                for i in self.selected_curves:
                    c = {**self.eq_curves[i], 'auto': False}
                    c['scale' if event.key == 'alt' else 'width'] *= factor
                    self.eq_curves[i] = normalize_curve(c)
                self.curve_selected.emit(index)
                self.curves_edited.emit([dict(c) for c in self.eq_curves])
                return
        return super()._on_scroll(event)
