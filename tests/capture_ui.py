"""Capture the real Qt UI using a local track, with isolated settings and no playback.

Usage: python tests/capture_ui.py --output artifacts/ui/01-analyzer.png
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time
from io import BytesIO
from PIL import Image
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QSettings, Qt, QPoint, QBuffer, QIODevice, QEventLoop, QTimer, QObject, QEvent
from PyQt6.QtGui import QFontDatabase, QPainter
import main


def capture(output, width=1600, height=1000, inspector=None, curves=True, shape_menu=False, scrub=False, auto_curves=False, animate=False, select_all=False, benchmark=False, collapsed=False, settings_menu=False, settings_tab='General', move_editor=False, view='spectrum'):
    app = main.QApplication.instance() or main.QApplication([])
    app.setStyle('Fusion')
    for name in ['segoeui.ttf', 'segoeuib.ttf', 'segoeuil.ttf']:
        font = Path('C:/Windows/Fonts') / name
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        settings = QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat)
        settings.setValue('use_gpu', False)
        patches = [patch.object(main, 'QSettings', lambda *a: settings)]
        for key, path in [('INPUT_DIR',root/'input'), ('OUTPUT_DIR',root/'output'),
                          ('PRESET_DIR',root/'presets'), ('HISTORY_FILE',root/'history.json'),
                          ('VERSION_DIR',root/'versions')]:
            patches.append(patch.object(main.MainWindow, key, path))
        for p in patches:
            p.start()
        win = main.MainWindow()
        win.resize(width, height)
        win.show()
        app.processEvents()
        if collapsed:
            win.btn_tracks.click()
            app.processEvents()
        win._request_clean = lambda **kw: None
        files = sorted((project/'input').glob('*.mp3'))
        win._add_paths(files)
        win.file_list.blockSignals(True)
        win.file_list.setCurrentRow(1 if len(files) > 1 else 0)
        selected = files[1] if len(files) > 1 else files[0]
        audio, sr = sf.read(selected, dtype='float32', always_2d=True)
        win._audio_orig, win.sample_rate, win.current_path = audio, sr, str(selected)
        win._update_config_buttons()
        win._slot = 0
        win._audio_play = None
        win.spectrum.set_nyquist(sr/2)
        win.lbl_track.setText(f'{main.fmt_time(len(audio)/sr)} \u00b7 {sr:,} Hz \u00b7 stereo')
        if hasattr(win, 'track_title'):
            win._set_track_path(selected)
        history = json.loads((project/'history.json').read_text(encoding='utf-8')) if (project/'history.json').exists() else []
        for entry in reversed(history):
            if Path(entry.get('path', '')).name == selected.name:
                win.fs.notches = entry['settings'].get('notches', [])
                break
        win.processor.load(audio, sr)
        win.fs.notches = win.processor.detect_tones(win.fs.threshold_db, win.fs.min_hz, s=win.fs)
        win.btn_detect.setEnabled(True)
        print(f'Real detection on {selected.name}: {len(win.fs.notches)} outliers')
        if auto_curves:
            win.fs.eq_curves = win.processor.suggest_eq_curves(win.fs)
            win.spectrum.selected_curve = 0 if win.fs.eq_curves else -1
            win._refresh_filter_response()
            win._refresh_curve_controls()
        elif curves:
            win._add_curve(11800, 'notch')
            win.curve_controls['gain'].setValue(-18)
            win.curve_controls['width'].setValue(.35)
            win.dynamic_controls['threshold'].setValue(-62)
        win.fs.tones_on = False
        trace = {}
        processed = win.processor.apply(win.fs, trace_out=trace)
        win._audio_proc = processed
        f, _ = win.processor.spectrum_original()
        win.closeup.set_original(f, win.processor.tone_prominence(win.fs), 4000, sr/2)
        win.tone_view.set_original(win.processor.tone_view(win.fs), 4000, sr/2)
        result = win.processor.analyse(processed, win.fs)
        result.update(audio=processed, eq_trace=trace, tones_on=False)
        win._set_b(result, None)
        win._set_busy(None)
        print(f'Rendered {len(win.fs.eq_curves)} dynamic bands; {trace.get("activity", np.zeros((0,0))).shape} envelope samples')
        win._refresh_notch_views()
        win.btn_play.setEnabled(True)
        win.btn_stop.setEnabled(True)
        win.btn_ab.setEnabled(True)
        win._update_ab_button()
        win.is_playing = True
        for seconds in np.linspace(91., 97.74, 45):
            win._cb_frame = int(min(seconds, len(audio)/sr-1)*sr)
            win._update_live_spectrum()
        win.is_playing = False
        win._update_live_spectrum()
        win._update_time_labels(win._cb_frame/sr)
        win._update_slider_from_frame()
        if inspector and hasattr(win, '_show_inspector'):
            win._show_inspector(inspector)
        if select_all:
            win.btn_select_curves.click()
        app.processEvents()
        win.spectrum.draw()
        app.processEvents()
        # Measure actual FFT + current/peak painting, with the axes cached.
        timings = []
        profiler = None
        if os.environ.get('FC_PROFILE'):
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        win.is_playing = True
        for i in range(20):
            win._cb_frame += 1764
            started = time.perf_counter()
            win._update_live_spectrum()
            app.processEvents()  # Include the actual Qt paint, not just scheduling it.
            timings.append((time.perf_counter()-started)*1000)
        if profiler:
            profiler.disable()
            profiler.dump_stats('artifacts/ui/frame-profile.bin')
        win.is_playing = False
        win._update_live_spectrum()
        win._update_time_labels(win._cb_frame/sr)
        win._update_slider_from_frame()
        print(f'Analyzer FFT + repaint: {np.median(timings):.1f} ms median; {max(timings):.1f} ms max')
        if benchmark:
            class PaintCounter(QObject):
                count = 0
                def eventFilter(self, obj, event):
                    if event.type() == QEvent.Type.Paint:
                        self.count += 1
                    return False
            counter = PaintCounter()
            win.spectrum.installEventFilter(counter)
            loop = QEventLoop()
            timer = QTimer()
            timer.setTimerType(Qt.TimerType.PreciseTimer)
            timer.setInterval(6)
            start_frame = win._cb_frame
            started = time.perf_counter()
            ticks = []
            def tick():
                elapsed = time.perf_counter()-started
                ticks.append(elapsed)
                # Emulate an audio callback delivering 1024 samples at a time.
                win._cb_frame = start_frame + int(elapsed*sr//1024)*1024
                win._cb_playing = True
                win._poll_position()
            win.is_playing = True
            timer.timeout.connect(tick)
            timer.start()
            QTimer.singleShot(2000, loop.quit)
            loop.exec()
            timer.stop()
            elapsed = time.perf_counter()-started
            print(f'Qt 6 ms timer: {len(ticks)/elapsed:.1f} updates/s; {counter.count/elapsed:.1f} painted FPS')
            # Measure real grouped drag, including controls and graph repaint.
            from matplotlib.backend_bases import MouseEvent, MouseButton
            canvas = win.spectrum
            win.btn_select_curves.click()
            initial = dict(canvas.eq_curves[canvas.selected_curve])
            def gesture(name, freq, gain):
                x, y = canvas.eq_ax.transData.transform((freq, gain))
                return MouseEvent(name, canvas, x, y, button=MouseButton.LEFT)
            canvas._on_press(gesture('button_press_event', initial['freq'], initial['gain']))
            drag_times = []
            for delta in np.linspace(0, .15, 30):
                t0 = time.perf_counter()
                canvas._on_motion(gesture('motion_notify_event', initial['freq']*2**delta, initial['gain']+delta*4))
                win._update_live_spectrum()
                app.processEvents()
                drag_times.append((time.perf_counter()-t0)*1000)
            canvas._on_release(gesture('button_release_event', initial['freq']*2**.15, initial['gain']+.6))
            print(f'Grouped drag + controls + FFT + paint: {np.median(drag_times):.1f} ms median; {max(drag_times):.1f} ms max')
            # Restore the already rendered curves and their recorded envelopes.
            win.fs.eq_curves = trace['curves']
            win._refresh_filter_response()
            win._refresh_curve_controls()
            win.is_playing = False
            win._update_live_spectrum()
            win._update_time_labels(win._cb_frame/sr)
            win._update_slider_from_frame()
            app.processEvents()
        if scrub:
            win._timeline_pressed()
            win.slider_pos.setValue(6500)
            win._timeline_moved(6500)
            win._scrub_was_playing = False
            win._timeline_released()
            app.processEvents()
        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if view == 'outliers':
            win.btn_view_close.click()
        elif view == 'spectrogram':
            win.btn_view_lines.click()
        win.tone_view.set_playhead(win._cb_frame/sr)
        app.processEvents()
        shot = win.grab()
        if settings_menu:
            dialog = main.SettingsDialog(win, win.name_suffix, win.export_format, win.detect_on_load,
                                         main.GPU.enabled, win._hist_lines, win._hist_clear)
            dialog.tabs.setCurrentIndex(next(i for i in range(dialog.tabs.count()) if dialog.tabs.tabText(i) == settings_tab))
            dialog.move(win.mapToGlobal(QPoint((win.width()-dialog.width())//2, (win.height()-dialog.height())//2)))
            dialog.show()
            app.processEvents()
            painter = QPainter(shot)
            painter.drawPixmap(win.mapFromGlobal(dialog.pos()), dialog.grab())
            painter.end()
            dialog.accept()
        if move_editor:
            win._drag_curve_editor(win.view_frame.mapToGlobal(QPoint(win.view_frame.width()-win.curve_card.width()-16, 110)))
            app.processEvents()
            shot = win.grab()
        if shape_menu:
            # Menus are separate native Qt windows. Capture their real widget
            # and composite at its actual global position over the app capture.
            button = win.btn_curve_add
            menu = button.menu()
            anchor = button.mapToGlobal(QPoint(0, button.height()+4))
            menu.popup(anchor)
            app.processEvents()
            # The offscreen screen is smaller than the requested wide window.
            # Keep its actual popup on the captured app's own viewport.
            menu.move(anchor)
            app.processEvents()
            painter = QPainter(shot)
            painter.drawPixmap(win.mapFromGlobal(menu.pos()), menu.grab())
            painter.end()
            menu.hide()
        shot.save(str(target))
        print(target)
        print(f'Canvas: {win.spectrum.width()} x {win.spectrum.height()}; tracks: {win.file_list.count()}; '
              f'auto notches: {len(win.fs.notches)}; elision: {win.file_list.textElideMode().name}')
        if animate:
            frames = []
            win.spectrum.reset_peak()
            win.is_playing = True
            win._show_play_state(True)
            for seconds in np.linspace(91., 103., 16):
                win._cb_frame = int(seconds*sr)
                win._update_live_spectrum()
                win.tone_view.set_playhead(seconds)
                win._update_time_labels(seconds)
                win._update_slider_from_frame()
                app.processEvents()
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                win.grab().save(buffer, 'PNG')
                frame = Image.open(BytesIO(bytes(buffer.data()))).convert('RGB')
                frame.thumbnail((1200, 750), Image.Resampling.LANCZOS)
                frames.append(frame)
            gif = target.with_suffix('.gif')
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=160, loop=0)
            print(f'Animated UI preview: {gif}')
            win.is_playing = False
        print(f'Edit-mode selector: visible={win.combo_edit_mode.isVisible()}; geometry={win.combo_edit_mode.geometry().getRect()}')
        win.close()
        settings.sync()
        for p in reversed(patches):
            p.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1000)
    parser.add_argument('--inspector')
    parser.add_argument('--no-curves', action='store_true')
    parser.add_argument('--shape-menu', action='store_true')
    parser.add_argument('--scrub', action='store_true')
    parser.add_argument('--auto-curves', action='store_true')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--select-all', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--collapsed', action='store_true')
    parser.add_argument('--settings-menu', action='store_true')
    parser.add_argument('--settings-tab', default='General')
    parser.add_argument('--move-editor', action='store_true')
    parser.add_argument('--view', choices=('spectrum', 'outliers', 'spectrogram'), default='spectrum')
    args = parser.parse_args()
    capture(args.output, args.width, args.height, args.inspector, not args.no_curves, args.shape_menu, args.scrub, args.auto_curves, args.animate, args.select_all, args.benchmark, args.collapsed, args.settings_menu, args.settings_tab, args.move_editor, args.view)
