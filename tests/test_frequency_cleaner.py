"""DSP and desktop interaction regressions. Uses isolated settings and no audio device."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from matplotlib.backend_bases import MouseEvent, MouseButton
from PyQt6.QtCore import QSettings, Qt, QPoint
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtTest import QTest

import main
from filter_curves import CURVE_TYPES, normalize_curve, curve_gain_db, combined_gain_db


class DSPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.GPU.enabled = False
        cls.sr = 48000
        cls.t = np.arange(cls.sr * 2) / cls.sr
        # Opposite-phase channels also exercise independent channel processing.
        x = .12 * np.sin(2 * np.pi * 1000 * cls.t) + .08 * np.sin(2 * np.pi * 8000 * cls.t)
        cls.audio = np.column_stack((x, -x)).astype(np.float32)
        cls.processor = main.FilterProcessor()
        cls.processor.load(cls.audio, cls.sr)

    def test_all_shapes_and_plateaus(self):
        f = np.geomspace(20, 24000, 4096)
        for _, kind in CURVE_TYPES:
            c = normalize_curve(dict(type=kind, freq=1000, gain=-12, width=2, plateau=.5))
            db = curve_gain_db(f, c)
            self.assertTrue(np.isfinite(db).all(), kind)
            self.assertGreater(np.ptp(db), 10, kind)
            np.testing.assert_array_equal(curve_gain_db(f, {**c, 'on': False}), 0)
        c = normalize_curve(dict(freq=1000, width=2, plateau=.5, gain=-12))
        np.testing.assert_allclose(curve_gain_db(np.array([800., 1000., 1200.]), c), -12)
        scaled = curve_gain_db(np.array([1000.]), {**c, 'scale': 2})
        self.assertAlmostEqual(scaled[0], -24)

    def _amplitude(self, audio, freq):
        # Ignore STFT boundaries when measuring a stationary sine.
        mono = audio[self.sr // 2:-self.sr // 2, 0]
        t = np.arange(len(mono)) / self.sr
        return abs(np.sum(mono * np.exp(-2j * np.pi * freq * t))) * 2 / len(mono)

    def test_eq_changes_audio_by_the_displayed_gain(self):
        for gain in (-12, 6):
            c = normalize_curve(dict(freq=1000, gain=gain, width=2, plateau=.4, dynamic=False))
            s = main.FilterSettings(tones_on=False, eq_curves=[c])
            out = self.processor.apply(s)
            change = 20 * np.log10(self._amplitude(out, 1000) / self._amplitude(self.audio, 1000))
            self.assertAlmostEqual(change, gain, delta=.15)
            change_high = 20 * np.log10(self._amplitude(out, 8000) / self._amplitude(self.audio, 8000))
            self.assertAlmostEqual(change_high, 0, delta=.1)
            np.testing.assert_allclose(out[:, 0], -out[:, 1], atol=1e-6)

    def test_modes_bypass_disabled_and_inactive_filters(self):
        c = normalize_curve(dict(freq=1000, gain=-12, width=2))
        s = main.FilterSettings(tones_on=False, shelf_on=True, fade_start=6000, cutoff=7000,
                                eq_curves=[{**c, 'on': False}])
        np.testing.assert_array_equal(self.processor.apply(s), self.audio)
        out = self.processor.apply(replace(s, edit_mode='range', eq_curves=[c]))
        self.assertAlmostEqual(self._amplitude(out, 1000), self._amplitude(self.audio, 1000), delta=.0001)
        self.assertLess(self._amplitude(out, 8000), self._amplitude(self.audio, 8000) * .3)

    def test_detector_rules_change_results(self):
        rng = np.random.default_rng(2)
        x = (.001 * rng.normal(size=len(self.t)) + .02 * np.sin(2*np.pi*8000*self.t)).astype(np.float32)
        p = main.FilterProcessor()
        p.load(x[:, None], self.sr)
        s = main.FilterSettings(threshold_db=8., second_pass=False)
        detected = p.detect_tones(s.threshold_db, s.min_hz, s=s)
        self.assertTrue(any(abs(n['freq'] - 8000) < 5 for n in detected))
        self.assertFalse(p.detect_tones(8, 5000, s=replace(s, max_hz=7000)))
        self.assertLessEqual(len(p.detect_tones(2, 5000, s=replace(s, max_outliers=1))), 1)
        capped = p.detect_tones(8, 5000, s=replace(s, notch_cap_hz=4))
        self.assertTrue(all(n['width'] <= 4 for n in capped))
        narrow_floor = p.tone_prominence(replace(s, floor_hz=20))
        broad_floor = p.tone_prominence(replace(s, floor_hz=1200))
        self.assertGreater(np.max(np.abs(narrow_floor - broad_floor)), 1)
        self.assertLess(p._detection_data(replace(s, min_level_db=-30))[2].sum(),
                        p._detection_data(replace(s, min_level_db=-90))[2].sum())
        # An intermittent tone survives a high temporal percentile, but not a low one.
        burst = .001 * rng.normal(size=len(self.t))
        burst[len(burst)//2:] += .04*np.sin(2*np.pi*9000*self.t[len(burst)//2:])
        p.load(burst[:, None].astype(np.float32), self.sr)
        low = p.tone_prominence(replace(s, steady_pct=10))
        high = p.tone_prominence(replace(s, steady_pct=90))
        k = np.argmin(abs(p._welch_f - 9000))
        self.assertGreater(high[k], low[k] + 15)

    def test_settings_round_trip_and_copy(self):
        s = main.FilterSettings(preset='custom', floor_hz=1100, steady_pct=30,
                                eq_curves=[normalize_curve(dict(type='high_shelf', freq=7000))])
        restored = main.FilterSettings.from_json(s.to_json())
        self.assertEqual(restored.eq_curves, s.eq_curves)
        self.assertEqual(restored.floor_hz, 1100)
        self.assertEqual(restored.steady_pct, 30)
        snapshot = s.copy()
        snapshot.eq_curves[0]['gain'] = -55
        self.assertNotEqual(snapshot.eq_curves, s.eq_curves)
        invalid = main.FilterSettings.from_dict(dict(eq_curves=[None, {'gain': 'bad', 'width': float('nan')}],
                                                     floor_hz=-10, max_outliers=0))
        self.assertEqual(invalid.floor_hz, 20)
        self.assertEqual(invalid.max_outliers, 1)
        self.assertEqual(invalid.eq_curves[0]['width'], 1.)
        self.assertEqual(main.FilterSettings.from_dict({'shelf_on': True}).edit_mode, 'range')
        self.assertEqual(main.FilterSettings.from_json('{"shelf_on": true}').edit_mode, 'curves')
        dynamic = main.FilterSettings(preset='normal', tones_on=False,
                    eq_curves=[normalize_curve(dict(freq=8000, threshold=-50))])
        restored = main.FilterSettings.from_json(dynamic.to_json())
        self.assertFalse(restored.tones_on)
        self.assertEqual(restored.eq_curves, dynamic.eq_curves)
        historical = main.FilterSettings.from_dict({'eq_curves':[{'freq':8000}]})
        self.assertFalse(historical.eq_curves[0]['dynamic'])
        migrated = main.FilterSettings.from_json('{"preset":"normal", "tones_on":true, "eq_curves":[{"freq":8000}]}')
        self.assertTrue(migrated.eq_curves[0]['dynamic'])
        self.assertFalse(migrated.tones_on)

    def test_maximum_peak_width_rejects_broad_resonances(self):
        p = self.processor
        f = p._welch_f
        rel = 20 * np.exp(-.5 * ((f - 8000) / 40) ** 2)
        data = (rel, np.zeros_like(f), np.ones_like(f, dtype=bool))
        s = main.FilterSettings(max_fwhm_hz=40)
        with patch.object(p, '_detection_data', return_value=data):
            self.assertEqual(p.count_tones(rel, 4, 5000, s), 0)
            self.assertEqual(p.count_tones(rel, 4, 5000, replace(s, max_fwhm_hz=150)), 1)

    def test_complete_clean_and_analysis_with_custom_rules_and_eq(self):
        rng = np.random.default_rng(5)
        x = (.003*rng.normal(size=len(self.t)) + .06*np.sin(2*np.pi*9000*self.t)
             + .1*np.sin(2*np.pi*440*self.t)).astype(np.float32)
        p = main.FilterProcessor()
        p.load(x[:, None], self.sr)
        s = main.FilterSettings(floor_hz=800, steady_pct=30, max_outliers=3,
                                eq_curves=[normalize_curve(dict(type='high_shelf', freq=7000, gain=-6))])
        notches, out = p.clean(s, True)
        s.notches = notches
        result = p.analyse(out, s)
        self.assertEqual(out.shape, x[:, None].shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertLessEqual(len(notches), 3)
        self.assertGreater(result['tones_before'], result['tones_after'])
        self.assertEqual(result['view_proc'].shape, result['view_res'].shape)


class DynamicEQTests(unittest.TestCase):
    def setUp(self):
        main.GPU.enabled = False
        self.sr = 48000
        self.t = np.arange(self.sr*6)/self.sr

    def amplitude(self, x, freq, start, stop):
        block = x[int(start*self.sr):int(stop*self.sr), 0]
        t = np.arange(len(block))/self.sr
        return abs(np.sum(block*np.exp(-2j*np.pi*freq*t)))*2/len(block)

    def test_actual_audio_changes_with_level_and_matches_recorded_response(self):
        from dynamic_eq import response_at
        level = np.where(self.t < 2, .6, np.where(self.t < 4, .1, .003))
        x = level*np.sin(2*np.pi*1000*self.t) + .02*np.sin(2*np.pi*3000*self.t)
        audio = np.column_stack((x, -x)).astype(np.float32)
        p = main.FilterProcessor()
        p.load(audio, self.sr)
        c = normalize_curve(dict(freq=1000, gain=-12, width=2, plateau=.4,
                                 threshold=-30, ratio=2, attack=20, release=100))
        trace = {}
        out = p.apply(main.FilterSettings(tones_on=False, eq_curves=[c]), trace_out=trace)
        measured = []
        for start, stop, frame in [(1, 1.8, 1.4), (3, 3.8, 3.4), (5, 5.8, 5.4)]:
            db = 20*np.log10(self.amplitude(out, 1000, start, stop)/self.amplitude(audio, 1000, start, stop))
            measured.append(db)
            f, response, factors = response_at(trace, frame*self.sr)
            predicted = np.interp(1000, f, response)
            self.assertAlmostEqual(db, predicted, delta=.2)
            self.assertTrue(np.all((factors >= 0) & (factors <= 1)))
            unaffected = 20*np.log10(self.amplitude(out, 3000, start, stop)/self.amplitude(audio, 3000, start, stop))
            self.assertAlmostEqual(unaffected, 0, delta=.05)
        self.assertAlmostEqual(measured[0], -12, delta=.3)
        self.assertAlmostEqual(measured[1], -5, delta=.8)
        self.assertAlmostEqual(measured[2], 0, delta=.05)
        np.testing.assert_allclose(out[:, 0], -out[:, 1], atol=1e-6)

    def test_separate_frequency_regions_react_independently(self):
        t = self.t
        x = np.where(t < 3, .3*np.sin(2*np.pi*1000*t), .3*np.sin(2*np.pi*6000*t)).astype(np.float32)
        p = main.FilterProcessor()
        p.load(x[:, None], self.sr)
        curves = [normalize_curve(dict(freq=f, gain=-12, width=.5, threshold=-30)) for f in [1000, 6000]]
        trace = {}
        out = p.apply(main.FilterSettings(tones_on=False, eq_curves=curves), trace_out=trace)
        from dynamic_eq import response_at
        first = response_at(trace, self.sr*1.5)[2]
        second = response_at(trace, self.sr*4.5)[2]
        self.assertGreater(first[0], .7)
        self.assertLess(first[1], .01)
        self.assertLess(second[0], .01)
        self.assertGreater(second[1], .7)
        self.assertTrue(np.isfinite(out).all())

    def test_attack_release_and_ratio_one_bypass(self):
        from dynamic_eq import activity_envelope
        levels = np.r_[np.full(30, -6), np.full(100, -100)]
        c = normalize_curve(dict(gain=-12, threshold=-30, attack=100, release=250))
        envelope = activity_envelope(levels, c, 12, .01)
        self.assertLess(envelope[0], envelope[15])
        self.assertGreater(envelope[30], envelope[60])
        self.assertGreater(envelope[30], .5)
        self.assertLess(envelope[-1], .03)
        np.testing.assert_array_equal(activity_envelope(levels, {**c, 'ratio':1}, 12, .01), 0)
        np.testing.assert_array_equal(activity_envelope(levels, {**c, 'dynamic':False}, 12, .01), 1)

    def test_automatic_curves_are_calibrated_and_worker_returns_their_trace(self):
        rng = np.random.default_rng(8)
        t = self.t[:self.sr*2]
        x = (.001*rng.normal(size=len(t)) + .04*np.sin(2*np.pi*8000*t)).astype(np.float32)
        p = main.FilterProcessor()
        p.load(x[:, None], self.sr)
        settings = main.FilterSettings(threshold_db=8, second_pass=False)
        curves = p.suggest_eq_curves(settings)
        self.assertTrue(any(abs(c['freq']-8000) < 10 for c in curves))
        self.assertTrue(all(c['dynamic'] and c['auto'] for c in curves))
        self.assertTrue(all(c['width'] >= .65 for c in curves))
        wider = p.suggest_eq_curves(replace(settings, auto_min_width=1.))
        self.assertTrue(wider)
        self.assertTrue(all(c['width'] >= 1. for c in wider))
        self.assertTrue(all(-100 < c['threshold'] < -6 for c in curves))
        self.assertLessEqual(len(curves), 6)
        worker = main.CleanWorker(p, settings, False, 42)
        results, errors = [], []
        worker.done.connect(results.append)
        worker.error.connect(lambda error, generation: errors.append(error))
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result['suggested'])
        self.assertFalse(result['settings'].tones_on)
        self.assertTrue(result['eq_trace'])
        self.assertEqual(result['eq_trace']['curves'], result['settings'].eq_curves)
        manual = normalize_curve(dict(freq=1000, gain=-3, auto=False))
        old_auto = normalize_curve(dict(freq=9000, auto=True))
        settings.eq_curves = [manual, old_auto]
        worker = main.CleanWorker(p, settings, False, 43, suggest=True)
        refreshed = []
        worker.done.connect(refreshed.append)
        worker.run()
        self.assertEqual(refreshed[0]['settings'].eq_curves[0], manual)
        self.assertFalse(any(c['freq'] == 9000 for c in refreshed[0]['settings'].eq_curves))


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = main.QApplication.instance() or main.QApplication([])
        cls.app.setStyle('Fusion')
        # Windows' offscreen platform does not discover system fonts itself.
        font = Path('C:/Windows/Fonts/segoeui.ttf')
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat)
        self.settings.setValue('use_gpu', False)
        self.patchers = [patch.object(main, 'QSettings', lambda *a: self.settings)]
        for key, value in [('INPUT_DIR', root/'input'), ('OUTPUT_DIR', root/'output'),
                           ('PRESET_DIR', root/'presets'), ('HISTORY_FILE', root/'history.json'),
                           ('VERSION_DIR', root/'versions')]:
            self.patchers.append(patch.object(main.MainWindow, key, value))
        for p in self.patchers:
            p.start()
        self.win = main.MainWindow()
        self.win.resize(1600, 1000)
        self.win.show()
        self.app.processEvents()
        self.sr = 48000
        t = np.arange(self.sr * 2) / self.sr
        x = np.where(t < 1, .3*np.sin(2*np.pi*600*t), .04*np.sin(2*np.pi*6000*t)).astype(np.float32)
        self.audio = np.column_stack((x, -x))
        self.win._audio_orig = self.audio
        self.win.sample_rate = self.sr
        # UI tests do not start background renders or write history.
        self.win._request_clean = lambda **kwargs: None

    def tearDown(self):
        self.win.close()
        self.app.processEvents()
        self.settings.sync()
        for p in reversed(self.patchers):
            p.stop()
        self.tmp.cleanup()

    def test_volume_and_analyzer_preferences_survive_restart(self):
        for volume in (37, 0, 100):
            with self.subTest(volume=volume):
                w = self.win
                w.slider_vol.setValue(volume)
                w._set_analyzer_smoothing(1/48)
                w._set_analyzer_range(48)
                w._set_peak_decay(0.)
                w.btn_auto_cuts.setChecked(True)
                w.close()
                self.settings = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
                self.win = w = main.MainWindow()
                self.assertEqual(w.slider_vol.value(), volume)
                self.assertAlmostEqual(w._cb_volume, volume/100.)
                self.assertEqual(w.spectrum.smoothing_octaves, 1/48)
                self.assertEqual(w.spectrum.gain_range, 48)
                self.assertEqual(w.spectrum.peak_decay_db, 0.)
                self.assertTrue(w.spectrum.show_auto_cuts)
                self.assertTrue(w.btn_auto_cuts.isChecked())


    def test_compact_sidebar_path_copy_and_uniform_toolbar_controls(self):
        from PyQt6.QtWidgets import QPushButton, QToolButton, QComboBox
        w = self.win
        path = str(Path(self.tmp.name)/'Long track name.wav')
        w._set_track_path(path)
        self.assertTrue(w.track_title.isReadOnly())
        self.assertEqual(w.track_title.text(), str(Path(path).resolve()))
        w.btn_copy_path.click()
        self.assertEqual(self.app.clipboard().text(), w.track_title.text())
        w.track_title.selectAll()
        QTest.keyClick(w.track_title, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.app.clipboard().text(), w.track_title.text())
        self.assertTrue(w.tracks_panel.isAncestorOf(w.btn_tracks))
        self.assertFalse(w.app_header.isAncestorOf(w.btn_tracks))
        expanded = w.tracks_panel.width()
        w.btn_tracks.click()
        self.app.processEvents()
        self.assertEqual(w.tracks_panel.width(), 44)
        self.assertTrue(w.btn_tracks.isVisible())
        self.assertLess(w.btn_tracks.mapTo(w.tracks_panel, QPoint()).y(), 15)
        self.assertFalse(w.file_list.isVisible())
        w.btn_tracks.click()
        self.app.processEvents()
        self.assertTrue(w.file_list.isVisible())
        self.assertAlmostEqual(w.tracks_panel.width(), expanded, delta=4)
        for panel in (w.app_header, w.plot_toolbar, w.curve_tools):
            for control in panel.findChildren((QPushButton, QToolButton, QComboBox)):
                if control.isVisible():
                    self.assertEqual(control.height(), 34)
                    self.assertEqual(control.font().pixelSize(), 14)
        w._add_paths([path])
        self.assertLessEqual(w.file_list.sizeHintForRow(0), 34)

    def test_save_menu_routes_choices_and_settings_share_one_entry(self):
        w = self.win
        menu = w.btn_save.menu()
        menu.aboutToShow.emit()
        self.assertFalse(w.save_current_action.isEnabled())
        w._audio_proc = self.audio.copy()
        w.btn_export.setEnabled(True)
        w.btn_export_diff.setEnabled(True)
        menu.aboutToShow.emit()
        with patch.object(w, '_save') as save:
            menu.actions()[0].trigger()
            self.assertEqual(save.call_args.args[1], w.name_suffix)
            np.testing.assert_array_equal(save.call_args.args[0], self.audio)
            menu.actions()[1].trigger()
            self.assertEqual(save.call_args.args[1], w.DIFF_SUFFIX)
            np.testing.assert_array_equal(save.call_args.args[0], np.zeros_like(self.audio))
        self.assertIsNone(w.btn_more.menu())

    def test_color_profiles_accent_and_sidebar_survive_restart(self):
        from PyQt6.QtGui import QColor
        w = self.win
        self.assertEqual(w.color_profile, 'dark')
        self.assertEqual(w.spectrum_accent, 'blue')
        before = w.btn_view_live.palette().button().color().name()
        w._set_spectrum_accent('teal')
        self.assertEqual(w.btn_view_live.palette().button().color().name(), before)
        self.assertEqual(w.spectrum._peak_line.get_color(), QColor('#58c7b2').darker(220).name())
        w._set_color_profile('black')
        w.btn_tracks.click()
        w.close()
        self.settings = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
        self.win = w = main.MainWindow()
        w.show()
        self.app.processEvents()
        self.assertEqual(w.color_profile, 'black')
        self.assertEqual(w.spectrum_accent, 'teal')
        self.assertTrue(w._tracks_collapsed)
        self.assertEqual(w.tracks_panel.width(), 44)
        self.assertTrue(w.btn_tracks.isVisible())
        w.btn_tracks.click()
        self.app.processEvents()
        self.assertTrue(w.file_list.isVisible())


    def test_eq_panel_drag_stays_put_and_clamps_after_resizing(self):
        w = self.win
        w._add_curve(1500, 'bell')
        self.app.processEvents()
        before = w.curve_card.pos()
        settings = w.fs.to_dict()
        handle = w.curve_drag_handle
        point = QPoint(80, handle.height()//2)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(handle, point+QPoint(160, -180))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=point)
        self.assertNotEqual(w.curve_card.pos(), before)
        self.assertEqual(w.fs.to_dict(), settings)
        placed = w.curve_card.pos()
        w._refresh_curve_controls()
        w._update_view_mode()
        self.app.processEvents()
        self.assertEqual(w.curve_card.pos(), placed)
        w.resize(1280, 820)
        w._show_inspector('outliers')
        self.app.processEvents()
        w._position_curve_editor()
        self.assertFalse(w._editor_docked)
        self.assertTrue(w.view_frame.rect().contains(w.curve_card.geometry()))
        self.assertEqual(len(self.settings.value('eq_editor_position', [], type=list)), 2)

    def test_settings_button_opens_tabbed_modal_and_restores_live_controls(self):
        from PyQt6.QtCore import QTimer
        w = self.win
        parents = {key: w.inspector_cards[key].parentWidget() for key in ('strength','shelf','advanced','tones')}
        errors = []
        def edit_dialog():
            dialog = self.app.activeModalWidget()
            try:
                self.assertIsInstance(dialog, main.SettingsDialog)
                self.assertEqual([dialog.tabs.tabText(i) for i in range(dialog.tabs.count())],
                                 ['General','Processing','Outliers','Analyzer','Appearance','Export','History'])
                dialog.combo_shaping.setCurrentIndex(dialog.combo_shaping.findData('range'))
                w.detector_controls['floor_hz'].setValue(900)
                dialog.combo_smoothing.setCurrentIndex(dialog.combo_smoothing.findData(1/48))
                dialog.combo_gain_range.setCurrentIndex(dialog.combo_gain_range.findData(48))
                dialog.combo_peak_decay.setCurrentIndex(dialog.combo_peak_decay.findData(0.))
                dialog.combo_accent.setCurrentIndex(dialog.combo_accent.findData('amber'))
                dialog.combo_profile.setCurrentIndex(dialog.combo_profile.findData('black'))
                dialog.edit_suffix.setText('_test')
                dialog.chk_lines['changes'].setChecked(False)
            except Exception as error:
                errors.append(error)
            finally:
                if dialog:
                    dialog.accept()
        QTimer.singleShot(0, edit_dialog)
        w.btn_more.click()
        if errors:
            raise errors[0]
        self.assertEqual(w.fs.edit_mode, 'range')
        self.assertEqual(w.fs.floor_hz, 900)
        self.assertEqual(w.spectrum.smoothing_octaves, 1/48)
        self.assertEqual(w.spectrum.gain_range, 48)
        self.assertEqual(w.spectrum.peak_decay_db, 0.)
        self.assertEqual(w.spectrum_accent, 'amber')
        self.assertEqual(w.color_profile, 'black')
        self.assertEqual(w.name_suffix, '_test')
        self.assertFalse(w._hist_lines['changes'])
        for key, parent in parents.items():
            self.assertIs(w.inspector_cards[key].parentWidget(), parent)
        w._show_inspector('outliers')
        self.app.processEvents()
        self.assertTrue(w.box_tones.isVisible())

    def test_spectrum_help_is_explicit_and_canvas_tooltips_are_empty(self):
        from PyQt6.QtCore import QTimer
        w = self.win
        for canvas in (w.spectrum, w.closeup, w.tone_view):
            self.assertEqual(canvas.toolTip(), '')
        titles = []
        def close_help():
            dialog = self.app.activeModalWidget()
            if dialog:
                titles.append(dialog.windowTitle())
                dialog.accept()
        QTimer.singleShot(0, close_help)
        w.btn_spectrum_info.click()
        self.assertEqual(titles, ['Spectrum and EQ help'])

    def test_single_display_and_mode_switch(self):
        w = self.win
        self.assertEqual(w.display_stack.currentIndex(), 0)
        self.assertGreater(w.spectrum.height(), 600)
        self.assertFalse(w.box_shelf.isVisible())
        w.btn_view_close.click()
        self.assertEqual(w.display_stack.currentIndex(), 1)
        w.btn_view_lines.click()
        self.assertEqual(w.lower_stack.currentIndex(), 1)
        w.btn_view_live.click()
        self.assertEqual(w.display_stack.currentIndex(), 0)
        w.combo_edit_mode.setCurrentIndex(1)
        self.assertEqual(w.fs.edit_mode, 'range')
        self.assertTrue(w.box_shelf.isVisible())
        self.assertFalse(w.curve_body.isVisible())
        w.combo_edit_mode.setCurrentIndex(0)
        self.assertEqual(w.fs.edit_mode, 'curves')

    def test_outlier_live_trace_follows_seek_and_audition_source(self):
        w = self.win
        w.btn_view_close.click()
        self.app.processEvents()
        w._seek_to(int(.25*self.sr))
        f, db = w.closeup._live.get_data()
        self.assertAlmostEqual(f[np.nanargmax(db)], .6, delta=.02)
        w._seek_to(int(1.5*self.sr))
        f, db = w.closeup._live.get_data()
        self.assertAlmostEqual(f[np.nanargmax(db)], 6., delta=.02)
        t = np.arange(len(self.audio))/self.sr
        cleaned = np.column_stack([.1*np.sin(2*np.pi*2000*t)]*2)
        w._audio_proc = w._audio_play = cleaned
        w._slot = 1
        w._update_view_mode()
        f, db = w.closeup._live.get_data()
        self.assertAlmostEqual(f[np.nanargmax(db)], 2., delta=.02)
        self.assertIn('B', w.analysis_position.text())

    def test_spectrogram_cursor_updates_without_rebuilding_analysis(self):
        from matplotlib.backend_bases import MouseEvent, MouseButton
        w = self.win
        w.processor.load(self.audio, self.sr)
        view = w.processor.tone_view(w.fs)
        w.tone_view.set_original(view, 1000, 20000)
        w._seek_to(int(.5*self.sr))
        w.btn_view_lines.click()
        self.app.processEvents()
        canvas = w.tone_view
        self.assertEqual(list(canvas._playhead.get_xdata()), [.5, .5])
        image = canvas._img
        w.is_playing = w._cb_playing = True
        w._cb_frame = int(1.25*self.sr)
        w._poll_position()
        self.assertEqual(list(canvas._playhead.get_xdata()), [1.25, 1.25])
        self.assertIs(canvas._img, image)
        self.assertEqual(canvas.ax.get_title(), '')
        w.is_playing = False
        x, y = canvas.ax.transData.transform((.8, 6.))
        canvas._on_press(MouseEvent('button_press_event', canvas, x, y, button=MouseButton.LEFT))
        canvas._on_release(MouseEvent('button_release_event', canvas, x, y, button=MouseButton.LEFT))
        self.assertAlmostEqual(w._cb_frame/self.sr, .8, delta=.01)

    def test_quick_presets_apply_and_stay_in_sync_with_custom_edits(self):
        w = self.win
        index = w.combo_quick_preset.findData('strong')
        w.combo_quick_preset.setCurrentIndex(index)
        w.combo_quick_preset.activated.emit(index)
        self.assertEqual(w.fs.preset, 'strong')
        self.assertEqual(w.fs.threshold_db, 3.)
        self.assertTrue(w._auto_curve_requested)
        self.assertFalse(w.fs.tones_on)
        w._on_sens_changed(6)
        self.assertEqual(w.fs.preset, 'custom')
        self.assertEqual(w.combo_quick_preset.currentData(), 'custom')
        w.preset_buttons['gentle'].click()
        self.assertEqual(w.combo_quick_preset.currentData(), 'gentle')

    def test_resizable_browser_and_end_only_filename_elision(self):
        from ui_widgets import TrackDelegate, ElidingLabel
        w = self.win
        path = self.tmp.name + '/An extremely long filename whose opening must stay readable.mp3'
        w._add_paths([path])
        self.assertEqual(w.file_list.textElideMode(), Qt.TextElideMode.ElideRight)
        self.assertIsInstance(w.file_list.itemDelegate(), TrackDelegate)
        self.assertTrue(w.file_list.item(0).text().startswith('An extremely long'))
        self.assertEqual(w.file_list.item(0).toolTip(), path)
        before = w.tracks_panel.width()
        w.workspace_splitter.setSizes([400, 1100])
        self.app.processEvents()
        self.assertGreater(w.tracks_panel.width(), before)
        title = ElidingLabel('Keep the opening and hide only the ending of this long title')
        title.resize(140, 30)
        title.show()
        self.app.processEvents()
        self.assertTrue(title.text().startswith('Keep the'))
        self.assertTrue(title.text().endswith(chr(0x2026)))
        self.assertEqual(title.toolTip(), title._full_text)
        title.close()

    def test_clean_default_and_optional_auto_response(self):
        w = self.win
        w.fs.notches = [dict(freq=6000+i*80, width=20, on=True) for i in range(126)]
        w._refresh_notch_views()
        self.assertFalse(w.spectrum.show_auto_cuts)
        np.testing.assert_array_equal(w.spectrum._response_db, 0)
        self.assertEqual(len(w.spectrum._overlay), 2)  # quiet zero line only
        self.assertIn('126', w.btn_inspector_outliers.text())
        w.btn_auto_cuts.click()
        self.assertTrue(w.spectrum.show_auto_cuts)
        self.assertLess(w.spectrum._auto_response.min(), -30)
        np.testing.assert_array_equal(w.spectrum._response_db, 0)
        w.btn_auto_cuts.click()
        self.assertEqual(len(w.spectrum._overlay), 2)

    def test_small_inspector_preserves_graph_and_curve_controls(self):
        w = self.win
        w.resize(1280, 820)
        w._add_curve(11800, 'notch')
        w._show_inspector('outliers')
        self.app.processEvents()
        w._position_curve_editor()
        self.app.processEvents()
        self.assertTrue(w.inspector.isVisible())
        self.assertTrue(w._editor_docked)
        self.assertGreater(w.spectrum.height(), 300)
        self.assertGreaterEqual(w.curve_card.y(), w.display_stack.geometry().bottom())
        self.assertLessEqual(w.curve_card.geometry().right(), w.view_frame.width())
        for control in w.rotary_controls.values():
            self.assertTrue(control.spin.isVisible())
        w._hide_inspector()
        self.app.processEvents()
        w._position_curve_editor()
        self.assertFalse(w._editor_docked)

    def test_dials_analyzer_options_and_deep_curve_handles(self):
        w = self.win
        w._add_curve(1000, 'bell')
        dial = w.rotary_controls['width'].dial
        old = w.fs.eq_curves[0]['width']
        dial.setValue(dial.value()+500)
        self.assertGreater(w.fs.eq_curves[0]['width'], old)
        w.combo_eq_type.setCurrentIndex(w.combo_eq_type.findData('low_shelf'))
        self.assertFalse(w.rotary_controls['plateau'].dial.isEnabled())
        self.assertFalse(w.curve_controls['plateau'].isEnabled())
        w.combo_eq_type.setCurrentIndex(w.combo_eq_type.findData('notch'))
        w.curve_controls['gain'].setValue(-60)
        w.spectrum.draw()
        c = w.spectrum
        y = c._display_handle_gain(w.fs.eq_curves[0])
        self.assertGreater(y, c.eq_ax.get_ylim()[0])
        self.assertEqual(c._curve_hit(self.event('button_press_event', 1000, y, MouseButton.LEFT)), 0)
        w._set_analyzer_range(48)
        self.assertEqual(c.eq_ax.get_ylim(), (-48, 48))
        w._set_analyzer_smoothing(1/12)
        self.assertEqual(c.smoothing_octaves, 1/12)
        w._select_curve(-1)
        self.assertFalse(w.curve_card.isVisible())
        self.assertFalse(w.rotary_controls['freq'].dial.isEnabled())

    def test_frame_updates_preserve_zoom_and_peak_on_scrub(self):
        w, c = self.win, self.win.spectrum
        c.set_view(300, 9000)
        w.is_playing = True
        for frame in [24000, 36000, 60000]:
            w._cb_frame = frame
            w._update_live_spectrum()
            self.assertEqual(c.ax.get_xlim(), (300, 9000))
        peak = c._peak_db.copy()
        w.is_playing = False
        w._seek_to(10000)
        self.assertIsNotNone(c._peak_db)
        self.assertTrue(np.all(c._peak_db >= peak-.1))
        self.assertEqual(c._last_frame, 10000)
        self.assertTrue(len(c._peak_line.get_xdata()))

    def test_current_peak_scrub_and_phase_cancellation(self):
        w, c = self.win, self.win.spectrum
        w._cb_frame = self.sr // 2
        w._update_live_spectrum()
        self.assertGreater(c._current_db.max(), -15)
        self.assertIsNotNone(c._peak_db)
        peak = c._peak_db.copy()
        w.is_playing = True
        w._cb_frame = int(self.sr * 1.5)
        w._update_live_spectrum()
        self.assertTrue(np.all(c._peak_db >= peak-.1))
        w._timeline_pressed()
        self.assertFalse(w.is_playing)
        w.slider_pos.setValue(7500)
        w._timeline_moved(7500)
        self.assertAlmostEqual(c._last_frame / self.sr, 1.5, delta=.01)
        self.assertGreater(c._current_db.max(), -33)
        w._scrub_was_playing = False
        w._timeline_released()
        self.assertIsNotNone(c._peak_db)
        w._seek_rel(-1)
        self.assertIsNotNone(c._peak_db)
        # Slow release uses wall time, and keeps the current frame under it.
        w._seek_to(int(self.sr*1.5))
        c._last_tick = time.perf_counter()-10
        before = c._peak_db.copy()
        c._decay_paused_peak()
        expected = np.maximum(c._current_db, before-5.)
        np.testing.assert_allclose(c._peak_db, expected, atol=.01)
        c.reset_peak()
        self.assertIsNone(c._peak_db)

    def event(self, name, freq, gain, button=None, **kw):
        c = self.win.spectrum
        x, y = c.eq_ax.transData.transform((freq, gain))
        return MouseEvent(name, c, x, y, button=button, **kw)

    def test_curve_drag_wheel_right_drag_and_numeric_controls(self):
        w = self.win
        c = w.spectrum
        w._add_curve(1000)
        c.draw()
        c._on_press(self.event('button_press_event', 1000, -6, MouseButton.LEFT))
        c._on_motion(self.event('motion_notify_event', 2000, -12, MouseButton.LEFT))
        c._on_release(self.event('button_release_event', 2000, -12, MouseButton.LEFT))
        self.assertAlmostEqual(w.fs.eq_curves[0]['freq'], 2000, delta=30)
        self.assertAlmostEqual(w.fs.eq_curves[0]['gain'], -12, delta=.3)
        before = w.fs.eq_curves[0]['width']
        c._on_scroll(self.event('scroll_event', 2000, -12, 'up', step=1))
        self.assertGreater(w.fs.eq_curves[0]['width'], before)
        c.draw()
        p = self.event('button_press_event', 2000, -12, MouseButton.RIGHT)
        c._on_press(p)
        move = MouseEvent('motion_notify_event', c, p.x+50, p.y+60, button=MouseButton.RIGHT)
        c._on_motion(move)
        c._on_release(move)
        self.assertGreater(w.fs.eq_curves[0]['plateau'], .2)
        w.curve_controls['freq'].setValue(4500)
        self.assertEqual(w.fs.eq_curves[0]['freq'], 4500)
        w.curve_controls['scale'].setValue(150)
        self.assertEqual(w.fs.eq_curves[0]['scale'], 1.5)
        w.chk_curve_on.setChecked(False)
        self.assertFalse(w.fs.eq_curves[0]['on'])
        w.btn_curve_remove.click()
        self.assertEqual(w.fs.eq_curves, [])

    def test_auto_curves_button_completes_real_background_render(self):
        w = self.win
        rng = np.random.default_rng(15)
        t = np.arange(self.sr*2)/self.sr
        audio = (.001*rng.normal(size=len(t))+.04*np.sin(2*np.pi*8000*t)).astype(np.float32)[:, None]
        w._audio_orig = audio
        w.current_path = str(Path(self.tmp.name)/'test.wav')
        w.processor.load(audio, self.sr)
        f, _ = w.processor.spectrum_original()
        w.closeup.set_original(f, w.processor.tone_prominence(w.fs), 4000, self.sr/2)
        w.tone_view.set_original(w.processor.tone_view(w.fs), 4000, self.sr/2)
        w._request_clean = main.MainWindow._request_clean.__get__(w)
        w.btn_auto_curves.click()
        deadline = time.monotonic()+10
        while w._audio_proc is None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNotNone(w._audio_proc)
        self.assertTrue(w.fs.eq_curves)
        self.assertTrue(all(c['dynamic'] for c in w.fs.eq_curves))
        self.assertTrue(w._b['eq_trace'])
        self.assertTrue(w.curve_card.isVisible())
        self.assertTrue(w.btn_export.isEnabled())
        self.assertFalse(w.box_tones.isChecked())
        self.assertTrue(w.detector_controls['floor_hz'].isEnabled())


    def test_dynamic_controls_chips_and_exact_playhead_response(self):
        from dynamic_eq import response_at
        w = self.win
        w._add_curve(6000, 'bell')
        self.assertEqual(w.btn_curve_add.text(), '+ Add curve')
        self.assertTrue(w.btn_auto_curves.isVisible())
        self.assertTrue(w.chk_curve_dynamic.isChecked())
        w.dynamic_controls['threshold'].setValue(-42)
        self.assertEqual(w.fs.eq_curves[0]['threshold'], -42)
        w.chk_curve_dynamic.setChecked(False)
        self.assertFalse(w.fs.eq_curves[0]['dynamic'])
        self.assertFalse(w.dynamic_controls['attack'].isEnabled())
        w.chk_curve_dynamic.setChecked(True)
        self.assertTrue(w.dynamic_controls['attack'].isEnabled())
        p = main.FilterProcessor()
        p.load(self.audio, self.sr)
        trace = {}
        processed = p.apply(main.FilterSettings(tones_on=False, eq_curves=w.fs.eq_curves), trace_out=trace)
        w._b = {'eq_trace':trace}
        w._audio_proc = processed
        w._cb_frame = int(self.sr*1.5)
        w._update_live_spectrum()
        expected = response_at(trace, w._cb_frame)[1]
        np.testing.assert_allclose(w.spectrum._response_artist.get_ydata(), expected)
        w._seek_to(int(self.sr*.5))
        earlier = response_at(trace, w._cb_frame)[1]
        np.testing.assert_allclose(w.spectrum._response_artist.get_ydata(), earlier)
        self.assertGreater(np.max(np.abs(earlier-expected)), 3)
        w.box_tones.setChecked(False)
        self.assertTrue(w.detector_controls['floor_hz'].isEnabled())
        w._select_curve(-1)
        self.assertFalse(w.curve_card.isVisible())
        w.band_chips.itemAt(0).widget().click()
        self.assertTrue(w.curve_card.isVisible())
        w.btn_curve_remove.click()
        self.assertFalse(w.fs.auto_eq)  # deleting the last curve must not regenerate it
        w.preset_buttons['normal'].click()
        self.assertTrue(w.fs.auto_eq)
        self.assertTrue(w._auto_curve_requested)
        self.assertFalse(w.fs.tones_on)


    def test_detector_controls_invalidate_detection(self):
        w = self.win
        before = w._detect_key()
        calls = []
        w._request_clean = lambda **kw: calls.append(kw)
        w.detector_controls['floor_hz'].setValue(900)
        self.assertEqual(w.fs.floor_hz, 900)
        self.assertNotEqual(before, w._detect_key())
        self.assertTrue(calls[-1]['detect'])
        saved = main.FilterSettings.from_json(self.settings.value('filter_settings_v2', ''))
        self.assertEqual(saved.floor_hz, 900)

    def test_analyzer_follows_audition_source_and_residual_boost(self):
        w = self.win
        w._cb_frame = self.sr // 2
        w._update_live_spectrum()
        original = w.spectrum._current_db.max()
        w._audio_play = self.audio * .1
        w._update_view_mode()
        self.assertAlmostEqual(w.spectrum._current_db.max() - original, -20, delta=.1)
        w._audio_solo = self.audio * .5
        w._solo_boost = 2.
        w._solo_residual = True
        w._update_view_mode()
        self.assertAlmostEqual(w.spectrum._current_db.max(), original, delta=.1)

    def test_add_menu_preserves_existing_shape(self):
        w = self.win
        w._add_curve(1000, 'bell')
        w.btn_curve_add.menu().actions()[2].trigger()  # Add a high shelf without changing the bell.
        self.assertEqual([c['type'] for c in w.fs.eq_curves], ['bell', 'high_shelf'])

    def test_tracks_detect_independently_and_restore_manual_edits_after_restart(self):
        import soundfile as sf
        w = self.win
        rng = np.random.default_rng(31)
        t = np.arange(self.sr*2)/self.sr
        paths = [str(Path(self.tmp.name)/name) for name in ('first.wav', 'second.wav')]
        for path, freq in zip(paths, (8000, 11000)):
            audio = .001*rng.normal(size=len(t))+.04*np.sin(2*np.pi*freq*t)
            sf.write(path, audio, self.sr, subtype='FLOAT')

        def open_track(index):
            w.file_list.setCurrentRow(index)
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                QTest.qWait(10)
                if w.current_path == paths[index] and w._audio_proc is not None and not w._loading_track:
                    return
            self.fail(f'Track {index} did not finish loading/rendering: {w.statusBar().currentMessage()}')

        w._request_clean = main.MainWindow._request_clean.__get__(w)
        w._add_paths(paths)
        open_track(0)
        self.assertTrue(any(abs(c['freq']-8000) < 10 for c in w.fs.eq_curves))
        w.spectrum.select_curve(0)
        w._refresh_curve_controls()
        w.curve_controls['width'].setValue(1.25)
        w.detector_controls['floor_hz'].setValue(900)
        w._add_curve(2500, 'bell')
        first = w.fs.to_dict()
        open_track(1)
        self.assertTrue(any(abs(c['freq']-11000) < 10 for c in w.fs.eq_curves))
        self.assertFalse(any(abs(c['freq']-2500) < 10 for c in w.fs.eq_curves))
        self.assertEqual(w.fs.floor_hz, 600)
        second = w.fs.to_dict()
        open_track(0)
        self.assertEqual(w.fs.to_dict(), first)
        open_track(1)
        self.assertEqual(w.fs.to_dict(), second)
        w.close()
        self.settings = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
        self.win = w = main.MainWindow()
        w.show()
        w._request_clean = main.MainWindow._request_clean.__get__(w)
        w._add_paths(paths)
        open_track(0)
        self.assertEqual(w.fs.to_dict(), first)
        open_track(1)
        self.assertEqual(w.fs.to_dict(), second)

    def test_config_copy_paste_is_explicit_independent_and_keeps_own_time_region(self):
        w = self.win
        source, target = (str(Path(self.tmp.name)/name) for name in ('source.wav', 'target.wav'))
        w.current_path = source
        w._track_detected = True
        w._add_curve(8000, 'bell')
        w.fs.floor_hz = 1200
        w.fs.profile_region = (.1, .4)
        w._save_filter_settings()
        w._update_config_buttons()
        w.btn_copy_config.click()
        original = w.fs.to_dict()
        w.current_path = target
        w._restore_track_config(target)
        self.assertEqual(w.fs.eq_curves, [])
        w.fs.profile_region = (.5, .9)
        w._update_config_buttons()
        self.assertTrue(w.btn_paste_config.isEnabled())
        w.btn_paste_config.click()
        self.assertEqual(w.fs.eq_curves, original['eq_curves'])
        self.assertEqual(w.fs.floor_hz, 1200)
        self.assertEqual(w.fs.profile_region, (.5, .9))
        w.curve_controls['gain'].setValue(-15)
        pasted = w.fs.to_dict()
        w.current_path = source
        w._restore_track_config(source)
        self.assertEqual(w.fs.to_dict(), original)
        w.current_path = target
        w._restore_track_config(target)
        self.assertEqual(w.fs.to_dict(), pasted)
        self.assertEqual(w._copied_config['settings'], original)

    def test_saved_empty_config_does_not_recreate_curves(self):
        w = self.win
        w.current_path = str(Path(self.tmp.name)/'empty.wav')
        w._track_detected = True
        w.fs.eq_curves = []
        w._save_filter_settings()
        w._restore_track_config(w.current_path)
        w.processor.load(self.audio, self.sr)
        results = []
        worker = main.CleanWorker(w.processor, w.fs, False, 1, auto_if_empty=not w._track_detected)
        worker.done.connect(results.append)
        with patch.object(w.processor, 'suggest_eq_curves', side_effect=AssertionError('Already initialized')):
            worker.run()
        self.assertTrue(results)
        self.assertFalse(results[0]['suggested'])
        self.assertEqual(results[0]['settings'].eq_curves, [])

    def test_add_curve_main_button_and_group_numeric_edits(self):
        w = self.win
        w.btn_curve_add.click()
        self.assertEqual(len(w.fs.eq_curves), 1)
        self.assertTrue(w.curve_card.isVisible())
        w._add_curve(4000, 'bell')
        w.btn_select_curves.click()
        self.assertEqual(w.spectrum.selected_curves, {0, 1})
        self.assertIn('2 curves selected', w.curve_selection_label.text())
        before = [dict(c) for c in w.fs.eq_curves]
        w.curve_controls['freq'].setValue(before[0]['freq']*2)
        for old, new in zip(before, w.fs.eq_curves):
            self.assertAlmostEqual(new['freq']/old['freq'], 2, delta=.01)
        w.curve_controls['width'].setValue(2)
        self.assertTrue(all(c['width'] == 2 for c in w.fs.eq_curves))
        w.curve_controls['gain'].setValue(-10)
        self.assertTrue(all(c['gain'] == -10 for c in w.fs.eq_curves))
        w.dynamic_controls['threshold'].setValue(-38)
        self.assertTrue(all(c['threshold'] == -38 for c in w.fs.eq_curves))
        w.btn_curve_remove.click()
        self.assertEqual(w.fs.eq_curves, [])
        self.assertFalse(w.fs.auto_eq)

    def test_group_handle_selection_drag_and_scroll(self):
        w, c = self.win, self.win.spectrum
        w._add_curve(1000, 'bell')
        w._add_curve(4000, 'bell')
        c.draw()
        c._on_press(self.event('button_press_event', 1000, -6, MouseButton.LEFT, key='control'))
        self.assertEqual(c.selected_curves, {0, 1})
        c._on_press(self.event('button_press_event', 1000, -6, MouseButton.LEFT))
        c._on_motion(self.event('motion_notify_event', 1500, -10, MouseButton.LEFT))
        c._on_release(self.event('button_release_event', 1500, -10, MouseButton.LEFT))
        self.assertEqual(c.selected_curves, {0, 1})
        self.assertAlmostEqual(w.fs.eq_curves[1]['freq']/w.fs.eq_curves[0]['freq'], 4., delta=.01)
        self.assertAlmostEqual(w.fs.eq_curves[1]['gain'], -10, delta=.2)
        before = [item['width'] for item in w.fs.eq_curves]
        c._on_scroll(self.event('scroll_event', 1500, -10, 'up', step=1))
        self.assertTrue(all(new['width'] > old for new, old in zip(w.fs.eq_curves, before)))
        c._on_press(self.event('button_press_event', w.fs.eq_curves[1]['freq'],
                              w.fs.eq_curves[1]['gain'], MouseButton.LEFT, key='shift'))
        self.assertEqual(c.selected_curves, {0})

    def test_gain_and_scale_both_have_distinct_effects(self):
        w = self.win
        w._add_curve(2000, 'bell')
        w.curve_controls['gain'].setValue(-12)
        c = w.fs.eq_curves[0]
        self.assertEqual(c['width']*c['scale'], 1)
        self.assertAlmostEqual(curve_gain_db(np.array([2000.]), c)[0], -12)
        w.curve_controls['scale'].setValue(150)
        c = w.fs.eq_curves[0]
        self.assertEqual(c['width']*c['scale'], 1.5)
        self.assertAlmostEqual(curve_gain_db(np.array([2000.]), c)[0], -18)
        self.assertIn('-18.0 dB', w.curve_effective_label.text())

    def test_edit_and_render_keep_native_spectrum_visible(self):
        w, c = self.win, self.win.spectrum
        w._cb_frame = self.sr//2
        w._update_live_spectrum()
        peak = c._peak_db.copy()
        w._add_curve(1000, 'bell')
        w.curve_controls['width'].setValue(1.5)
        w._update_view_mode()  # Also invoked when a background render finishes.
        self.assertTrue(np.all(c._peak_db >= peak-.1))
        c.draw()
        self.app.processEvents()
        painted = c.grab().toImage()
        # Native animated layers must survive a full axes redraw.
        self.assertFalse(painted.isNull())
        self.assertIsNotNone(c._axes_pixmap)
        self.assertGreater(np.max(c._current_line.get_ydata()), -15)

    def test_peak_fade_and_auto_width_controls(self):
        w = self.win
        w._set_peak_decay(0.)
        self.assertEqual(w.spectrum.peak_decay_db, 0.)
        w.auto_width_actions[2][0].trigger()
        self.assertEqual(w.fs.auto_min_width, 1.)
        saved = main.FilterSettings.from_json(self.settings.value('filter_settings_v2', ''))
        self.assertEqual(saved.auto_min_width, 1.)
        self.assertEqual(w._pos_timer.interval(), 6)
        self.assertEqual(w._pos_timer.timerType(), Qt.TimerType.PreciseTimer)

    def test_real_timeline_mouse_scrub(self):
        w = self.win
        w.is_playing = True
        w._update_live_spectrum()
        slider = w.slider_pos
        QTest.mousePress(slider, Qt.MouseButton.LeftButton,
                         pos=QPoint(slider.width()//2, slider.height()//2))
        self.app.processEvents()
        self.assertIsNotNone(w.spectrum._peak_db)
        self.assertFalse(w.is_playing)
        QTest.mouseMove(slider, QPoint(int(slider.width()*.75), slider.height()//2))
        self.app.processEvents()
        self.assertGreater(w.spectrum._last_frame, len(self.audio)*.6)
        w._scrub_was_playing = False
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton,
                           pos=QPoint(int(slider.width()*.75), slider.height()//2))
        self.assertIsNotNone(w.spectrum._peak_db)


if __name__ == '__main__':
    unittest.main()
