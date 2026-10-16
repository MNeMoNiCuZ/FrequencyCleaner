"""Spectrum-first workspace layout. Processing and project state live in main.py."""
from pathlib import Path
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QComboBox, QCheckBox, QSlider, QListWidget, QScrollArea,
    QApplication, QStackedWidget, QSplitter, QButtonGroup, QMenu, QSizePolicy, QDoubleSpinBox, QToolButton, QLineEdit)
from ui_widgets import (TrackDelegate, RotaryControl, PanelDragHandle, ui_icon, COLOR_PROFILES,
                        SPECTRUM_ACCENTS, studio_stylesheet)


class StudioWorkspaceMixin:
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.menuBar().hide()

        self.app_header = QWidget()
        self.app_header.setObjectName('app_header')
        header = QHBoxLayout(self.app_header)
        header.setContentsMargins(8, 7, 8, 7)
        header.setSpacing(6)
        self.track_title = QLineEdit()
        self.track_title.setObjectName('track_path')
        self.track_title.setReadOnly(True)
        self.track_title.setPlaceholderText('Open a track')
        self.track_title.setAccessibleName('Full path to the current track')
        self.track_title.setMinimumWidth(100)
        self.track_title.setFixedHeight(34)
        header.addWidget(self.track_title, stretch=1)
        self.btn_copy_path = QToolButton()
        self.btn_copy_path.setIcon(ui_icon('copy'))
        self.btn_copy_path.setToolTip('Copy full file path')
        self.btn_copy_path.setAccessibleName('Copy full file path')
        self.btn_copy_path.setEnabled(False)
        self.btn_copy_path.clicked.connect(lambda: QApplication.clipboard().setText(self.track_title.text()))
        header.addWidget(self.btn_copy_path)
        self.btn_copy_config = QPushButton('Copy config')
        self.btn_paste_config = QPushButton('Paste config')
        self.btn_copy_config.setObjectName('btn_small')
        self.btn_paste_config.setObjectName('btn_small')
        self.btn_copy_config.setToolTip('Copy this track’s curves, detector rules and processing settings. Time selections stay with each track.')
        self.btn_paste_config.setToolTip('Apply the copied config to this track. The source track keeps its own config.')
        self.btn_copy_config.setEnabled(False)
        self.btn_paste_config.setEnabled(False)
        self.btn_copy_config.clicked.connect(self._copy_track_config)
        self.btn_paste_config.clicked.connect(self._paste_track_config)
        header.addWidget(self.btn_copy_config)
        header.addWidget(self.btn_paste_config)
        self.btn_inspector_outliers = QPushButton('Outliers')
        self.btn_inspector_history = QPushButton('History')
        self.btn_inspector_outliers.setCheckable(True)
        self.btn_inspector_history.setCheckable(True)
        header.addWidget(self.btn_inspector_outliers)
        header.addWidget(self.btn_inspector_history)
        self.btn_more = QToolButton()
        self.btn_more.setIcon(ui_icon('settings'))
        self.btn_more.setToolTip('Settings')
        self.btn_more.setAccessibleName('Settings')
        self.btn_more.clicked.connect(self._open_settings)
        outer.addWidget(self.app_header)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(4)
        self.tracks_panel = self._build_tracks_panel()
        self.workspace_splitter.addWidget(self.tracks_panel)
        workspace = QWidget()
        workspace.setObjectName('workspace')
        work = QHBoxLayout(workspace)
        work.setContentsMargins(6, 6, 6, 6)
        work.setSpacing(6)
        center = QWidget()
        center.setObjectName('workspace')
        center.setMinimumWidth(650)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        self.view_frame = QFrame()
        self.view_frame.setObjectName('view_frame')
        vf = QVBoxLayout(self.view_frame)
        vf.setContentsMargins(1, 1, 1, 1)
        vf.setSpacing(0)
        # The existing processor owns these canvas types and stateful controls.
        self.spectrum = self.make_spectrum()
        self.closeup = self.make_closeup()
        self.tone_view = self.make_tone_view()
        for canvas in (self.spectrum, self.closeup, self.tone_view):
            canvas.setToolTip('')
        self.lower_stack = QStackedWidget()
        self.lower_stack.addWidget(self.closeup)
        self.lower_stack.addWidget(self.tone_view)
        self.display_stack = QStackedWidget()
        spec_pane = QWidget()
        sl = QVBoxLayout(spec_pane)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        sl.addWidget(self.spectrum, stretch=1)
        self.slider_pan = QSlider(Qt.Orientation.Horizontal)
        self.slider_pan.setRange(0, 10000)
        self.slider_pan.setEnabled(False)
        self.slider_pan.setToolTip(self.TIPS['pan'])
        self.slider_pan.hide()
        sl.addWidget(self.slider_pan)
        self.display_stack.addWidget(spec_pane)
        self.display_stack.addWidget(self.lower_stack)

        # Build the processing controls once, then organize them into pages.
        self._legacy_controls_container = self._build_controls_panel()
        self._legacy_controls_container.hide()
        self._legacy_controls_container.widget().layout().removeWidget(self.curve_card)
        self.curve_card.setParent(self.view_frame)

        self.plot_toolbar = QWidget()
        self.plot_toolbar.setObjectName('plot_toolbar')
        toolbar = QHBoxLayout(self.plot_toolbar)
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.setSpacing(6)
        self.view_group = QButtonGroup(self)
        self.btn_view_live = QPushButton('Spectrum')
        self.btn_view_close = QPushButton('Outliers')
        self.btn_view_lines = QPushButton('Spectrogram')
        for button in (self.btn_view_live, self.btn_view_close, self.btn_view_lines):
            button.setObjectName('btn_small')
            button.setCheckable(True)
            self.view_group.addButton(button)
            toolbar.addWidget(button)
        self.btn_view_live.setChecked(True)
        toolbar.addSpacing(6)
        toolbar.addStretch()
        self.btn_zoom_full = QPushButton('Full')
        self.btn_zoom_band = QPushButton('Highs')
        for button, key in [(self.btn_zoom_full, 'zoom_full'), (self.btn_zoom_band, 'zoom_band')]:
            button.setObjectName('btn_small')
            button.setCheckable(True)
            button.setToolTip(self.TIPS[key])
            toolbar.addWidget(button)
        self.btn_zoom_full.setChecked(True)
        self.btn_auto_cuts = QPushButton('Show fixed cuts')
        self.btn_auto_cuts.setObjectName('btn_small')
        self.btn_auto_cuts.setCheckable(True)
        self.btn_auto_cuts.setChecked(self._settings.value('analyzer_auto_cuts', False, type=bool))
        self.spectrum.set_auto_cuts_visible(self.btn_auto_cuts.isChecked())
        self.btn_auto_cuts.setToolTip('Show the fixed notch-filter response. This only changes the graph; filtering is controlled in Settings > Outliers.')
        self.btn_auto_cuts.toggled.connect(self._toggle_auto_cuts)
        toolbar.addWidget(self.btn_auto_cuts)
        self.btn_spectrum_info = QToolButton()
        self.btn_spectrum_info.setIcon(ui_icon('info'))
        self.btn_spectrum_info.setAccessibleName('Spectrum help')
        self.btn_spectrum_info.clicked.connect(self._show_spectrum_help)
        toolbar.addWidget(self.btn_spectrum_info)
        self._restore_analyzer_settings()
        self.btn_curve_add.setText('+ Add curve')
        self.btn_curve_add.setObjectName('btn_primary')
        self.btn_curve_add.setMinimumWidth(148)
        vf.addWidget(self.plot_toolbar)
        self.curve_tools = QWidget()
        self.curve_tools.setObjectName('curve_tools')
        tools = QHBoxLayout(self.curve_tools)
        tools.setContentsMargins(8, 6, 8, 6)
        tools.setSpacing(6)
        self.combo_edit_mode.setFixedWidth(174)
        self.combo_edit_mode.setAccessibleName('Processing mode')
        tools.addWidget(self.combo_edit_mode)
        self.combo_quick_preset = QComboBox()
        self.combo_quick_preset.setMinimumWidth(156)
        self.combo_quick_preset.setAccessibleName('Cleaning preset')
        self.combo_quick_preset.setToolTip('Gentle, Normal or Strong automatic cleaning. Saved presets are also available here; manage them in Settings > Processing.')
        self.combo_quick_preset.activated.connect(self._activate_quick_preset)
        tools.addWidget(self.combo_quick_preset)
        self.curve_actions = QWidget()
        curve_actions_layout = QHBoxLayout(self.curve_actions)
        curve_actions_layout.setContentsMargins(0, 0, 0, 0)
        curve_actions_layout.setSpacing(6)
        tools.addWidget(self.curve_actions, stretch=1)
        tools.addStretch(0)
        tools = curve_actions_layout
        tools.addWidget(self.btn_curve_add)
        self.btn_auto_curves = QToolButton()
        self.btn_auto_curves.setText('Auto curves')
        self.btn_auto_curves.setMinimumWidth(132)
        self.btn_auto_curves.setStyleSheet('QToolButton {padding-right:28px;}')
        self.btn_auto_curves.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        auto_menu = QMenu(self.btn_auto_curves)
        auto_menu.addSection('Minimum suggested width')
        self.auto_width_actions = []
        for width in (.35, .65, 1., 1.5):
            action = auto_menu.addAction(f'{width:g} octave' + (' · default' if width == .65 else ''))
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=width: self._set_auto_width(value))
            self.auto_width_actions.append((action, width))
        auto_menu.aboutToShow.connect(lambda: [a.setChecked(v == self.fs.auto_min_width) for a, v in self.auto_width_actions])
        self.btn_auto_curves.setMenu(auto_menu)
        self.btn_auto_curves.setToolTip('Find the strongest outliers and create up to six dynamic bands. Existing manual bands are kept.')
        self.btn_auto_curves.clicked.connect(self._request_auto_curves)
        tools.addWidget(self.btn_auto_curves)
        self.btn_select_curves = QPushButton('Select all')
        self.btn_select_curves.setToolTip('Select every curve. Ctrl-click or Shift-click handles or band buttons to select individual curves together.')
        self.btn_select_curves.clicked.connect(self.spectrum.select_all_curves)
        tools.addWidget(self.btn_select_curves)
        band_scroll = QScrollArea()
        self.band_scroll = band_scroll
        band_scroll.setWidgetResizable(True)
        band_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        band_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        band_scroll.setFixedHeight(34)
        band_body = QWidget()
        self.band_chips = QHBoxLayout(band_body)
        self.band_chips.setContentsMargins(0, 0, 0, 0)
        self.band_chips.setSpacing(5)
        band_scroll.setWidget(band_body)
        tools.addWidget(band_scroll, stretch=1)
        self.curve_legend = QLabel('Fill: maximum  /  Line: current EQ')
        self.curve_legend.setStyleSheet('color:#a8a8a8; font-size:12px;')
        self.curve_legend.setToolTip('Colored band regions are the maximum permitted EQ. The yellow curve is the response used at the playhead, recorded during audio processing.')
        tools.addWidget(self.curve_legend)
        vf.addWidget(self.curve_tools)
        self.analysis_header = QWidget()
        ah = QHBoxLayout(self.analysis_header)
        ah.setContentsMargins(12, 7, 12, 7)
        self.analysis_legend = QLabel()
        self.analysis_legend.setStyleSheet('font-size:13px; color:#bbbbbb;')
        self.analysis_position = QLabel()
        self.analysis_position.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ah.addWidget(self.analysis_legend, stretch=1)
        ah.addWidget(self.analysis_position)
        self.analysis_header.hide()
        vf.addWidget(self.analysis_header)
        vf.addWidget(self.display_stack, stretch=1)
        cl.addWidget(self.view_frame, stretch=1)
        cl.addWidget(self._build_transport())
        self.lbl_steps = QLabel()  # Retained for workflow state; no instruction banner.
        self.lbl_steps.hide()
        self.lbl_busy = QLabel(self.view_frame)
        self.lbl_busy.setObjectName('lbl_busy')
        self.lbl_busy.hide()
        self.view_frame.installEventFilter(self)
        work.addWidget(center, stretch=1)
        self.inspector = self._build_inspector()
        work.addWidget(self.inspector)
        self.inspector.hide()
        self.workspace_splitter.addWidget(workspace)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([280, 1300])
        outer.addWidget(self.workspace_splitter, stretch=1)

        self.btn_save = QToolButton()
        self.btn_save.setIcon(ui_icon('save'))
        self.btn_save.setToolTip('Save audio')
        self.btn_save.setAccessibleName('Save audio')
        self.btn_save.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self.btn_save)
        self.save_current_action = export_menu.addAction('Save cleaned audio', self.btn_export.click)
        export_menu.addAction('Save removed audio', self.btn_export_diff.click)
        export_menu.addAction('Batch export…', self.btn_batch.click)
        export_menu.aboutToShow.connect(lambda: [action.setEnabled(button.isEnabled()) for action, button in
            zip(export_menu.actions(), (self.btn_export, self.btn_export_diff, self.btn_batch))])
        self.btn_save.setMenu(export_menu)
        header.addWidget(self.btn_save)
        header.addWidget(self.btn_more)
        for container in (self.app_header, self.plot_toolbar, self.curve_tools):
            for control in container.findChildren((QPushButton, QToolButton, QComboBox)):
                self._style_toolbar_control(control)
        self._tracks_width = self._settings.value('sidebar_width', 260, type=int)
        self._set_tracks_collapsed(self._settings.value('sidebar_collapsed', False, type=bool))
        self.btn_inspector_outliers.clicked.connect(lambda: self._show_inspector('outliers', toggle=True))
        self.btn_inspector_history.clicked.connect(lambda: self._show_inspector('history', toggle=True))
        self.display_stack.currentChanged.connect(lambda _: self._sync_view_controls())
        self.workspace_splitter.splitterMoved.connect(lambda *_: self._position_curve_editor())
        self.lbl_device = QLabel('CPU')
        self.statusBar().addPermanentWidget(self.lbl_device)
        self.statusBar().showMessage('+ Add curve creates a bell · arrow chooses a shape · Ctrl/Shift-click selects several · wheel changes width')
        self._update_view_mode()

    def _build_tracks_panel(self):
        panel = QWidget()
        panel.setObjectName('tracks_panel')
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 6, 5, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        row = QHBoxLayout()
        self.lbl_track_count = QLabel('TRACKS')
        self.lbl_track_count.setObjectName('lbl_section')
        row.addWidget(self.lbl_track_count)
        row.addStretch()
        self.btn_add_tracks = QToolButton()
        self.btn_add_tracks.setIcon(ui_icon('add'))
        self.btn_add_tracks.setToolTip('Add audio files')
        self.btn_add_tracks.setAccessibleName('Add audio files')
        self.btn_add_tracks.clicked.connect(self._browse_files)
        row.addWidget(self.btn_add_tracks)
        self.btn_tracks = QToolButton()
        self.btn_tracks.setToolTip('Collapse track browser')
        self.btn_tracks.clicked.connect(lambda: self._set_tracks_collapsed(not self._tracks_collapsed))
        row.addWidget(self.btn_tracks)
        for button in (self.btn_add_tracks, self.btn_tracks):
            button.setFixedSize(34, 34)
            button.setStyleSheet('QToolButton {padding:0;}')
        layout.addLayout(row)
        self.tracks_body = QWidget()
        tracks = QVBoxLayout(self.tracks_body)
        tracks.setContentsMargins(0, 0, 0, 0)
        tracks.setSpacing(4)
        self.file_list = QListWidget()
        self.file_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.file_list.setItemDelegate(TrackDelegate(self.file_list))
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.file_list.setMouseTracking(True)
        self.file_list.setToolTip(self.TIPS['tracks'])
        tracks.addWidget(self.file_list, stretch=1)
        self.lbl_track = QLabel('Drop audio files here\nor use + to add a track.')
        self.lbl_track.setWordWrap(True)
        self.lbl_track.setStyleSheet('color:#aaaaaa; font-size:15px; padding:3px;')
        tracks.addWidget(self.lbl_track)
        layout.addWidget(self.tracks_body, stretch=1)
        return panel

    @staticmethod
    def _style_toolbar_control(control):
        control.setFixedHeight(34)
        control.setProperty('toolbarControl', True)
        if isinstance(control, QToolButton) and not control.text():
            control.setFixedWidth(46 if control.menu() else 34)
            control.setIconSize(QSize(20, 20))

    def _set_tracks_collapsed(self, collapsed):
        if collapsed and not getattr(self, '_tracks_collapsed', True):
            self._tracks_width = self.tracks_panel.width()
        self._tracks_collapsed = bool(collapsed)
        self.tracks_body.setVisible(not collapsed)
        self.lbl_track_count.setVisible(not collapsed)
        self.btn_add_tracks.setVisible(not collapsed)
        self.tracks_panel.setMinimumWidth(44 if collapsed else 210)
        self.tracks_panel.setMaximumWidth(44 if collapsed else 520)
        self.btn_tracks.setIcon(ui_icon('expand' if collapsed else 'collapse'))
        title = 'Expand track browser' if collapsed else 'Collapse track browser'
        self.btn_tracks.setToolTip(title)
        self.btn_tracks.setAccessibleName(title)
        width = 44 if collapsed else max(210, min(520, self._tracks_width))
        self.workspace_splitter.setSizes([width, max(540, self.width()-width)])
        self._settings.setValue('sidebar_collapsed', bool(collapsed))

    def _set_track_path(self, path):
        full_path = str(Path(path).resolve()) if path else ''
        self.track_title.setText(full_path)
        self.track_title.setToolTip(full_path)
        self.track_title.setCursorPosition(len(full_path))
        self.btn_copy_path.setEnabled(bool(full_path))

    def _set_color_profile(self, key):
        self.color_profile = key
        self._settings.setValue('color_profile', key)
        self._apply_appearance()

    def _set_spectrum_accent(self, key):
        self.spectrum_accent = key
        self._settings.setValue('spectrum_accent', key)
        self._apply_appearance()

    def _apply_appearance(self):
        self.setStyleSheet(studio_stylesheet(self.color_profile))
        background = COLOR_PROFILES[self.color_profile][3]
        self.spectrum.set_colors(background, SPECTRUM_ACCENTS[self.spectrum_accent][1])
        for canvas in (self.closeup, self.tone_view):
            canvas.fig.set_facecolor(background)
            for axis in canvas.fig.axes:
                axis.set_facecolor(background)
            canvas.draw_idle()

    def _build_curve_controls(self, layout):
        self.curve_card = QFrame()
        self.curve_card.setObjectName('curve_editor')
        self.curve_card.setMinimumWidth(510)
        body = QVBoxLayout(self.curve_card)
        body.setContentsMargins(14, 4, 14, 4)
        body.setSpacing(3)
        self._editor_position = None
        self._editor_docked = False
        saved_position = self._settings.value('eq_editor_position', [], type=list)
        if len(saved_position) == 2:
            try:
                self._editor_position = tuple(max(0., min(1., float(value))) for value in saved_position)
            except (TypeError, ValueError):
                pass
        self.curve_drag_handle = PanelDragHandle(self.curve_card)
        self.curve_drag_handle.moved.connect(self._drag_curve_editor)
        body.addWidget(self.curve_drag_handle)
        self.curve_selection_label = QLabel()
        self.curve_selection_label.setStyleSheet('color:#c5c5c5; font-size:13px;')
        self.curve_selection_label.setFixedHeight(16)
        body.addWidget(self.curve_selection_label)
        self.combo_edit_mode = QComboBox()
        self.combo_edit_mode.addItem('EQ curves', 'curves')
        self.combo_edit_mode.addItem('Soften highs', 'range')
        self.combo_edit_mode.setToolTip('EQ curves: target individual frequency bands. Soften highs: reduce a broad high-frequency region with adjustable start, end and depth.')
        row = QHBoxLayout()
        row.setSpacing(8)
        self.combo_eq_bands = QComboBox()
        self.combo_eq_bands.setMinimumContentsLength(7)
        self.combo_eq_bands.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.combo_eq_bands, stretch=1)
        self.combo_eq_type = QComboBox()
        self.combo_eq_type.setMinimumContentsLength(6)
        self.combo_eq_type.setIconSize(QSize(32, 16))
        for title, key in self.CURVE_TYPES:
            self.combo_eq_type.addItem(self.curve_icon(key), title, key)
        row.addWidget(self.combo_eq_type, stretch=1)
        self.chk_curve_on = QCheckBox('On')
        row.addWidget(self.chk_curve_on)
        self.btn_curve_remove = QPushButton('Delete')
        self.btn_curve_remove.setObjectName('btn_small')
        row.addWidget(self.btn_curve_remove)
        close = QPushButton('×')
        close.setObjectName('btn_small')
        close.setFixedWidth(25)
        close.setToolTip('Deselect the band')
        close.clicked.connect(lambda: self._select_curve(-1))
        row.addWidget(close)
        body.addLayout(row)
        self.btn_curve_add = QToolButton()
        self.btn_curve_add.setText('+ Add curve')
        self.btn_curve_add.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_curve_add.clicked.connect(lambda: self._add_curve(self.spectrum.view_center(), 'bell'))
        self.btn_curve_add.setObjectName('btn_small')
        menu = QMenu(self.btn_curve_add)
        for title, key in self.CURVE_TYPES:
            action = menu.addAction(self.curve_icon(key), title)
            action.triggered.connect(lambda _checked=False, k=key: self._add_curve(self.spectrum.view_center(), k))
        self.btn_curve_add.setMenu(menu)
        self.btn_curve_add.setToolTip('Click + Add curve to create a bell. Use the arrow for eight curve shapes. You can also double-click anywhere on the graph.')
        self.curve_body = QWidget()
        knobs = QHBoxLayout(self.curve_body)
        knobs.setContentsMargins(0, 0, 0, 0)
        knobs.setSpacing(8)
        self.curve_controls, self.rotary_controls = {}, {}
        specs = [
            ('freq', 'Frequency', 20, 96000, 10, 0, ' Hz', 1000, True, '#a3a3a3'),
            ('gain', 'Max gain', -60, 24, .5, 1, ' dB', -6, False, '#bdbdbd'),
            ('width', 'Width', .02, 10, .05, 2, ' oct', 1, True, '#a3a3a3'),
            ('plateau', 'Plateau', 0, 95, 5, 0, ' %', 0, False, '#a3a3a3'),
            ('scale', 'Scale both', 10, 300, 5, 0, ' %', 100, True, '#a3a3a3'),
        ]
        tooltips = {
            'freq': 'Frequency: drag the band left or right, turn the dial, or type an exact frequency.',
            'gain': 'Maximum gain / reduction depth in dB. Changes height only. Scale both multiplies this depth and the width.',
            'width': 'Overall width in octaves. Scroll over the graph, or right-drag horizontally.',
            'plateau': 'Flat centre width, for Bell, Notch and Band pass. Right-drag vertically.',
            'scale': 'Multiplies BOTH width and depth. 150% makes the curve 1.5 times wider AND deeper. Max gain changes depth alone. Alt+wheel on the graph.',
        }
        for key, title, lo, hi, step, decimals, suffix, default, logarithmic, color in specs:
            control = RotaryControl(title, lo, hi, step, decimals, suffix, default, logarithmic, color)
            control.setToolTip(tooltips[key])
            self.rotary_controls[key] = control
            self.curve_controls[key] = control.spin
            knobs.addWidget(control, stretch=1)
        body.addWidget(self.curve_body)
        self.curve_effective_label = QLabel()
        self.curve_effective_label.setStyleSheet('color:#9e9e9e; font-size:12px;')
        self.curve_effective_label.setFixedHeight(16)
        body.addWidget(self.curve_effective_label)
        self.dynamics_row = QWidget()
        dynamics = QHBoxLayout(self.dynamics_row)
        dynamics.setContentsMargins(0, 3, 0, 0)
        dynamics.setSpacing(8)
        self.chk_curve_dynamic = QCheckBox('Dynamic')
        self.chk_curve_dynamic.setChecked(True)
        self.chk_curve_dynamic.setToolTip('Evaluate this band for every audio frame. Uncheck to apply its fixed maximum gain.')
        dynamics.addWidget(self.chk_curve_dynamic)
        self.dynamic_controls = {}
        for key, title, lo, hi, step, suffix in [
            ('threshold', 'Threshold', -120, 0, 1, ' dBFS'),
            ('ratio', 'Ratio', 1, 20, .5, ':1'),
            ('attack', 'Attack', 0, 500, 1, ' ms'),
            ('release', 'Release', 0, 3000, 10, ' ms')]:
            control = QWidget()
            column = QVBoxLayout(control)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            label = QLabel(title)
            label.setStyleSheet('color:#9b9b9b; font-size:11px;')
            column.addWidget(label)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1 if key == 'ratio' else 0)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(65)
            spin.setToolTip({'threshold':'Level inside this curve that starts the adjustment. Below this level the band releases to zero.',
                            'ratio':'How strongly levels above threshold increase the adjustment. 4:1 removes 3 dB for every 4 dB above threshold.',
                            'attack':'How quickly the band engages when its level rises.',
                            'release':'How quickly the band returns to zero when its level falls.'}[key])
            self.dynamic_controls[key] = spin
            column.addWidget(spin)
            dynamics.addWidget(control, stretch=1)
        body.addWidget(self.dynamics_row)
        layout.addWidget(self.curve_card)

    def _build_inspector(self):
        inspector = QWidget()
        inspector.setObjectName('inspector')
        inspector.setFixedWidth(324)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(10, 12, 6, 0)
        row = QHBoxLayout()
        self.inspector_title = QLabel('Spectral outliers')
        self.inspector_title.setObjectName('section_title')
        row.addWidget(self.inspector_title)
        row.addStretch()
        close = QPushButton('×')
        close.setObjectName('btn_small')
        close.setFixedWidth(26)
        close.clicked.connect(self._hide_inspector)
        row.addWidget(close)
        layout.addLayout(row)
        self.inspector_stack = QStackedWidget()
        self.inspector_pages = {}
        cards = self.inspector_cards
        for key, group in [('outliers', ['strength', 'tones', 'result']),
                            ('history', ['history']), ('advanced', ['shelf', 'advanced']),
                            ('export', ['save'])]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 5, 8)
            page_layout.setSpacing(10)
            for name in group:
                page_layout.addWidget(cards[name])
            page_layout.addStretch()
            scroll.setWidget(page)
            self.inspector_pages[key] = self.inspector_stack.addWidget(scroll)
        layout.addWidget(self.inspector_stack, stretch=1)
        self._inspector_key = None
        return inspector

    def _show_inspector(self, key, toggle=False):
        if toggle and self.inspector.isVisible() and self._inspector_key == key:
            self._hide_inspector()
            return
        self._inspector_key = key
        self.inspector_title.setText({'outliers':'Spectral outliers', 'history':'Versions & history',
                                     'advanced':'Audio processing', 'export':'Export audio'}[key])
        self.inspector_stack.setCurrentIndex(self.inspector_pages[key])
        self.inspector.show()
        self.btn_inspector_outliers.setChecked(key == 'outliers')
        self.btn_inspector_history.setChecked(key == 'history')

    def _hide_inspector(self):
        self.inspector.hide()
        self._inspector_key = None
        self.btn_inspector_outliers.setChecked(False)
        self.btn_inspector_history.setChecked(False)

    def _refresh_outlier_badge(self):
        if hasattr(self, 'btn_inspector_outliers'):
            count = sum(n.get('on', True) for n in self.fs.notches)
            self.btn_inspector_outliers.setText(f'Outliers · {count}' if count else 'Outliers')

    def _sync_editor_visibility(self):
        if not hasattr(self, 'curve_card') or not hasattr(self, 'display_stack'):
            return
        visible = (self.fs.edit_mode == 'curves' and self.spectrum.selected_curve >= 0
                   and self.display_stack.currentIndex() == 0)
        self.curve_card.setVisible(visible)
        if visible:
            self._position_curve_editor()
            self.curve_card.raise_()

    def _position_curve_editor(self):
        if not hasattr(self, 'curve_card'):
            return
        # A docked panel preserves the whole graph when the inspector is open
        # on a small window. Larger windows use the reference's floating panel.
        dock = self.view_frame.width() < 850 and self._editor_position is None
        compact = self.view_frame.width() < 850
        self.combo_edit_mode.setFixedWidth(148 if compact else 174)
        self.combo_quick_preset.setFixedWidth(136 if compact else 156)
        self.btn_curve_add.setMinimumWidth(118 if compact else 148)
        self.btn_auto_curves.setMinimumWidth(112 if compact else 132)
        self.btn_select_curves.setVisible(not compact)
        self.band_scroll.setVisible(not compact)
        self.curve_effective_label.setVisible(not dock)
        self.curve_selection_label.setVisible(len(self.spectrum.selected_curves) > 1)
        was_docked = getattr(self, '_editor_docked', False)
        if dock != was_docked:
            self._editor_docked = dock
            if dock:
                self.view_frame.layout().addWidget(self.curve_card, alignment=Qt.AlignmentFlag.AlignHCenter)
            else:
                self.view_frame.layout().removeWidget(self.curve_card)
                self.curve_card.setParent(self.view_frame)
            visible = (self.fs.edit_mode == 'curves' and self.spectrum.selected_curve >= 0
                       and self.display_stack.currentIndex() == 0)
            self.curve_card.setVisible(visible)
        width = min(710, int(self.view_frame.width()*.65), self.view_frame.width()-36)
        self.curve_card.setFixedWidth(max(510, width))
        self.curve_card.adjustSize()
        if dock:
            return
        if self._editor_position is not None:
            left, top, right, bottom = self._curve_editor_bounds()
            x, y = self._editor_position
            self.curve_card.move(round(left+x*(right-left)), round(top+y*(bottom-top)))
            return
        x = max(12, (self.view_frame.width()-self.curve_card.width())//2)
        y = max(64, self.view_frame.height()-self.curve_card.height()-55)
        index = self.spectrum.selected_curve
        if 0 <= index < len(self.spectrum.eq_curves):
            c = self.spectrum.eq_curves[index]
            px, py = self.spectrum.eq_ax.transData.transform((c['freq'], self.spectrum._display_handle_gain(c)))
            ratio = self.spectrum.device_pixel_ratio
            px, py = px/ratio, py/ratio
            canvas_top = self.spectrum.mapTo(self.view_frame, QPoint(0, 0))
            px += canvas_top.x()
            py = canvas_top.y() + self.spectrum.height() - py
            if y-40 < py < y+self.curve_card.height()+30:
                if x-42 < px < x+self.curve_card.width()+42:
                    x = 12 if px > self.view_frame.width()/2 else self.view_frame.width()-self.curve_card.width()-12
                if x-42 < px < x+self.curve_card.width()+42:
                    y = self.plot_toolbar.height() + 58
        self.curve_card.move(x, y)

    def _curve_editor_bounds(self):
        right = max(8, self.view_frame.width()-self.curve_card.width()-8)
        bottom = max(8, self.view_frame.height()-self.curve_card.height()-8)
        top = min(bottom, self.plot_toolbar.height()+self.curve_tools.height()+4)
        return 8, top, right, bottom

    def _drag_curve_editor(self, global_position):
        if getattr(self, '_editor_docked', False):
            self.view_frame.layout().removeWidget(self.curve_card)
            self.curve_card.setParent(self.view_frame)
            self._editor_docked = False
            self.curve_card.show()
        point = self.view_frame.mapFromGlobal(global_position)
        left, top, right, bottom = self._curve_editor_bounds()
        x, y = max(left, min(right, point.x())), max(top, min(bottom, point.y()))
        self._editor_position = ((x-left)/max(1, right-left), (y-top)/max(1, bottom-top))
        self.curve_card.move(x, y)
        self.curve_card.raise_()
        self._settings.setValue('eq_editor_position', list(self._editor_position))

    def _select_band_chip(self, index):
        modifiers = QApplication.keyboardModifiers()
        self._select_curve(index, bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)))

    def _refresh_band_chips(self):
        if not hasattr(self, 'band_chips'):
            return
        buttons = getattr(self, '_band_buttons', [])
        if len(buttons) != len(self.fs.eq_curves):
            while self.band_chips.count():
                item = self.band_chips.takeAt(0)
                if item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()
            buttons = []
            for index in range(len(self.fs.eq_curves)):
                button = QPushButton()
                button.setObjectName('btn_small')
                button.setCheckable(True)
                button.clicked.connect(lambda checked=False, i=index: self._select_band_chip(i))
                self._style_toolbar_control(button)
                self.band_chips.addWidget(button)
                buttons.append(button)
            if not buttons:
                hint = QLabel('Click + Add curve, or double-click the graph')
                hint.setStyleSheet('color:#ababab; font-size:13px;')
                self.band_chips.addWidget(hint)
            self.band_chips.addStretch()
            self._band_buttons = buttons
        for index, (button, curve) in enumerate(zip(buttons, self.fs.eq_curves)):
            hz = curve['freq']
            label = f'{hz/1000:.1f}k' if hz >= 1000 else f'{hz:.0f}'
            button.setText(f'{index+1}  {label}')
            button.setChecked(index in self.spectrum.selected_curves)
            button.setToolTip(f"{curve['type'].replace('_', ' ').title()} at {hz:,.0f} Hz. Ctrl/Shift-click to select together.")
        count = len(self.spectrum.selected_curves)
        self.btn_select_curves.setText(f'All ({count})' if count > 1 else 'Select all')
        self.curve_legend.setVisible(self.view_frame.width() > 1050)

    def _toggle_auto_cuts(self, on):
        self.spectrum.set_auto_cuts_visible(on)
        self._settings.setValue('analyzer_auto_cuts', on)
        self._refresh_filter_response()

    def _refresh_quick_presets(self):
        if not hasattr(self, 'combo_quick_preset'):
            return
        combo = self.combo_quick_preset
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('Custom preset', 'custom')
        combo.model().item(0).setEnabled(False)
        for key, label in (('gentle', 'Gentle'), ('normal', 'Normal'), ('strong', 'Strong')):
            combo.addItem(label + ' preset', key)
        for name in sorted(getattr(self, '_user_preset_map', {}), key=str.lower):
            combo.addItem(name, 'user:' + name)
        combo.setCurrentIndex(max(0, combo.findData(self.fs.preset)))
        combo.blockSignals(False)

    def _activate_quick_preset(self, index):
        key = self.combo_quick_preset.itemData(index)
        if key in self.preset_buttons:
            self._on_preset_clicked(self.preset_buttons[key])
        elif key and key.startswith('user:'):
            self._on_user_preset_activated(self.combo_user_presets.findData(key[5:]))

    def _sync_view_controls(self):
        live = self.display_stack.currentIndex() == 0
        self.curve_actions.setVisible(live and self.fs.edit_mode == 'curves')
        self.btn_auto_cuts.setVisible(live)
        self.analysis_header.setVisible(not live)
        spectrogram = self.lower_stack.currentIndex() == 1
        for button in (self.btn_zoom_full, self.btn_zoom_band):
            button.setVisible(live)
        if not live:
            self.analysis_legend.setText(
                'Brightness: outlier strength  |  Click: seek  |  Shift-drag: loop'
                if spectrogram else 'Track: original (blue), cleaned (green)  |  White: now  |  Dashed: detection threshold')
        self._sync_editor_visibility()
        self._update_analysis_position()
        if not live:
            self.tone_view.set_playhead(self._cb_frame/self.sample_rate)
        self._update_live_spectrum()

    def _update_analysis_position(self):
        if not hasattr(self, 'analysis_position'):
            return
        label = 'Removed' if self._solo_residual else self._slot_label()[0]
        self.analysis_position.setText(f'{label}  |  {self.lbl_time_cur.text()}')

    def _set_auto_width(self, value):
        self.fs.auto_min_width = value
        self._save_filter_settings()
        self._request_auto_curves()

    def _restore_analyzer_settings(self):
        self.spectrum.peak_decay_db = self._settings.value('analyzer_peak_decay', .5, type=float)
        smoothing = self._settings.value('analyzer_smoothing', 1/12, type=float)
        gain_range = self._settings.value('analyzer_gain_range', 24, type=int)
        self.spectrum.set_smoothing(smoothing if smoothing in (1/48, 1/24, 1/12) else 1/12)
        self.spectrum.set_gain_range(gain_range if gain_range in (12, 24, 48, 72) else 24)

    def _set_peak_decay(self, value):
        self.spectrum.peak_decay_db = value
        self._settings.setValue('analyzer_peak_decay', value)

    def _set_analyzer_smoothing(self, value):
        self.spectrum.set_smoothing(value)
        self._settings.setValue('analyzer_smoothing', value)
        self._update_live_spectrum()

    def _set_analyzer_range(self, value):
        self.spectrum.set_gain_range(value)
        self._settings.setValue('analyzer_gain_range', value)

    def _build_transport(self):
        frame = QFrame()
        frame.setObjectName('transport')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(4)
        row = QHBoxLayout()
        self.lbl_time_cur, self.lbl_time_tot = QLabel('0:00'), QLabel('0:00')
        for label in (self.lbl_time_cur, self.lbl_time_tot):
            label.setFixedWidth(38)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('color:#ababab; font-size:13px;')
        self.slider_pos = self.make_click_slider(Qt.Orientation.Horizontal)
        self.slider_pos.setRange(0, 10000)
        self.slider_pos.setToolTip(self.TIPS['position'])
        row.addWidget(self.lbl_time_cur)
        row.addWidget(self.slider_pos, stretch=1)
        row.addWidget(self.lbl_time_tot)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(7)
        def group(title):
            wrapper = QWidget()
            column = QVBoxLayout(wrapper)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(0)
            box = QFrame()
            box.setObjectName('transport_group')
            box.setAccessibleName(title)
            controls = QHBoxLayout(box)
            controls.setContentsMargins(5, 5, 5, 5)
            controls.setSpacing(6)
            column.addWidget(box)
            row.addWidget(wrapper)
            return controls
        playback = group('PLAYBACK')
        self._icon_play = self.transport_icon('play', '#eeeeee', 26)
        self._icon_pause = self.transport_icon('pause', '#eeeeee', 26)
        self.btn_play = QPushButton('')
        self.btn_play.setIcon(self._icon_play)
        self.btn_play.setObjectName('btn_primary')
        self.btn_stop = QPushButton('')
        self.btn_stop.setIcon(self.transport_icon('stop', '#eeeeee', 26))
        self.btn_play.setAccessibleName('Play')
        self.btn_stop.setAccessibleName('Stop')
        for button in (self.btn_play, self.btn_stop):
            button.setFixedWidth(66)
            button.setIconSize(QSize(26, 26))
            playback.addWidget(button)
        comparison = group('COMPARE')
        self.btn_ab = QPushButton('B  Cleaned')
        self.btn_ab.setObjectName('btn_ab_b')
        self.btn_solo = QPushButton('Residual')
        self.btn_solo.setObjectName('btn_ab_c')
        self.btn_solo.setCheckable(True)
        comparison.addWidget(self.btn_ab)
        comparison.addWidget(self.btn_solo)
        for button, key in [(self.btn_play, 'play'), (self.btn_stop, 'stop'),
                             (self.btn_ab, 'ab'), (self.btn_solo, 'solo')]:
            button.setEnabled(False)
            button.setToolTip(self.TIPS[key])
            button.setFixedHeight(40)
        repeat = group('LOOP / SECTION')
        self.btn_loop = QPushButton('Loop')
        self.btn_loop.setObjectName('btn_small')
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(self._loop_on)
        self.btn_loop.setToolTip(self.TIPS['loop'])
        self.btn_loop.setFixedHeight(40)
        repeat.addWidget(self.btn_loop)
        self.btn_loop_sec = QPushButton('Section')
        self.btn_loop_sec.setObjectName('btn_small')
        self.btn_loop_sec.setToolTip(self.TIPS['loop_sec'])
        self.btn_loop_sec.setFixedHeight(40)
        repeat.addWidget(self.btn_loop_sec)
        self.lbl_loop = QLabel('')
        self.lbl_loop.setStyleSheet('color:#999999; font-size:12px;')
        repeat.addWidget(self.lbl_loop)
        row.addStretch()
        volume = QLabel('VOL')
        volume.setStyleSheet('color:#828282; font-size:11px;')
        row.addWidget(volume)
        self.slider_vol = self.make_click_slider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(round(self._cb_volume * 100))
        self.slider_vol.setFixedWidth(74)
        self.slider_vol.setToolTip(self.TIPS['volume'])
        row.addWidget(self.slider_vol)
        layout.addLayout(row)
        return frame
