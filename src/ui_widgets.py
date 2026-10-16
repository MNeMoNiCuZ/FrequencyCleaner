"""Desktop controls and visual theme for the spectrum workspace."""

import math
from pathlib import Path
from PyQt6.QtCore import Qt, QRectF, QSize, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QLinearGradient, QFont, QIcon, QPixmap, QPainterPath
from PyQt6.QtWidgets import (QWidget, QLabel, QDial, QDoubleSpinBox, QVBoxLayout,
                             QStyledItemDelegate, QStyle, QSizePolicy)


COLOR_PROFILES = {
    'dark': ('Dark gray', '#202020', '#262626', '#181818'),
    'graphite': ('Graphite', '#292929', '#303030', '#222222'),
    'black': ('Black', '#151515', '#1d1d1d', '#101010'),
}
SPECTRUM_ACCENTS = {
    'blue': ('Blue', '#68b7e8'), 'teal': ('Teal', '#58c7b2'),
    'amber': ('Amber', '#e3b660'), 'violet': ('Violet', '#b28ad6'),
}


def studio_stylesheet(profile='dark'):
    _, base, panel, canvas = COLOR_PROFILES.get(profile, COLOR_PROFILES['dark'])
    return STUDIO_QSS.replace('@BASE@', base).replace('@PANEL@', panel).replace('@CANVAS@', canvas)


STUDIO_QSS = """
QMainWindow, QWidget { background:@BASE@; color:#e1e1e1; font-family:'Segoe UI'; font-size:14px; }
QLabel { background:transparent; }
QWidget#app_header { background:@PANEL@; border-bottom:1px solid #3b3b3b; }
QWidget#tracks_panel { background:@BASE@; border-right:1px solid #393939; }
QWidget#workspace, QFrame#view_frame { background:@CANVAS@; }
QFrame#view_frame { border:1px solid #383838; border-radius:5px; }
QWidget#curve_tools, QWidget#plot_toolbar { background:@PANEL@; border-bottom:1px solid #383838; }
QFrame#curve_editor { background:@PANEL@; border:1px solid #575757; border-radius:7px; }
QFrame#curve_editor QWidget { background:transparent; }
QWidget#inspector { background:@BASE@; border-left:1px solid #444444; }
QLabel#section_title { font-size:15px; font-weight:600; }
QLabel#lbl_section { color:#ababab; font-size:12px; font-weight:600; }
QLabel#lbl_hint { color:#ababab; font-size:13px; }
QLabel#lbl_big { font-size:21px; font-weight:600; }
QLabel#lbl_metric { font-size:15px; font-weight:600; }
QListWidget { background:transparent; color:#dedede; border:none; outline:0; }
QListWidget::item { border:none; padding:0; }
QListWidget::item:selected { background:transparent; }
QPushButton, QToolButton { background:#303030; color:#dedede; border:1px solid #4b4b4b;
    border-radius:4px; padding:5px 10px; font-size:14px; }
QPushButton:hover, QToolButton:hover { background:#3d3d3d; border-color:#777777; color:#ffffff; }
QPushButton:pressed, QToolButton:pressed { background:#484848; }
QPushButton:disabled, QToolButton:disabled { color:#777777; background:#292929; border-color:#3b3b3b; }
QPushButton:checked, QToolButton:checked { color:#141414; background:#d0d0d0; border-color:#ffffff; font-weight:600; }
QPushButton:checked:hover, QToolButton:checked:hover { background:#e4e4e4; color:#111111; }
QPushButton#btn_ab_a { background:#284e6b; border:2px solid #6cb8ed; color:#ffffff; font-weight:600; }
QPushButton#btn_ab_b { background:#285341; border:2px solid #70d4a7; color:#ffffff; font-weight:600; }
QPushButton#btn_ab_x { background:#65512b; border:2px solid #ebc16d; color:#ffffff; font-weight:600; }
QPushButton#btn_ab_c:checked { background:#ebc16d; color:#171717; border:2px solid #ffe0a2; }
QPushButton#btn_ab_a:disabled, QPushButton#btn_ab_b:disabled, QPushButton#btn_ab_x:disabled { background:#292929; color:#777777; border:1px solid #444444; }
QPushButton#btn_primary { font-weight:600; }
QPushButton#btn_small { padding:5px 9px; }
QPushButton#btn_link { border:none; background:transparent; color:#d0d0d0; padding:4px; text-align:left; }
QPushButton#btn_preset:checked { background:#d0d0d0; color:#141414; border-color:#ffffff; }
QPushButton#btn_card_head { border:none; background:transparent; text-align:left; padding:2px 0; font-weight:600; }
QFrame#card { background:transparent; border:none; border-bottom:1px solid #444444; border-radius:0; }
QComboBox { background:#303030; color:#dedede; border:1px solid #666666; border-radius:4px; padding:5px 30px 5px 8px; }
QComboBox::drop-down { border-left:1px solid #666666; width:26px; background:#404040; }
QComboBox::down-arrow { image:url(@DOWN@); width:12px; height:12px; }
QComboBox QAbstractItemView { background:#303030; color:#eeeeee; border:1px solid #666666; selection-background-color:#505050; }
QSpinBox, QDoubleSpinBox, QLineEdit { background:@CANVAS@; color:#e2e2e2; border:1px solid #4b4b4b; border-radius:4px; padding:5px 7px; }
QLineEdit#track_path { background:@CANVAS@; border:1px solid #404040; selection-background-color:#555555; font-size:14px; }
QDoubleSpinBox#knob_value { font-size:15px; font-weight:600; padding:4px; }
QDoubleSpinBox:disabled, QSpinBox:disabled { color:#777777; border-color:#3b3b3b; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width:0; border:none; }
QSlider::groove:horizontal { height:4px; background:#494949; border-radius:2px; }
QSlider::sub-page:horizontal { background:#999999; border-radius:2px; }
QSlider::handle:horizontal { background:#dedede; width:11px; margin:-4px 0; border-radius:5px; }
QCheckBox, QRadioButton { spacing:7px; color:#cccccc; }
QCheckBox::indicator { width:15px; height:15px; border:1px solid #777777; border-radius:3px; background:#222222; }
QCheckBox::indicator:checked { background:#cccccc; border-color:#dddddd; image:url(@CHECK@); }
QCheckBox::indicator:disabled { background:#303030; border-color:#444444; }
QScrollArea { background:transparent; border:none; }
QScrollBar:vertical { background:#242424; width:7px; margin:0; }
QScrollBar::handle:vertical { background:#606060; border-radius:3px; min-height:24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QSplitter::handle { background:#393939; width:4px; }
QSplitter::handle:hover { background:#777777; }
QFrame#transport { background:@PANEL@; border:1px solid #404040; border-radius:5px; }
QFrame#transport_group { background:transparent; border:1px solid #505050; border-radius:5px; }
QFrame#transport_group QPushButton { padding:4px 10px; }
QStatusBar { background:@BASE@; color:#aaaaaa; border-top:1px solid #393939; font-size:12px; }
QStatusBar::item { border:none; }
QLabel#lbl_busy { color:#eeeeee; background:#353535; border:1px solid #777777; border-radius:5px; padding:5px 12px; font-size:13px; }
QMenu { background:@PANEL@; color:#dddddd; border:1px solid #606060; padding:5px; font-size:14px; }
QMenu::item { padding:7px 24px 7px 12px; }
QMenu::item:selected { background:#494949; border-radius:3px; }
QMenu::item:disabled { color:#777777; }
QMenu::separator { height:1px; background:#494949; margin:5px; }
QTableWidget { background:@CANVAS@; border:1px solid #444444; gridline-color:#383838; }
QTableWidget::item:selected { background:#484848; }
QHeaderView::section { background:#333333; color:#cccccc; border:none; padding:5px; }
QProgressBar { background:#333333; border:none; max-height:3px; color:transparent; }
QProgressBar::chunk { background:#bbbbbb; }
QToolTip { background:#303030; color:#eeeeee; border:1px solid #777777; padding:8px; }
QTabWidget::pane { border:1px solid #444444; border-radius:4px; }
QTabBar::tab { background:#303030; color:#cccccc; padding:9px 14px; border:1px solid #444444; }
QTabBar::tab:selected { background:#484848; color:#ffffff; }
QToolButton::menu-button { width:23px; border-left:1px solid #505050; }
QToolButton[toolbarControl="true"], QPushButton[toolbarControl="true"], QComboBox[toolbarControl="true"] { font-size:14px; }
""".replace('@CHECK@', (Path(__file__).resolve().parent / 'icons' / 'check.svg').as_posix()).replace('@DOWN@', (Path(__file__).resolve().parent / 'icons' / 'chevron-down.svg').as_posix())


def ui_icon(kind, color='#d8d8d8'):
    """Small vector icons with consistent strokes, independent of installed fonts."""
    pm = QPixmap(40, 40)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2, 2)
    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    if kind == 'info':
        p.drawEllipse(QRectF(3, 3, 14, 14))
        p.drawLine(10, 9, 10, 14)
        p.drawPoint(10, 6)
    elif kind == 'settings':
        p.drawEllipse(QRectF(5, 5, 10, 10))
        p.drawEllipse(QRectF(8, 8, 4, 4))
        for degrees in range(0, 360, 45):
            angle = math.radians(degrees)
            p.drawLine(QPointF(10+5*math.cos(angle), 10+5*math.sin(angle)),
                       QPointF(10+8*math.cos(angle), 10+8*math.sin(angle)))
    elif kind == 'save':
        p.drawRoundedRect(QRectF(3, 3, 14, 14), 1, 1)
        p.drawRect(QRectF(6, 3, 7, 5))
        p.drawRect(QRectF(6, 11, 8, 6))
    elif kind == 'copy':
        p.drawRoundedRect(QRectF(7, 7, 10, 10), 1, 1)
        path = QPainterPath(QPointF(4, 13))
        path.lineTo(3, 13)
        path.lineTo(3, 3)
        path.lineTo(13, 3)
        path.lineTo(13, 4)
        p.drawPath(path)
    elif kind == 'add':
        p.drawLine(4, 10, 16, 10)
        p.drawLine(10, 4, 10, 16)
    else:
        p.drawRoundedRect(QRectF(2, 3, 16, 14), 1, 1)
        p.drawLine(7, 3, 7, 17)
        x = 10 if kind == 'collapse' else 14
        end = 13 if kind == 'collapse' else 11
        p.drawLine(x, 10, end, 7)
        p.drawLine(x, 10, end, 13)
    p.end()
    pm.setDevicePixelRatio(2.)
    return QIcon(pm)



class PanelDragHandle(QLabel):
    moved = pyqtSignal(object)

    def __init__(self, panel):
        super().__init__('EQ band · drag to move', panel)
        self._panel = panel
        self._offset = None
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAccessibleName('Drag EQ panel')
        self.setContentsMargins(16, 0, 0, 0)
        self.setStyleSheet('color:#aaaaaa; font-size:12px; background:transparent;')

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor('#aaaaaa'), 2))
        for x in (3, 8):
            for y in (5, 9, 13):
                painter.drawPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._offset = event.globalPosition().toPoint() - self._panel.mapToGlobal(self._panel.rect().topLeft())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._offset is not None:
            self.moved.emit(event.globalPosition().toPoint() - self._offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._offset is not None:
            self.moved.emit(event.globalPosition().toPoint() - self._offset)
            self._offset = None
            event.accept()


class ElidingLabel(QLabel):
    """Keep the complete title, and elide only its right edge when space runs out."""
    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(60)
        self.setText(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        self._refresh()

    def _refresh(self):
        super().setText(self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, max(1, self.width())))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class TrackDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(220, max(30, option.fontMetrics.height()+10))

    def paint(self, painter, option, index):
        painter.save()
        r = option.rect.adjusted(1, 1, -1, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setBrush(QColor('#414141' if selected else '#303030'))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(r), 5, 5)
        painter.setFont(option.font)
        title = index.data(Qt.ItemDataRole.DisplayRole) or ''
        text_rect = r.adjusted(7, 0, -7, 0)
        text = painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setPen(QColor('#ffffff' if selected else '#d0d0d0'))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()


class RotaryDial(QDial):
    def __init__(self, color='#bdbdbd', parent=None):
        super().__init__(parent)
        self.setRange(0, 10000)
        self.setSingleStep(100)
        self.setPageStep(500)
        self.setFixedSize(62, 62)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.color = color
        self._drag = None

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(7, 7, 48, 48)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor('#505050'), 3.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(r, 225*16, -270*16)
        fraction = self.value()/10000
        pen.setColor(QColor(self.color if self.isEnabled() else '#606060'))
        p.setPen(pen)
        p.drawArc(r, 225*16, int(-270*16*fraction))
        face = QLinearGradient(0, 14, 0, 50)
        face.setColorAt(0, QColor('#666666'))
        face.setColorAt(1, QColor('#303030'))
        p.setBrush(face)
        p.setPen(QPen(QColor('#202020'), 1))
        p.drawEllipse(QRectF(14, 14, 34, 34))
        a = math.radians(225-270*fraction)
        p.setPen(QPen(QColor('#eeeeee' if self.isEnabled() else '#888888'), 2.3))
        p.drawLine(int(31+math.cos(a)*9), int(31-math.sin(a)*9),
                   int(31+math.cos(a)*16), int(31-math.sin(a)*16))
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = (event.globalPosition().y(), self.value())
            self.setFocus()
            self.sliderPressed.emit()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag:
            y, value = self._drag
            fine = .15 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.
            self.setValue(round(value + (y-event.globalPosition().y())*55*fine))
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = None
            self.sliderReleased.emit()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(getattr(self, 'default_value', 5000))
            event.accept()

    def wheelEvent(self, event):
        step = 25 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 200
        self.setValue(self.value() + round(event.angleDelta().y()/120*step))
        event.accept()


class RotaryControl(QWidget):
    def __init__(self, title, lo, hi, step, decimals, suffix, default, logarithmic=False, color='#bdbdbd'):
        super().__init__()
        self.setMinimumWidth(85)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(1)
        label = QLabel(title.upper())
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet('color:#b0b0b0; font-size:11px;')
        layout.addWidget(label)
        self.dial = RotaryDial(color)
        self.dial.setAccessibleName(f'{title} dial')
        layout.addWidget(self.dial, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.spin = QDoubleSpinBox()
        self.spin.setObjectName('knob_value')
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin.setKeyboardTracking(False)
        self.spin.setAccessibleName(title)
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.spin)
        self.lo, self.hi, self.logarithmic = lo, hi, logarithmic
        self.dial.valueChanged.connect(self._from_dial)
        self.spin.valueChanged.connect(self._from_spin)
        self.spin.setValue(default)
        self._from_spin(self.spin.value())
        self.dial.default_value = self.dial.value()

    def _from_dial(self, value):
        t = value/10000
        self.spin.setValue(self.lo*(self.hi/self.lo)**t if self.logarithmic else self.lo+(self.hi-self.lo)*t)

    def _from_spin(self, value):
        t = math.log(max(value, self.lo)/self.lo)/math.log(self.hi/self.lo) if self.logarithmic else (value-self.lo)/(self.hi-self.lo)
        self.dial.blockSignals(True)
        self.dial.setValue(round(t*10000))
        self.dial.blockSignals(False)
