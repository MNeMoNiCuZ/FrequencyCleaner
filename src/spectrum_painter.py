"""Native Qt painting for the animated analyzer; Matplotlib draws only the axes.

QPolygonF storage is filled with NumPy to avoid allocating thousands of Python
QPointF objects every frame. All coordinates here are logical Qt pixels.
"""
import numpy as np
from matplotlib.path import Path
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QPixmap


def polygon(x, y):
    result = QPolygonF([QPointF()] * len(x))
    pointer = result.data()
    pointer.setsize(len(x) * 16)
    points = np.frombuffer(pointer, dtype=np.float64).reshape(-1, 2)
    points[:, 0], points[:, 1] = x, y
    return result


def simplified(x, y):
    # Subpixel collinear points need no rasterization. In particular a linear
    # FFT grid has thousands of points along a flat EQ response at high Hz.
    path = Path(np.column_stack((x, y)))
    path.simplify_threshold = .25
    vertices = path.cleaned(simplify=True).vertices[:-1]
    return vertices[:, 0], vertices[:, 1]


class SpectrumPainterMixin:
    def paintEvent(self, event):
        self._draw_idle()
        if not getattr(self, '_live_ready', False):
            super().paintEvent(event)
            return
        axes = getattr(self, '_axes_pixmap', None)
        if axes is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, axes)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ratio = self.device_pixel_ratio
        box = self.ax.bbox
        rect = QRectF(box.x0/ratio, (self.fig.bbox.height-box.y1)/ratio,
                      box.width/ratio, box.height/ratio)
        lo, hi = np.log2(self.ax.get_xlim())

        def xcoord(freq):
            return rect.left() + (np.log2(np.maximum(freq, 1.e-8))-lo) * rect.width()/(hi-lo)

        def ycoord(db, eq=False):
            bottom, top = self.eq_ax.get_ylim() if eq else self.ax.get_ylim()
            return rect.bottom() - (np.asarray(db)-bottom)*rect.height()/(top-bottom)

        def line(freq, db, color, width=1., dashed=False):
            if len(freq) == 0:
                return None
            start = max(0, np.searchsorted(freq, 2**lo)-1)
            stop = min(len(freq), np.searchsorted(freq, 2**hi)+1)
            freq, db = np.asarray(freq)[start:stop], np.asarray(db)[start:stop]
            if not len(freq):
                return None
            poly = polygon(*simplified(xcoord(freq), ycoord(db, eq=True)))
            pen = QPen(QColor(color), width)
            pen.setCosmetic(True)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(poly)
            return poly

        painter.save()
        painter.setClipRect(rect)
        for artist, peak, width in [(self._peak_line, True, 1.),
                                   (self._current_line, False, 1.)]:
            freq, db = artist.get_data()
            if len(freq) == 0:
                continue
            x, y = simplified(xcoord(freq), ycoord(db))
            path = QPainterPath()
            path.addPolygon(polygon(np.r_[x[0], x, x[-1]],
                                    np.r_[rect.bottom(), y, rect.bottom()]))
            path.closeSubpath()
            # Start at the visible curve's crest, rather than the top of the
            # axes, so quiet audio still gets the complete vertical gradient.
            visible = (x >= rect.left()) & (x <= rect.right())
            crest = float(np.min(y[visible])) if np.any(visible) else rect.top()
            top = min(rect.bottom()-1., max(rect.top(), crest))
            gradient = QLinearGradient(0, top, 0, rect.bottom())
            background = self._spectrum_background
            if peak:
                # Keep the held envelope subdued, with just enough contrast
                # at its crest to reveal the gradient down to its dark base.
                stops = ((0., self._spectrum_accent.darker(400), 120),
                         (.5, background.darker(115), 150),
                         (1., background.darker(240), 170))
            else:
                stops = ((0., self._spectrum_accent.lighter(125), 72),
                         (.5, self._spectrum_accent.darker(180), 48),
                         (1., background.darker(150), 24))
            for position, tint, opacity in stops:
                tint.setAlpha(opacity)
                gradient.setColorAt(position, tint)
            painter.fillPath(path, gradient)
            pen = QPen(QColor(artist.get_color()), width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPolyline(polygon(x, y))

        key = (self._eq_revision, rect.getRect(), lo, hi, ratio,
               self.gain_range, self.show_auto_cuts, self.shelf_on)
        cached = getattr(self, '_eq_pixmap_cache', None)
        if cached is not None and cached[0] == key:
            painter.drawPixmap(0, 0, cached[1])
            if self._response_artist is not None:
                line(*self._response_artist.get_data(), '#f1ce70', 1.8)
            painter.restore()
            self._paint_frame_label(painter, rect)
            painter.end()
            return
        destination = painter
        overlay = QPixmap(self.size()*ratio)
        overlay.setDevicePixelRatio(ratio)
        overlay.fill(Qt.GlobalColor.transparent)
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(rect)
        line(np.array([2**lo, 2**hi]), np.zeros(2), '#7b7048', .8)
        if self.show_auto_cuts and self._auto_response is not None:
            line(self._response_freq, self._auto_response, '#656565', .65)
        if self.shelf_on:
            painter.setPen(QPen(QColor('#c5ac70'), 1., Qt.PenStyle.DashLine))
            for hz in (self.fade_start_freq, self.cutoff_freq):
                x = float(xcoord(hz))
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index, curve, gain, color in getattr(self, '_native_bands', []):
            active = index in self.selected_curves
            poly = polygon(xcoord(self._response_freq), ycoord(gain, eq=True))
            path = QPainterPath()
            baseline = float(ycoord(0, eq=True))
            path.addPolygon(polygon(np.r_[poly[0].x(), xcoord(self._response_freq), poly[-1].x()],
                                    np.r_[baseline, ycoord(gain, eq=True), baseline]))
            path.closeSubpath()
            fill = QColor(color)
            fill.setAlpha(70 if active else 30)
            painter.fillPath(path, fill)
            line(self._response_freq, gain, color, 1., curve['dynamic'])
        for index, curve, _, color in getattr(self, '_native_bands', []):
            active = index in self.selected_curves
            x = float(xcoord(curve['freq']))
            y = float(ycoord(self._display_handle_gain(curve), eq=True))
            painter.setPen(QPen(QColor('#ffffff'), 1.5 if active else .8))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(x, y), 7 if active else 4, 7 if active else 4)
            if active:
                painter.setPen(QColor('#1c2020'))
                original_font = painter.font()
                font = painter.font()
                font.setPixelSize(9)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(QRectF(x-7, y-7, 14, 14), Qt.AlignmentFlag.AlignCenter, str(index+1))
                painter.setFont(original_font)
            if index == self.selected_curve:
                label = f"{index+1}  {curve['freq']:,.0f} Hz"
                bounds = painter.fontMetrics().boundingRect(label)
                label_rect = QRectF(x-bounds.width()/2-8, y-36, bounds.width()+16, 24)
                painter.setPen(QPen(QColor('#515151'), .8))
                painter.setBrush(QColor('#262626'))
                painter.drawRoundedRect(label_rect, 5, 5)
                painter.setPen(QColor('#e9e9e9'))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.end()
        self._eq_pixmap_cache = (key, overlay)
        painter = destination
        painter.drawPixmap(0, 0, overlay)
        if self._response_artist is not None:
            line(*self._response_artist.get_data(), '#f1ce70', 1.8)
        painter.restore()
        self._paint_frame_label(painter, rect)
        painter.end()

    def _paint_frame_label(self, painter, rect):
        painter.setPen(QColor('#999999'))
        painter.drawText(QRectF(rect.left(), rect.top()-36, rect.width(), 24),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         self._frame_label.get_text())
