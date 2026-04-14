import math

from PySide6 import QtCore, QtGui, QtWidgets

from dlgo import Player, Point
from dlgo.goboard import Move


class GoBoardWidget(QtWidgets.QWidget):
    """
    Custom painted Go board widget.
    """

    point_clicked = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._game_state = None
        self._hover_point = None
        self._outer_margin = 24
        self._inner_margin = 34
        self.setMinimumSize(560, 560)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CrossCursor)

    def set_game_state(self, game_state):
        self._game_state = game_state
        self._hover_point = None
        self.update()

    def sizeHint(self):
        return QtCore.QSize(700, 700)

    def leaveEvent(self, event):
        self._hover_point = None
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if self._game_state is None:
            return
        point = self._pixel_to_point(event.position())
        if point != self._hover_point:
            self._hover_point = point
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self._game_state is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            point = self._pixel_to_point(event.position())
            if point is not None:
                self.point_clicked.emit(point)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#f3f1ec"))

        if self._game_state is None:
            self._draw_empty_hint(painter)
            return

        geo = self._board_geometry()
        self._draw_board_background(painter, geo)
        self._draw_grid(painter, geo)
        self._draw_star_points(painter, geo)
        self._draw_stones(painter, geo)
        self._draw_last_move_marker(painter, geo)
        self._draw_hover_preview(painter, geo)

    def _draw_empty_hint(self, painter):
        painter.setPen(QtGui.QColor("#6a665e"))
        painter.setFont(QtGui.QFont("Segoe UI", 13, QtGui.QFont.Weight.Medium))
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Start a new game to play.")

    def _board_geometry(self):
        board = self._game_state.board
        size = board.num_rows

        area = self.rect().adjusted(
            self._outer_margin,
            self._outer_margin,
            -self._outer_margin,
            -self._outer_margin,
        )
        board_span = min(area.width(), area.height()) - self._inner_margin * 2
        board_span = max(50, board_span)

        step = board_span / max(1, (size - 1))
        origin_x = area.center().x() - board_span / 2
        origin_y = area.center().y() - board_span / 2

        return {
            "size": size,
            "step": step,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "span": board_span,
            "rect": QtCore.QRectF(origin_x - 18, origin_y - 18, board_span + 36, board_span + 36),
        }

    def _draw_board_background(self, painter, geo):
        grad = QtGui.QLinearGradient(geo["rect"].topLeft(), geo["rect"].bottomRight())
        grad.setColorAt(0.0, QtGui.QColor("#f2cd82"))
        grad.setColorAt(0.6, QtGui.QColor("#dbb06a"))
        grad.setColorAt(1.0, QtGui.QColor("#b88443"))
        painter.setBrush(QtGui.QBrush(grad))
        painter.setPen(QtGui.QPen(QtGui.QColor("#a77437"), 2))
        painter.drawRoundedRect(geo["rect"], 20, 20)

    def _draw_grid(self, painter, geo):
        line_pen = QtGui.QPen(QtGui.QColor("#654321"), 1.5)
        painter.setPen(line_pen)
        size = geo["size"]
        step = geo["step"]
        ox = geo["origin_x"]
        oy = geo["origin_y"]
        span = geo["span"]

        for i in range(size):
            x = ox + i * step
            painter.drawLine(QtCore.QPointF(x, oy), QtCore.QPointF(x, oy + span))
            y = oy + i * step
            painter.drawLine(QtCore.QPointF(ox, y), QtCore.QPointF(ox + span, y))

    def _draw_star_points(self, painter, geo):
        size = geo["size"]
        if size < 5:
            return

        if size <= 7:
            anchors = [2, size - 1]
        else:
            anchors = [4, size - 3]

        stars = set()
        for r in anchors:
            for c in anchors:
                stars.add((r, c))
        if size % 2 == 1:
            center = (size + 1) // 2
            stars.add((center, center))

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#3a2612"))
        radius = max(2.0, geo["step"] * 0.08)
        for row, col in stars:
            x, y = self._point_to_pixel(Point(row, col), geo)
            painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)

    def _draw_stones(self, painter, geo):
        board = self._game_state.board
        radius = geo["step"] * 0.44
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                point = Point(r, c)
                stone = board.get(point)
                if stone is None:
                    continue
                x, y = self._point_to_pixel(point, geo)
                self._draw_stone(painter, x, y, radius, stone)

    def _draw_stone(self, painter, x, y, radius, stone):
        rect = QtCore.QRectF(x - radius, y - radius, radius * 2, radius * 2)
        if stone == Player.black:
            grad = QtGui.QRadialGradient(
                QtCore.QPointF(x - radius * 0.3, y - radius * 0.35),
                radius * 1.3,
            )
            grad.setColorAt(0.0, QtGui.QColor("#7d7d7d"))
            grad.setColorAt(0.25, QtGui.QColor("#2d2d2d"))
            grad.setColorAt(1.0, QtGui.QColor("#090909"))
            pen = QtGui.QPen(QtGui.QColor("#111111"), 1.2)
        else:
            grad = QtGui.QRadialGradient(
                QtCore.QPointF(x - radius * 0.35, y - radius * 0.4),
                radius * 1.25,
            )
            grad.setColorAt(0.0, QtGui.QColor("#ffffff"))
            grad.setColorAt(0.45, QtGui.QColor("#f4f4f4"))
            grad.setColorAt(1.0, QtGui.QColor("#cdcdcd"))
            pen = QtGui.QPen(QtGui.QColor("#9a9a9a"), 1.1)

        painter.setBrush(QtGui.QBrush(grad))
        painter.setPen(pen)
        painter.drawEllipse(rect)

    def _draw_last_move_marker(self, painter, geo):
        last_move = self._game_state.last_move
        if last_move is None or not last_move.is_play:
            return

        x, y = self._point_to_pixel(last_move.point, geo)
        marker_radius = max(3.0, geo["step"] * 0.10)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e53935"), 2))
        painter.drawEllipse(QtCore.QPointF(x, y), marker_radius, marker_radius)

    def _draw_hover_preview(self, painter, geo):
        if self._hover_point is None:
            return
        if self._game_state.board.get(self._hover_point) is not None:
            return

        move = Move.play(self._hover_point)
        if not self._game_state.is_valid_move(move):
            return

        color = self._game_state.next_player
        x, y = self._point_to_pixel(self._hover_point, geo)
        radius = geo["step"] * 0.40

        if color == Player.black:
            brush = QtGui.QColor(30, 30, 30, 130)
            pen = QtGui.QPen(QtGui.QColor(10, 10, 10, 130), 1)
        else:
            brush = QtGui.QColor(240, 240, 240, 170)
            pen = QtGui.QPen(QtGui.QColor(120, 120, 120, 150), 1)

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawEllipse(QtCore.QRectF(x - radius, y - radius, radius * 2, radius * 2))

    def _point_to_pixel(self, point, geo):
        x = geo["origin_x"] + (point.col - 1) * geo["step"]
        y = geo["origin_y"] + (point.row - 1) * geo["step"]
        return x, y

    def _pixel_to_point(self, pos):
        if self._game_state is None:
            return None
        geo = self._board_geometry()
        size = geo["size"]
        step = geo["step"]

        col = round((pos.x() - geo["origin_x"]) / step) + 1
        row = round((pos.y() - geo["origin_y"]) / step) + 1
        if row < 1 or row > size or col < 1 or col > size:
            return None

        point = Point(row=row, col=col)
        x, y = self._point_to_pixel(point, geo)
        distance = math.hypot(pos.x() - x, pos.y() - y)
        if distance > step * 0.48:
            return None
        return point
