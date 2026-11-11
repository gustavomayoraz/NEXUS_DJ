"""Paleta, utilidades de dibujo con brillo (glow) y hoja de estilos de NEXUS."""
from __future__ import annotations

import functools

import colorsys

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

BG0 = QColor("#04050d")
BG1 = QColor("#0a0f22")
PANEL = QColor(255, 255, 255, 10)
BORDE = QColor(120, 160, 255, 46)
CYAN = QColor("#19e6ff")
MAGENTA = QColor("#ff2fd0")
VIOLETA = QColor("#8a5cff")
BLANCO = QColor("#eaf4ff")
GRIS = QColor("#7f8bb0")
AMBAR = QColor("#ffb347")
VERDE = QColor("#43ffa5")


def color_deck(indice: int) -> QColor:
    """Color de cada tema según su posición en la cola (alternan cian / magenta como los decks A y B)."""
    return CYAN if indice % 2 == 0 else MAGENTA


def letra_deck(indice: int) -> str:
    return "A" if indice % 2 == 0 else "B"


@functools.lru_cache(maxsize=4096)
def _mezclar_c(r1: int, r2: int, tq: int) -> QColor:
    return _mezclar_calc(QColor.fromRgba(r1), QColor.fromRgba(r2), tq / 64.0)


def mezclar(c1: QColor, c2: QColor, t: float) -> QColor:
    """Color intermedio (se guarda en caché en pasos de 1/64: se llama cientos de veces por cuadro)."""
    return _mezclar_c(c1.rgba(), c2.rgba(), int(round(max(0.0, min(1.0, t)) * 64)))


def _mezclar_calc(c1: QColor, c2: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    # se interpola por la vía del violeta (luminosidad extra en el medio) para que la fusión "brille"
    r = c1.redF() * (1 - t) + c2.redF() * t
    g = c1.greenF() * (1 - t) + c2.greenF() * t
    b = c1.blueF() * (1 - t) + c2.blueF() * t
    brillo = 4 * t * (1 - t) * 0.18
    return QColor.fromRgbF(min(1.0, r + brillo), min(1.0, g + brillo), min(1.0, b + brillo))


@functools.lru_cache(maxsize=8192)
def _con_alfa_c(rgba: int, aq: int) -> QColor:
    q = QColor.fromRgba(rgba)
    q.setAlpha(aq)
    return q


def con_alfa(c: QColor, a: float) -> QColor:
    """El mismo color con otra transparencia (en caché, pasos de 1/255)."""
    return _con_alfa_c(c.rgba() | 0xFF000000, int(round(max(0.0, min(1.0, a)) * 255)))


def color_camelot(cam: str) -> QColor:
    """Color de la rueda armónica: el número fija el matiz; A (menor) más oscuro, B (mayor) más claro."""
    try:
        n = int(cam[:-1])
    except ValueError:
        return QColor("#888")
    h = ((n - 1) * 30) / 360.0
    s, v = (0.70, 0.95) if cam.endswith("B") else (0.80, 0.80)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return QColor.fromRgbF(r, g, b)


def fuente(px: int, peso: int = 400, familia: str = "Segoe UI") -> QFont:
    f = QFont(familia)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(peso))
    return f


def brillo_linea(p: QPainter, trazo, color: QColor, ancho: float = 2.0, intensidad: float = 1.0) -> None:
    """Dibuja una línea/ruta con halo luminoso (capas anchas y transparentes debajo de la línea nítida)."""
    for k, (m, a) in enumerate(((5.0, 0.07), (3.0, 0.14), (1.8, 0.30))):
        p.setPen(QPen(con_alfa(color, a * intensidad), ancho * m, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        _trazar(p, trazo)
    p.setPen(QPen(con_alfa(color, min(1.0, 0.95 * intensidad)), ancho, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    _trazar(p, trazo)


def _trazar(p: QPainter, trazo) -> None:
    if isinstance(trazo, QPainterPath):
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(trazo)
    else:
        a, b = trazo
        p.drawLine(a, b)


def tiempo_txt(seg: float) -> str:
    seg = max(0, int(seg))
    return f"{seg // 60}:{seg % 60:02d}"


HOJA = """
QWidget { background: transparent; color: #eaf4ff; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QMainWindow, QDialog { background: #04050d; }
QToolTip { background: #0d1430; color: #eaf4ff; border: 1px solid #2c3f78; padding: 5px 8px; border-radius: 6px; }
QPushButton { background: rgba(255,255,255,0.05); border: 1px solid rgba(120,160,255,0.30); border-radius: 10px;
              padding: 8px 14px; color: #eaf4ff; font-weight: 600; }
QPushButton:hover { background: rgba(80,140,255,0.18); border-color: rgba(120,200,255,0.8); }
QPushButton:pressed { background: rgba(80,140,255,0.32); }
QPushButton:disabled { color: #56607f; border-color: rgba(120,160,255,0.12); }
QPushButton#primario { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(25,230,255,0.35), stop:1 rgba(255,47,208,0.35));
                       border: 1px solid rgba(180,220,255,0.7); }
QPushButton#primario:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(25,230,255,0.55), stop:1 rgba(255,47,208,0.55)); }
QComboBox { background: rgba(255,255,255,0.05); border: 1px solid rgba(120,160,255,0.30); border-radius: 10px; padding: 7px 12px; }
QComboBox QAbstractItemView { background: #0b1126; border: 1px solid #2c3f78; selection-background-color: #1d3a7a; outline: 0; }
QComboBox::drop-down { border: none; width: 22px; }
QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,0.12); border-radius: 2px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #19e6ff, stop:1 #ff2fd0); border-radius: 2px; }
QSlider::handle:horizontal { background: #eaf4ff; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }
QListWidget { background: transparent; border: none; outline: 0; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: rgba(140,170,255,0.30); border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: rgba(140,200,255,0.55); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { height: 0; background: none; }
QMessageBox { background: #0b1126; }
QMessageBox QLabel { color: #eaf4ff; }
"""
