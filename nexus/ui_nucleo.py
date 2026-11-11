"""NÚCLEO: la pieza central. Un reactor circular donde se ve, banda por banda, cómo un tema se transforma en el otro."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from . import ui_tema as T

N_BANDAS = 48


@dataclass
class EstadoNucleo:
    bandas: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDAS))
    share: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDAS))     # 0 = color del que sale · 1 = color del que entra
    u: float = -1.0                                                           # progreso de la fusión (−1 = tema normal)
    col_sale: QColor = field(default_factory=lambda: QColor(T.CYAN))
    col_entra: QColor = field(default_factory=lambda: QColor(T.MAGENTA))
    bpm_sale: float = 0.0
    bpm_entra: float = 0.0
    cam_sale: str = ""
    cam_entra: str = ""
    beats: float = 0.0                                                        # beats transcurridos (hace girar el plato al ritmo)
    kick: float = 0.0
    t_swap: float = 0.5
    sonando: bool = False
    titulo: str = "NEXUS"
    subtitulo: str = "Arrastrá temas a la cola"
    swap_en: float = -1.0                                                     # segundos que faltan para el relevo de bajo


class Nucleo(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.est = EstadoNucleo()
        self.setMinimumSize(420, 420)
        rng = np.random.default_rng(7)
        self._estrellas = rng.random((150, 4))                                # x, y, fase, brillo
        n = 190
        self._p_ang = rng.random(n) * 2 * np.pi
        self._p_rho = 1.18 + rng.random(n) * 0.75
        self._p_w = (rng.random(n) - 0.5) * 0.5 + 0.12
        self._p_tam = 1.2 + rng.random(n) * 2.6
        self._p_thr = rng.random(n)
        self._t0 = time.perf_counter()
        self._t_ant = self._t0
        self._u_ant = -1.0
        self._flash = -10.0
        self._suave = np.zeros(N_BANDAS)
        self._pico = np.zeros(N_BANDAS)

    def actualizar(self, est: EstadoNucleo) -> None:
        if self._u_ant >= 0 and est.u >= 0 and self._u_ant < est.t_swap <= est.u:
            self._flash = time.perf_counter()
        self._u_ant = est.u
        est.bandas = np.nan_to_num(est.bandas)
        est.share = np.nan_to_num(est.share)
        self.est = est
        self.update()

    # ------------------------------------------------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        ahora = time.perf_counter()
        dt = min(0.1, ahora - self._t_ant)
        self._t_ant = ahora
        t = ahora - self._t0
        e = self.est
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base = min(w, h)
        R = base * 0.285 * (1.0 + 0.035 * e.kick)
        fusionando = e.u >= 0

        # ---- fondo: viñeta + estrellas
        g = QRadialGradient(cx, cy, base * 0.78)
        mix = T.mezclar(e.col_sale, e.col_entra, max(0.0, e.u)) if fusionando else e.col_sale
        g.setColorAt(0.0, T.con_alfa(mix, 0.16))
        g.setColorAt(0.45, QColor(10, 14, 34, 120))
        g.setColorAt(1.0, QColor(4, 5, 13, 0))
        p.fillRect(self.rect(), QBrush(g))
        p.setPen(Qt.PenStyle.NoPen)
        for x, y, fase, br in self._estrellas:
            a = 0.10 + 0.35 * br * (0.5 + 0.5 * math.sin(t * (0.6 + br) + fase * 6.28))
            p.setBrush(QColor(190, 215, 255, int(255 * a)))
            p.drawEllipse(QPointF(x * w, y * h), 0.9 + br * 0.9, 0.9 + br * 0.9)

        # ---- espectro suavizado (sube rápido, baja lento) con marca de pico
        if e.sonando:
            obj = np.clip(e.bandas, 0.0, 1.0)
        else:
            obj = 0.05 + 0.03 * np.sin(t * 1.3 + np.arange(N_BANDAS) * 0.4)
        k_sube, k_baja = 1 - math.exp(-dt * 40), 1 - math.exp(-dt * 7)
        self._suave += np.where(obj > self._suave, k_sube, k_baja) * (obj - self._suave)
        self._pico = np.maximum(self._pico - dt * 0.35, self._suave)
        s = self._suave

        # ---- anillos orbitales (guías)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k, (rr, al) in enumerate(((1.62, 0.10), (2.0, 0.06))):
            pen = QPen(T.con_alfa(mix, al * 2.2), 1.0, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2, 7 + 3 * k])
            pen.setDashOffset(-t * (8 + 6 * k) * (1 if k == 0 else -1))
            p.setPen(pen)
            p.drawEllipse(QPointF(cx, cy), R * rr, R * rr)

        # ---- partículas: cada una "cambia de bando" cuando la fusión avanza
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self._p_ang += self._p_w * dt * (1.0 + 2.0 * e.kick)
        idx_b = ((np.cos(self._p_ang - math.pi / 2) * 0.5 + 0.5) * (N_BANDAS - 1)).astype(int)
        for i in range(len(self._p_ang)):
            if fusionando:
                m = max(0.0, min(1.0, (e.u - self._p_thr[i] * 0.78) / 0.22))
            else:
                m = 0.0
            col = T.mezclar(e.col_sale, e.col_entra, m)
            rho = self._p_rho[i] + 0.16 * s[idx_b[i]] + 0.04 * math.sin(t * 1.7 + i)
            x = cx + math.cos(self._p_ang[i]) * R * rho
            y = cy + math.sin(self._p_ang[i]) * R * rho * 0.96
            al = (0.35 + 0.5 * s[idx_b[i]]) * (0.55 if not e.sonando else 1.0)
            tam = self._p_tam[i] * (1.0 + 0.8 * s[idx_b[i]])
            p.setBrush(T.con_alfa(col, al * 0.10))
            p.drawEllipse(QPointF(x, y), tam * 1.9, tam * 1.9)
            p.setBrush(T.con_alfa(col, al))
            p.drawEllipse(QPointF(x, y), tam * 0.7, tam * 0.7)

        # ---- barras radiales (48 bandas espejadas); el color de cada banda dice de quién es ese tramo del espectro
        r0 = R * 0.93
        ancho_barra = max(2.0, 2 * math.pi * r0 / 96 * 0.52)
        for b in range(N_BANDAS):
            col = T.mezclar(e.col_sale, e.col_entra, float(e.share[b])) if fusionando else T.mezclar(e.col_sale, e.col_sale, 0)
            lon = R * (0.06 + 0.62 * float(s[b]) ** 0.9)
            for lado in (-1, 1):
                ang = math.pi / 2 + lado * (b + 0.5) * math.pi / N_BANDAS
                ca, sa = math.cos(ang), math.sin(ang)
                a0 = QPointF(cx + ca * r0, cy + sa * r0)
                a1 = QPointF(cx + ca * (r0 + lon), cy + sa * (r0 + lon))
                p.setPen(QPen(T.con_alfa(col, 0.10), ancho_barra * 3.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(a0, a1)
                p.setPen(QPen(T.con_alfa(col, 0.85), ancho_barra, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(a0, a1)
                pk = r0 + R * (0.08 + 0.64 * float(self._pico[b]) ** 0.9) + 4
                p.setPen(QPen(T.con_alfa(T.BLANCO, 0.55), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawPoint(QPointF(cx + ca * pk, cy + sa * pk))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # ---- disco central de cristal
        disco = QRadialGradient(cx - R * 0.2, cy - R * 0.25, R * 1.0)
        disco.setColorAt(0.0, QColor(24, 32, 66, 235))
        disco.setColorAt(1.0, QColor(6, 8, 20, 245))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(disco))
        p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

        # ---- plato giratorio: marcas que giran al ritmo (una vuelta cada 4 beats)
        giro = (e.beats / 4.0) * 2 * math.pi
        for k, (rr, sentido, col) in enumerate(((0.80, 1, e.col_sale), (0.69, -1, e.col_entra if fusionando else e.col_sale))):
            alfa = 0.9
            if fusionando:
                alfa = (1.0 - 0.8 * e.u) if k == 0 else (0.2 + 0.8 * e.u)
            for j in range(64):
                ang = sentido * giro + j * 2 * math.pi / 64
                largo = 0.045 if j % 8 else 0.085
                a0 = QPointF(cx + math.cos(ang) * R * rr, cy + math.sin(ang) * R * rr)
                a1 = QPointF(cx + math.cos(ang) * R * (rr - largo), cy + math.sin(ang) * R * (rr - largo))
                p.setPen(QPen(T.con_alfa(col, alfa * (0.95 if j % 8 == 0 else 0.45)), 2.0 if j % 8 == 0 else 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(a0, a1)

        # ---- arco de progreso de la fusión
        if fusionando:
            arco = QRectF(cx - R * 0.595, cy - R * 0.595, R * 1.19, R * 1.19)
            p.setPen(QPen(QColor(255, 255, 255, 28), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(arco, 0, 360 * 16)
            tr = QPainterPath()
            tr.arcMoveTo(arco, 90)
            tr.arcTo(arco, 90, -360 * e.u)
            T.brillo_linea(p, tr, T.mezclar(e.col_sale, e.col_entra, e.u), 4.0)
            # marca del relevo del bajo
            ang = -math.pi / 2 + 2 * math.pi * e.t_swap
            p.setPen(QPen(T.BLANCO if e.u < e.t_swap else T.con_alfa(T.BLANCO, 0.4), 2))
            p.drawLine(QPointF(cx + math.cos(ang) * R * 0.545, cy + math.sin(ang) * R * 0.545),
                       QPointF(cx + math.cos(ang) * R * 0.65, cy + math.sin(ang) * R * 0.65))

        # ---- onda expansiva en el relevo
        dtf = time.perf_counter() - self._flash
        if 0 <= dtf < 1.1:
            f = dtf / 1.1
            p.setBrush(Qt.BrushStyle.NoBrush)
            for q, wd in ((0, 5), (0.12, 2.5)):
                ff = max(0.0, f - q)
                p.setPen(QPen(T.con_alfa(T.BLANCO, (1 - ff) ** 2 * 0.8), wd * (1 - ff) + 0.5))
                p.drawEllipse(QPointF(cx, cy), R * (1.0 + 1.6 * ff), R * (1.0 + 1.6 * ff))

        # ---- textos centrales
        col_txt = T.mezclar(e.col_sale, e.col_entra, max(0.0, e.u)) if fusionando else e.col_sale
        if e.bpm_sale > 0:
            if fusionando:
                wt = max(0.0, min(1.0, (e.u - 0.18) / 0.50))
                wt = wt * wt * (3 - 2 * wt)
                bpm = e.bpm_sale * (1 - wt) + e.bpm_entra * wt
            else:
                bpm = e.bpm_sale
            p.setFont(T.fuente(int(R * 0.46), 300, "Segoe UI Light"))
            p.setPen(T.BLANCO)
            p.drawText(QRectF(cx - R * 0.7, cy - R * 0.42, R * 1.4, R * 0.62), Qt.AlignmentFlag.AlignCenter, f"{bpm:.0f}")
            p.setFont(T.fuente(int(R * 0.105), 700))
            p.setPen(T.con_alfa(col_txt, 0.95))
            p.drawText(QRectF(cx - R * 0.7, cy + R * 0.16, R * 1.4, R * 0.16), Qt.AlignmentFlag.AlignCenter, "B P M")
            # tonalidades
            if fusionando and e.cam_entra:
                self._chip(p, cx - R * 0.30, cy + R * 0.43, e.cam_sale, R * 0.10)
                p.setPen(T.con_alfa(T.BLANCO, 0.6))
                p.setFont(T.fuente(int(R * 0.10), 700))
                p.drawText(QRectF(cx - R * 0.12, cy + R * 0.40, R * 0.24, R * 0.14), Qt.AlignmentFlag.AlignCenter, "→")
                self._chip(p, cx + R * 0.30, cy + R * 0.43, e.cam_entra, R * 0.10)
            elif e.cam_sale:
                self._chip(p, cx, cy + R * 0.43, e.cam_sale, R * 0.10)
        else:
            p.setFont(T.fuente(int(R * 0.30), 200, "Segoe UI Light"))
            p.setPen(T.BLANCO)
            p.drawText(QRectF(cx - R, cy - R * 0.4, R * 2, R * 0.5), Qt.AlignmentFlag.AlignCenter, "NEXUS")
            p.setFont(T.fuente(int(R * 0.095), 600))
            p.setPen(T.GRIS)
            p.drawText(QRectF(cx - R * 0.8, cy + R * 0.12, R * 1.6, R * 0.3), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, e.subtitulo)

        # ---- cartel superior: estado de la fusión
        p.setFont(T.fuente(max(11, int(base * 0.027)), 700))
        if fusionando:
            txt = f"F U S I Ó N   {int(e.u * 100):02d} %"
            p.setPen(T.con_alfa(col_txt, 1.0))
        elif e.sonando and e.bpm_sale > 0:
            txt = "E N   R E P R O D U C C I Ó N"
            p.setPen(T.con_alfa(T.GRIS, 1.0))
        else:
            txt = ""
        p.drawText(QRectF(0, h * 0.015, w, base * 0.07), Qt.AlignmentFlag.AlignCenter, txt)
        if e.swap_en > 0 and fusionando and e.u < e.t_swap:
            p.setFont(T.fuente(max(10, int(base * 0.022)), 600))
            p.setPen(T.con_alfa(T.BLANCO, 0.75))
            p.drawText(QRectF(0, h - base * 0.075, w, base * 0.05), Qt.AlignmentFlag.AlignCenter, f"relevo de bajo en {e.swap_en:0.0f} s")
        p.end()

    @staticmethod
    def _chip(p: QPainter, x: float, y: float, cam: str, tam: float) -> None:
        col = T.color_camelot(cam)
        r = QRectF(x - tam * 1.55, y - tam * 0.75, tam * 3.1, tam * 1.5)
        p.setPen(QPen(T.con_alfa(col, 0.9), 1.4))
        p.setBrush(T.con_alfa(col, 0.18))
        p.drawRoundedRect(r, tam * 0.75, tam * 0.75)
        p.setFont(T.fuente(int(tam * 1.05), 800))
        p.setPen(col.lighter(125))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, cam)
