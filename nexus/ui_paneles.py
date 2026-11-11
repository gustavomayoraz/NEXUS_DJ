"""Paneles de NEXUS: tarjetas de deck, matriz de fusión por capas y línea de tiempo de toda la mezcla."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from . import ui_tema as T
from .capas import CAPAS, NOMBRES_CAPAS

SIGLAS = {"bateria": "BAT", "bajo": "BAJ", "lead": "VOZ", "ambiente": "AMB"}


def _panel(p: QPainter, r: QRectF, borde: QColor, radio: float = 16, relleno: float = 0.035) -> None:
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0.0, QColor(255, 255, 255, int(255 * (relleno + 0.02))))
    g.setColorAt(1.0, QColor(255, 255, 255, int(255 * relleno * 0.4)))
    p.setPen(QPen(borde, 1.2))
    p.setBrush(QBrush(g))
    p.drawRoundedRect(r, radio, radio)


def _onda_pix(onda: np.ndarray, w: int, h: int, color: QColor, rango: tuple[int, int] | None = None) -> tuple[QPixmap, QPixmap]:
    """Dos pixmaps de la onda (apagada y brillante) para poder pintar la parte ya sonada con más luz."""
    pix = []
    if rango is not None:
        onda = onda[rango[0]:rango[1]]
    if len(onda) < 2 or w < 4:
        onda = np.zeros(2, np.float32)
    n = max(1, w // 3)
    idx = np.linspace(0, len(onda), n + 1).astype(int)
    barras = np.array([onda[a:max(b, a + 1)].max() if b > a or a < len(onda) else 0.0 for a, b in zip(idx[:-1], idx[1:])])
    barras = barras / (np.percentile(barras, 98) + 1e-9)
    for alfa in (0.30, 0.95):
        pm = QPixmap(max(1, w), max(1, h))
        pm.fill(Qt.GlobalColor.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setPen(Qt.PenStyle.NoPen)
        q.setBrush(T.con_alfa(color, alfa))
        paso = w / n
        for i, v in enumerate(barras):
            alto = max(1.5, min(1.0, v ** 0.8) * h * 0.92)
            q.drawRoundedRect(QRectF(i * paso, (h - alto) / 2, max(1.0, paso - 1.2), alto), 1, 1)
        q.end()
        pix.append(pm)
    return pix[0], pix[1]


# =====================================================================================================
class TarjetaDeck(QWidget):
    """Muestra un tema con su rol (sale / entra / suena / en espera), la onda con la posición y el nivel de sus 4 capas."""

    buscar = Signal(int, float)                      # (número de tema en la cola, segundo al que se quiere saltar)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.d = dict(pista=None, indice=0, rol="", pos=0.0, niveles=np.zeros(4), gan=np.zeros(4), activo=False, nota="")
        self._cache = {}
        self._drag = False

    def _en_onda(self, x: float, y: float) -> bool:
        pista = self.d["pista"]
        if pista is None or pista.analisis is None:
            return False
        return 16 <= x <= self.width() - 16 and 128 - 6 <= y <= 128 + 64 + 6          # misma zona donde se dibuja la onda

    def _ir(self, x: float) -> None:
        a = self.d["pista"].analisis
        frac = max(0.0, min(1.0, (x - 16) / max(1.0, self.width() - 32)))
        self.buscar.emit(int(self.d["indice"]), frac * a.duracion)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._en_onda(e.position().x(), e.position().y()):
            self._drag = True
            self._ir(e.position().x())

    def mouseMoveEvent(self, e) -> None:
        if self._drag and self.d["pista"] is not None and self.d["pista"].analisis is not None:
            self._ir(e.position().x())

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = False

    def poner(self, **kw) -> None:
        self.d.update(kw)
        self.update()

    def paintEvent(self, _ev) -> None:
        d = self.d
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        pista = d["pista"]
        col = T.color_deck(d["indice"]) if pista is not None else T.GRIS
        act = d["activo"]
        _panel(p, QRectF(1, 1, w - 2, h - 2), T.con_alfa(col, 0.55 if act else 0.20), 18, 0.05 if act else 0.025)
        if pista is None:
            p.setPen(T.GRIS)
            p.setFont(T.fuente(14, 500))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, d["nota"] or "—")
            p.end()
            return
        a = pista.analisis
        # letra del deck
        p.setFont(T.fuente(46, 800))
        for k, al in ((4, 0.06), (2, 0.12)):
            p.setPen(T.con_alfa(col, al))
            p.drawText(QRectF(16 - k, 6 - k, 70 + 2 * k, 62 + 2 * k), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, T.letra_deck(d["indice"]))
        p.setPen(col)
        p.drawText(QRectF(16, 6, 70, 62), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, T.letra_deck(d["indice"]))
        # rol
        rol = d["rol"]
        if rol:
            p.setFont(T.fuente(11, 800))
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(rol) + 22
            r = QRectF(w - tw - 14, 16, tw, 24)
            colr = {"SALIENDO": T.AMBAR, "ENTRANDO": T.VERDE}.get(rol, col)
            p.setPen(QPen(T.con_alfa(colr, 0.9), 1.2))
            p.setBrush(T.con_alfa(colr, 0.16))
            p.drawRoundedRect(r, 12, 12)
            p.setPen(colr)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, rol)
        # título
        p.setFont(T.fuente(16, 700))
        p.setPen(T.BLANCO)
        fm = QFontMetrics(p.font())
        p.drawText(QRectF(16, 70, w - 32, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, fm.elidedText(pista.nombre, Qt.TextElideMode.ElideRight, w - 32))
        # datos
        y = 98
        if a is not None:
            p.setFont(T.fuente(12, 600))
            p.setPen(T.GRIS)
            p.drawText(QRectF(16, y, 120, 20), Qt.AlignmentFlag.AlignVCenter, f"{a.bpm:.0f} BPM  ·  {T.tiempo_txt(a.duracion)}")
            cc = T.color_camelot(a.camelot)
            r = QRectF(w - 72, y, 56, 20)
            p.setPen(QPen(T.con_alfa(cc, 0.9), 1.2))
            p.setBrush(T.con_alfa(cc, 0.18))
            p.drawRoundedRect(r, 10, 10)
            p.setFont(T.fuente(12, 800))
            p.setPen(cc.lighter(125))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, a.camelot)
            # onda
            ow, oh = w - 32, 64
            clave = (pista.id, ow, oh, d["indice"] % 2)
            if clave not in self._cache:
                self._cache = {clave: _onda_pix(a.onda, ow, oh, col)}
            dim, vivo = self._cache[clave]
            ox, oy = 16, y + 30
            p.drawPixmap(ox, oy, dim)
            frac = max(0.0, min(1.0, d["pos"] / max(1.0, a.duracion)))
            p.save()
            p.setClipRect(QRectF(ox, oy, ow * frac, oh))
            p.drawPixmap(ox, oy, vivo)
            p.restore()
            for s0, s1, tipo_m in d.get("marcas", ()):                                 # dónde está programado el enganche en este tema
                if a.duracion <= 0:
                    break
                x0 = ox + ow * max(0.0, min(1.0, s0 / a.duracion))
                x1 = ox + ow * max(0.0, min(1.0, s1 / a.duracion))
                cm = T.AMBAR if tipo_m == "sale" else T.VERDE
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(T.con_alfa(cm, 0.22))
                p.drawRect(QRectF(x0, oy, max(2.0, x1 - x0), oh))
                p.setPen(QPen(T.con_alfa(cm, 0.95), 1.6))
                p.drawLine(QPointF(x0, oy - 4), QPointF(x0, oy + oh))
                p.drawLine(QPointF(x1, oy), QPointF(x1, oy + oh + 4))
                p.setFont(T.fuente(9, 700))
                p.drawText(QRectF(x0 + 3, oy - 1, 60, 12), Qt.AlignmentFlag.AlignLeft, "enganche" if tipo_m == "sale" else "entra")
            xp = ox + ow * frac
            T.brillo_linea(p, (QPointF(xp, oy - 2), QPointF(xp, oy + oh + 2)), T.BLANCO, 1.4, 0.8)
            p.setFont(T.fuente(11, 600))
            p.setPen(T.GRIS)
            p.drawText(QRectF(ox, oy + oh + 2, ow, 16), Qt.AlignmentFlag.AlignLeft, T.tiempo_txt(d["pos"]))
            p.drawText(QRectF(ox, oy + oh + 2, ow, 16), Qt.AlignmentFlag.AlignRight, "-" + T.tiempo_txt(max(0.0, a.duracion - d["pos"])))
            # medidores de las 4 capas de este deck
            my = oy + oh + 26
            mh = max(24, h - my - 30)
            bw = (w - 32) / 4
            for i, nombre in enumerate(CAPAS):
                x = 16 + i * bw
                nivel = float(min(1.1, d["niveles"][i])) if act else 0.0
                gan = float(d["gan"][i])
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 14))
                p.drawRoundedRect(QRectF(x + 6, my, bw - 12, mh), 5, 5)
                alto = mh * min(1.0, nivel)
                g = QLinearGradient(0, my + mh, 0, my)
                g.setColorAt(0, T.con_alfa(col, 0.35))
                g.setColorAt(1, T.con_alfa(col.lighter(130), 0.95))
                p.setBrush(QBrush(g))
                p.drawRoundedRect(QRectF(x + 6, my + mh - alto, bw - 12, alto), 5, 5)
                if d["activo"] and 0 < gan < 0.999:                                   # raya: cuánto de esa capa deja pasar la fusión
                    yy = my + mh * (1 - gan)
                    p.setPen(QPen(T.BLANCO, 1.5))
                    p.drawLine(QPointF(x + 3, yy), QPointF(x + bw - 3, yy))
                p.setFont(T.fuente(9, 700))
                p.setPen(T.GRIS)
                p.drawText(QRectF(x, my + mh + 3, bw, 14), Qt.AlignmentFlag.AlignCenter, d.get("siglas", tuple(SIGLAS.values()))[i])
        else:
            p.setFont(T.fuente(12, 500))
            p.setPen(T.AMBAR if pista.estado != "error" else QColor("#ff6b6b"))
            p.drawText(QRectF(16, y, w - 32, 80), Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       "Analizando el tema…" if pista.estado != "error" else f"No se pudo analizar: {pista.error}")
        p.end()


# =====================================================================================================
class MatrizFusion(QWidget):
    """Cuatro carriles (batería, bajo, voz/melodía, ambiente) con la curva de cada tema: se ve cómo se reparten la fusión."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.d = dict(fusion=None, u=-1.0, titulo="", detalle="", col_sale=T.CYAN, col_entra=T.MAGENTA, calculando=False, cuenta="", mensaje="")
        self._t0 = time.perf_counter()
        self._cache_pm = None

    def poner(self, **kw) -> None:
        self.d.update(kw)
        self.update()

    def _pixmaps(self, f, w: int, h: int, top: int, lx: int, rx: int, ca: QColor, cb: QColor) -> tuple[QPixmap, QPixmap]:
        """Dibuja una sola vez las curvas de la fusión (y el nivel real) en imágenes; cada cuadro solo las copia."""
        dpr = self.devicePixelRatioF()
        clave = (id(f), w, h, dpr, ca.rgb(), cb.rgb())
        if self._cache_pm is not None and self._cache_pm[0] == clave:
            return self._cache_pm[1], self._cache_pm[2]
        gv = np.asarray(f.viz_gan, np.float32)
        nv = np.asarray(f.viz_nivel, np.float32)
        nf = len(gv)
        paso = max(1, nf // max(1, int((rx - lx) * 1.5)))                       # no más de ~1,5 puntos por píxel
        idx = np.unique(np.append(np.arange(0, nf, paso), nf - 1))
        xs_all = lx + (rx - lx) * (np.arange(nf) / max(1, nf - 1))
        xs = xs_all[idx]
        carril = (h - top - 12) / 4
        est = QPixmap(int(w * dpr), int(h * dpr))
        est.setDevicePixelRatio(dpr)
        est.fill(Qt.GlobalColor.transparent)
        rea = QPixmap(int(w * dpr), int(h * dpr))
        rea.setDevicePixelRatio(dpr)
        rea.fill(Qt.GlobalColor.transparent)
        q = QPainter(est)
        q.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        qr = QPainter(rea)
        qr.setRenderHints(QPainter.RenderHint.Antialiasing)
        L = f.plan.L
        for b in range(L + 1):
            x = lx + (rx - lx) * b / L
            fuerte = (b == int(round(L * f.t_swap)))
            q.setPen(QPen(T.con_alfa(T.BLANCO, 0.55 if fuerte else 0.07), 1.6 if fuerte else 1.0, Qt.PenStyle.DashLine if fuerte else Qt.PenStyle.SolidLine))
            q.drawLine(QPointF(x, top), QPointF(x, h - 10))
        for i, nombre in enumerate(CAPAS):
            y0 = top + i * carril
            y1 = y0 + carril - 6
            q.setFont(T.fuente(11, 700))
            q.setPen(T.BLANCO)
            q.drawText(QRectF(14, y0, lx - 18, y1 - y0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       getattr(f, "carriles", tuple(NOMBRES_CAPAS.values()))[i])
            q.setPen(Qt.PenStyle.NoPen)
            q.setBrush(QColor(255, 255, 255, 9))
            q.drawRoundedRect(QRectF(lx, y0, rx - lx, y1 - y0), 6, 6)
            alto = y1 - y0 - 6
            for k, col in ((0, ca), (1, cb)):
                g = gv[idx, i, k]
                nivel = np.clip(nv[idx, i, k], 0, 1.0)
                ruta = QPainterPath()
                ruta.moveTo(xs[0], y1 - 3)
                for x, v in zip(xs, g):
                    ruta.lineTo(x, y1 - 3 - float(v) * alto)
                ruta.lineTo(xs[-1], y1 - 3)
                ruta.closeSubpath()
                q.setPen(Qt.PenStyle.NoPen)
                q.setBrush(T.con_alfa(col, 0.13))
                q.drawPath(ruta)
                lin = QPainterPath()
                lin.moveTo(xs[0], y1 - 3 - float(g[0]) * alto)
                for x, v in zip(xs[1:], g[1:]):
                    lin.lineTo(x, y1 - 3 - float(v) * alto)
                T.brillo_linea(q, lin, col, 1.8, 0.8)
                real = QPainterPath()                                          # nivel real (se muestra hasta el cabezal)
                real.moveTo(xs[0], y1 - 3)
                for x, v in zip(xs, nivel):
                    real.lineTo(x, y1 - 3 - float(min(v, 1.0)) * alto * 0.92)
                real.lineTo(xs[-1], y1 - 3)
                real.closeSubpath()
                qr.setPen(Qt.PenStyle.NoPen)
                qr.setBrush(T.con_alfa(col, 0.40))
                qr.drawPath(real)
        q.end()
        qr.end()
        self._cache_pm = (clave, est, rea)
        return est, rea

    def paintEvent(self, _ev) -> None:
        d = self.d
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        _panel(p, QRectF(1, 1, w - 2, h - 2), T.BORDE, 16)
        f = d["fusion"]
        top = 34
        p.setFont(T.fuente(12, 800))
        p.setPen(T.BLANCO)
        p.drawText(QRectF(16, 6, w * 0.6, 24), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, d["titulo"] or "MATRIZ DE FUSIÓN")
        p.setFont(T.fuente(11, 600))
        p.setPen(T.GRIS)
        p.drawText(QRectF(w * 0.35, 6, w * 0.65 - 16, 24), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, d["detalle"] or d["cuenta"])
        if f is None:
            t = time.perf_counter() - self._t0
            p.setFont(T.fuente(13, 500))
            p.setPen(T.con_alfa(T.GRIS, 0.7 + 0.3 * math.sin(t * 3)) if d["calculando"] else T.GRIS)
            p.drawText(QRectF(0, top, w, h - top), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       d["mensaje"] or ("Calculando la fusión inteligente…" if d["calculando"] else "Agregá dos temas o más para ver cómo se funden"))
            p.end()
            return
        lx, rx = 92, w - 18
        u = d["u"]
        estat, real = self._pixmaps(f, w, h, top, lx, rx, d["col_sale"], d["col_entra"])
        # etiquetas fijas de los carriles + curvas ya dibujadas (se calculan una sola vez por fusión y tamaño)
        p.drawPixmap(0, 0, estat)
        if u > 0:
            p.save()
            p.setClipRect(QRectF(0, 0, lx + (rx - lx) * min(1.0, u), h))
            p.drawPixmap(0, 0, real)
            p.restore()
        # cabezal
        if u >= 0:
            x = lx + (rx - lx) * u
            T.brillo_linea(p, (QPointF(x, top - 4), QPointF(x, h - 8)), T.BLANCO, 1.8, 0.9)
        p.end()


# =====================================================================================================
class LineaMezcla(QWidget):
    """Toda la mezcla en una tira: temas con su onda, fusiones como puentes luminosos y el cabezal de reproducción."""
    buscar = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bloques: list[dict] = []
        self.pistas: list = []
        self.pos = 0.0
        self._cache = {}
        self._drag = False
        self._v0, self._v1 = 0.0, 1.0

    def poner(self, bloques: list[dict], pistas: list, pos: float) -> None:
        self.bloques, self.pistas, self.pos = bloques, pistas, pos
        self.update()

    def _total(self) -> float:
        return max((b["t1"] for b in self.bloques), default=1.0) or 1.0

    def _ventana(self) -> tuple[list, float, float]:
        """Solo se muestran 3 temas (el que suena y los 2 siguientes) con sus fusiones: con toda la cola no se distinguiría nada."""
        bl = self.bloques
        if not bl:
            return [], 0.0, 1.0
        k = len(bl) - 1
        for j, b in enumerate(bl):
            if self.pos < b["t1"]:
                k = j
                break
        if bl[k]["tipo"] != "solo" and k > 0:                     # dentro de una fusión: se empieza por el tema que sale
            k -= 1
        solos, fin = 0, k
        for j in range(k, len(bl)):
            fin = j
            if bl[j]["tipo"] == "solo":
                solos += 1
                if solos == 3:
                    break
        sub = bl[k:fin + 1]
        return sub, sub[0]["t0"], max(sub[-1]["t1"], sub[0]["t0"] + 1.0)

    def _x(self, t: float) -> float:
        return 12 + (self.width() - 24) * (t - self._v0) / max(1e-6, self._v1 - self._v0)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        _panel(p, QRectF(1, 1, w - 2, h - 2), T.BORDE, 16)
        if not self.bloques:
            p.setFont(T.fuente(13, 500))
            p.setPen(T.GRIS)
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "La línea de la mezcla aparece acá")
            p.end()
            return
        y0, y1 = 22, h - 16
        alto = y1 - y0
        visibles, self._v0, self._v1 = self._ventana()
        for b in visibles:
            x0, x1 = self._x(b["t0"]), self._x(b["t1"])
            if x1 - x0 < 1:
                continue
            if b["tipo"] == "solo":
                pi = b["i"]
                pista = self.pistas[pi] if pi < len(self.pistas) else None
                col = T.color_deck(pi)
                r = QRectF(x0, y0, x1 - x0, alto)
                p.setPen(QPen(T.con_alfa(col, 0.45), 1.0))
                p.setBrush(T.con_alfa(col, 0.07))
                p.drawRoundedRect(r, 6, 6)
                if pista is not None and pista.analisis is not None and len(pista.analisis.onda) > 4:
                    o = pista.analisis.onda
                    rango = (int(b["s0"] * 20), int(b["s1"] * 20))
                    clave = (pista.id, int(r.width()), int(alto), rango)
                    if clave not in self._cache:
                        if len(self._cache) > 60:
                            self._cache.clear()
                        self._cache[clave] = _onda_pix(o, int(r.width()), int(alto - 4), col, rango)[1]
                    p.setOpacity(0.75)
                    p.drawPixmap(int(x0), int(y0 + 2), self._cache[clave])
                    p.setOpacity(1.0)
                    p.setFont(T.fuente(10, 700))
                    p.setPen(T.con_alfa(T.BLANCO, 0.9))
                    fm = QFontMetrics(p.font())
                    if r.width() > 60:
                        p.drawText(QRectF(x0 + 8, y0 + 2, r.width() - 12, 16), Qt.AlignmentFlag.AlignLeft, fm.elidedText(pista.nombre, Qt.TextElideMode.ElideRight, int(r.width() - 14)))
            else:
                pi = b["i"]
                ca, cb = T.color_deck(pi), T.color_deck(pi + 1)
                r = QRectF(x0, y0 - 6, x1 - x0, alto + 12)
                g = QLinearGradient(r.left(), 0, r.right(), 0)
                g.setColorAt(0, T.con_alfa(ca, 0.55))
                g.setColorAt(0.5, QColor(255, 255, 255, 120))
                g.setColorAt(1, T.con_alfa(cb, 0.55))
                for k, al in ((6, 0.08), (3, 0.14)):
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(T.con_alfa(T.mezclar(ca, cb, 0.5), al))
                    p.drawRoundedRect(r.adjusted(-k, -k, k, k), 9, 9)
                p.setPen(QPen(QColor(255, 255, 255, 150), 1.2))
                p.setBrush(QBrush(g))
                p.drawRoundedRect(r, 8, 8)
                # curvas cruzadas
                for sentido, col in ((1, ca), (-1, cb)):
                    tr = QPainterPath()
                    for j in range(25):
                        s = j / 24
                        sg = s * s * (3 - 2 * s)
                        y = r.bottom() - 4 - (r.height() - 8) * (sg if sentido == 1 else 1 - sg)
                        (tr.moveTo if j == 0 else tr.lineTo)(r.left() + r.width() * s, y)
                    p.setPen(QPen(col.lighter(140), 1.6))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(tr)
        # cabezal y reloj
        x = self._x(self.pos)
        T.brillo_linea(p, (QPointF(x, 6), QPointF(x, h - 6)), T.BLANCO, 1.8, 1.0)
        p.setFont(T.fuente(10, 700))
        p.setPen(T.GRIS)
        p.drawText(QRectF(12, 3, 200, 16), Qt.AlignmentFlag.AlignLeft, f"{T.tiempo_txt(self.pos)}  /  {T.tiempo_txt(self._total())}")
        p.end()

    def mousePressEvent(self, e) -> None:
        self._drag = True
        self._ir(e.position().x())

    def mouseMoveEvent(self, e) -> None:
        if self._drag:
            self._ir(e.position().x())

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = False

    def _ir(self, x: float) -> None:
        _, v0, v1 = self._ventana()
        t = v0 + (x - 12) / max(1.0, self.width() - 24) * (v1 - v0)
        self.buscar.emit(max(v0, min(v1, t)))
