"""Cola de reproducción: se le arrastran archivos o carpetas desde el explorador y se reordena arrastrando."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QMenu, QStyle, QStyledItemDelegate, QToolTip

from . import ui_tema as T
from .config import EXTENSIONES

ALTO_FILA = 22                     # una sola línea por tema: caben tres veces más temas que antes


def expandir_rutas(urls: list[str]) -> list[str]:
    """Archivos de audio sueltos o dentro de carpetas (recursivo), ordenados por nombre."""
    res: list[str] = []
    for u in urls:
        if os.path.isdir(u):
            for raiz, _dirs, archivos in os.walk(u):
                for a in sorted(archivos):
                    if a.lower().endswith(EXTENSIONES):
                        res.append(os.path.join(raiz, a))
        elif u.lower().endswith(EXTENSIONES) and os.path.isfile(u):
            res.append(u)
    return res


class _Delegado(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), ALTO_FILA)

    def paint(self, p: QPainter, option, index) -> None:
        """Una sola línea por tema: número · nombre · BPM · duración · tonalidad · punto de estado de la fusión."""
        lista: "ListaCola" = self.parent()
        pista = index.data(Qt.ItemDataRole.UserRole)
        i = index.row()
        r = QRectF(option.rect).adjusted(3, 1, -3, -1)
        p.save()
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        col = T.color_deck(i)
        actual = (i == lista.i_actual)
        sel = bool(option.state & QStyle.StateFlag.State_Selected)
        p.setPen(QPen(T.con_alfa(col, 0.75 if actual else 0.12), 1.2 if actual else 1.0))
        p.setBrush(QColor(255, 255, 255, 30 if actual else (22 if sel else 8)))
        p.drawRoundedRect(r, 7, 7)
        # franja de color del deck
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.con_alfa(col, 0.95 if actual else 0.55))
        p.drawRoundedRect(QRectF(r.left() + 4, r.top() + 4, 3, r.height() - 8), 1.5, 1.5)
        # número / altavoz
        p.setFont(T.fuente(10, 800))
        p.setPen(col if actual else T.GRIS)
        p.drawText(QRectF(r.left() + 9, r.top(), 24, r.height()), Qt.AlignmentFlag.AlignCenter, "▶" if (actual and lista.sonando) else f"{i + 1:02d}")
        # de derecha a izquierda: punto de estado de la fusión, tonalidad, BPM · duración
        x = r.right() - 6
        if lista.motor is not None and i + 1 < lista.count():
            sig = lista.item(i + 1).data(Qt.ItemDataRole.UserRole)
            est = lista.motor.estado_fusion(pista, sig)
            colf = {"lista": T.VERDE, "calculando": T.AMBAR, "error": QColor("#ff6b6b")}.get(est)
            if colf is not None:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(T.con_alfa(colf, 0.95))
                p.drawEllipse(QPointF(x - 3, r.center().y()), 3.0, 3.0)
        x -= 12
        a = pista.analisis
        p.setFont(T.fuente(10, 600))
        if a is not None:
            cc = T.color_camelot(a.camelot)
            rc = QRectF(x - 30, r.top() + 3, 30, r.height() - 6)
            p.setPen(QPen(T.con_alfa(cc, 0.9), 1.0))
            p.setBrush(T.con_alfa(cc, 0.18))
            p.drawRoundedRect(rc, 6, 6)
            p.setPen(cc.lighter(125))
            p.setFont(T.fuente(9, 800))
            p.drawText(rc, Qt.AlignmentFlag.AlignCenter, a.camelot)
            x = rc.left() - 6
            p.setFont(T.fuente(10, 600))
            p.setPen(T.GRIS)
            txt = f"{a.bpm:.0f} · {T.tiempo_txt(a.duracion)}"
            w = QFontMetrics(p.font()).horizontalAdvance(txt) + 4
            p.drawText(QRectF(x - w, r.top(), w, r.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, txt)
            x -= w + 6
        elif pista.estado == "error":
            p.setPen(QColor("#ff6b6b"))
            w = 58
            p.drawText(QRectF(x - w, r.top(), w, r.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "⚠ error")
            x -= w + 6
        else:
            n = int(time.perf_counter() * 3) % 4
            p.setPen(T.AMBAR)
            w = 70
            p.drawText(QRectF(x - w, r.top(), w, r.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "analizando" + "." * n)
            x -= w + 6
        # nombre
        x0 = r.left() + 36
        p.setFont(T.fuente(12, 700))
        p.setPen(T.BLANCO)
        fm = QFontMetrics(p.font())
        ancho = max(20, int(x - x0))
        p.drawText(QRectF(x0, r.top(), ancho, r.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(pista.nombre, Qt.TextElideMode.ElideRight, ancho))
        p.restore()


class ListaCola(QListWidget):
    archivos_soltados = Signal(list)
    orden_cambiado = Signal()
    reproducir_desde = Signal(int)
    quitar_fila = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.motor = None
        self.i_actual = -1
        self.sonando = False
        self._arrastrando = False
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setItemDelegate(_Delegado(self))
        self.setSpacing(0)
        self.model().rowsMoved.connect(lambda *_: self.orden_cambiado.emit())
        self.itemDoubleClicked.connect(lambda it: self.reproducir_desde.emit(self.row(it)))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def viewportEvent(self, e) -> bool:
        """Al pasar el mouse por un tema se ve su nombre completo (y el error, si lo hay)."""
        if e.type() == QEvent.Type.ToolTip:
            it = self.itemAt(e.pos())
            if it is not None:
                pista = it.data(Qt.ItemDataRole.UserRole)
                txt = pista.nombre + (f"\n⚠ {pista.error}" if getattr(pista, "estado", "") == "error" and pista.error else "")
                QToolTip.showText(e.globalPos(), txt, self)
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(e)

    def pistas(self) -> list:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def poner_pistas(self, pistas: list) -> None:
        self.clear()
        for p in pistas:
            self.agregar_pista(p)

    def agregar_pista(self, pista) -> None:
        it = QListWidgetItem()
        it.setData(Qt.ItemDataRole.UserRole, pista)
        it.setSizeHint(QSize(100, ALTO_FILA))
        self.addItem(it)

    def _menu(self, pos) -> None:
        it = self.itemAt(pos)
        if it is None:
            return
        m = QMenu(self)
        m.setStyleSheet("QMenu { background:#0b1126; border:1px solid #2c3f78; border-radius:8px; padding:4px; } "
                        "QMenu::item { padding:7px 18px; border-radius:6px; } QMenu::item:selected { background:#1d3a7a; }")
        a1 = m.addAction("▶  Reproducir desde acá")
        a2 = m.addAction("✕  Quitar de la cola")
        sel = m.exec(self.viewport().mapToGlobal(pos))
        if sel is a1:
            self.reproducir_desde.emit(self.row(it))
        elif sel is a2:
            self.quitar_fila.emit(self.row(it))

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.currentRow() >= 0:
            self.quitar_fila.emit(self.currentRow())
            return
        super().keyPressEvent(e)

    # ---- arrastrar y soltar desde el explorador
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            self._arrastrando = True
            self.viewport().update()
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dragLeaveEvent(self, e) -> None:
        self._arrastrando = False
        self.viewport().update()
        super().dragLeaveEvent(e)

    def dropEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            self._arrastrando = False
            rutas = expandir_rutas([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
            self.viewport().update()
            e.acceptProposedAction()
            if rutas:
                self.archivos_soltados.emit(rutas)
            return
        super().dropEvent(e)
        self.orden_cambiado.emit()

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        vacia = self.count() == 0
        if not (vacia or self._arrastrando):
            return
        p = QPainter(self.viewport())
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.viewport().rect()).adjusted(10, 10, -10, -10)
        t = time.perf_counter()
        col = T.CYAN if self._arrastrando else T.GRIS
        pen = QPen(T.con_alfa(col, 0.9 if self._arrastrando else 0.4), 2.0, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 5])
        pen.setDashOffset(-t * 6)
        p.setPen(pen)
        p.setBrush(T.con_alfa(col, 0.07 if self._arrastrando else 0.025))
        p.drawRoundedRect(r, 20, 20)
        p.setFont(T.fuente(54, 200))
        p.setPen(T.con_alfa(col, 0.9))
        p.drawText(QRectF(r.left(), r.center().y() - 90, r.width(), 80), Qt.AlignmentFlag.AlignCenter, "⤓")
        p.setFont(T.fuente(17, 700))
        p.setPen(T.BLANCO)
        p.drawText(QRectF(r.left() + 20, r.center().y() - 10, r.width() - 40, 30), Qt.AlignmentFlag.AlignCenter,
                   "Soltá acá tus temas" if self._arrastrando else "Arrastrá tus temas acá")
        p.setFont(T.fuente(12, 500))
        p.setPen(T.GRIS)
        p.drawText(QRectF(r.left() + 20, r.center().y() + 22, r.width() - 40, 44), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                   "archivos o carpetas completas (mp3, wav, flac…)")
        p.end()
