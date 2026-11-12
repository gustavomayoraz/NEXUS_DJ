"""Ventana principal de NEXUS."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QShortcut)
from PySide6.QtWidgets import (QCheckBox, QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

from . import ui_tema as T
from .fusion import SIGLAS_NATURAL
from .config import EXTENSIONES, SR, ajustes_mezcla, guardar_config, leer_config
from .motor import Motor
from .planificador import orden_cola
from .tecla_global import TeclaGlobal
from .ui_ajustes import DialogoAjustes
from .ui_lista import ListaCola, expandir_rutas
from .ui_nucleo import EstadoNucleo, N_BANDAS, Nucleo
from .ui_paneles import LineaMezcla, MatrizFusion, TarjetaDeck

LARGOS = [("Auto · el programa decide", (16, 8)), ("Corta · 8 compases", (8,)), ("Larga · 16 compases", (16,)), ("Extra larga · 24 compases", (24, 16, 8))]


def icono_app() -> QIcon:
    pm = QPixmap(256, 256)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    g = QLinearGradient(0, 0, 256, 256)
    g.setColorAt(0, QColor("#19e6ff"))
    g.setColorAt(1, QColor("#ff2fd0"))
    p.setBrush(QColor("#070a1a"))
    p.setPen(QPen(QBrush(g), 10))
    p.drawEllipse(18, 18, 220, 220)
    ruta = QPainterPath()
    f = QFont("Segoe UI")
    f.setPixelSize(150)
    f.setWeight(QFont.Weight.Bold)
    ruta.addText(0, 0, f, "N")
    br = ruta.boundingRect()
    ruta.translate(128 - br.center().x(), 128 - br.center().y())
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(ruta)
    p.end()
    return QIcon(pm)


class Vidrio(QWidget):
    """Panel de cristal oscuro."""

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0, QColor(255, 255, 255, 16))
        g.setColorAt(1, QColor(255, 255, 255, 5))
        p.setPen(QPen(T.BORDE, 1.2))
        p.setBrush(QBrush(g))
        p.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 18, 18)
        p.end()


class Logo(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setMinimumWidth(330)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        f = T.fuente(34, 800)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 7)
        ruta = QPainterPath()
        ruta.addText(4, 38, f, "NEXUS")
        g = QLinearGradient(0, 0, 190, 0)
        g.setColorAt(0, T.CYAN)
        g.setColorAt(0.5, T.VIOLETA)
        g.setColorAt(1, T.MAGENTA)
        for w, a in ((9, 0.06), (5, 0.12)):
            p.setPen(QPen(QColor(138, 92, 255, int(255 * a)), w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(ruta)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawPath(ruta)
        p.setFont(T.fuente(10, 700))
        p.setPen(T.GRIS)
        p.drawText(QRectF(222, 14, 300, 30), Qt.AlignmentFlag.AlignVCenter, "F U S I Ó N   I N T E L I G E N T E")
        p.end()


class BotonPlay(QWidget):
    """Botón circular grande de reproducir / pausar."""
    from PySide6.QtCore import Signal
    clic = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sonando = False
        self._hover = False

    def poner(self, sonando: bool) -> None:
        if sonando != self.sonando:
            self.sonando = sonando
            self.update()

    def enterEvent(self, _e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self.rect().contains(e.position().toPoint()):
            self.clic.emit()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(32, 32)
        g = QLinearGradient(0, 0, 64, 64)
        g.setColorAt(0, T.CYAN)
        g.setColorAt(1, T.MAGENTA)
        for r, a in ((31, 0.10), (29, 0.18)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(138, 92, 255, int(255 * a * (1.6 if self._hover else 1.0))))
            p.drawEllipse(c, r, r)
        p.setPen(QPen(QBrush(g), 3))
        p.setBrush(QColor(8, 12, 30, 245))
        p.drawEllipse(c, 26, 26)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.BLANCO)
        if self.sonando:
            p.drawRoundedRect(QRectF(23, 21, 6.5, 22), 2, 2)
            p.drawRoundedRect(QRectF(34.5, 21, 6.5, 22), 2, 2)
        else:
            tri = QPainterPath()
            tri.moveTo(26, 20)
            tri.lineTo(46, 32)
            tri.lineTo(26, 44)
            tri.closeSubpath()
            p.drawPath(tri)
        p.end()


class VentanaNexus(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NEXUS — fusión inteligente de temas")
        self.setWindowIcon(icono_app())
        self.motor = Motor()
        self.cfg = leer_config()
        self.motor.ajustes = ajustes_mezcla(self.cfg)
        self.motor.gustos = [g for g in self.cfg.get("gustos", []) if isinstance(g, dict)][-300:]
        self._t_tick = time.perf_counter()
        self._t_rev = 0.0
        self._t_bloques = 0.0
        self._bloques: list[dict] = []
        self._beats = 0.0
        self._kick_avg = 0.0
        self._bandas_idx: np.ndarray | None = None
        self._ventana_fft = np.hanning(2048).astype(np.float32)
        self._firma_orden = None
        self._t_orden = 0.0
        self._t_chk_orden = 0.0
        self._X = np.zeros(1025, np.float32)                    # último espectro calculado (lo reutilizan los medidores de los decks)
        self._ref_db = np.full(4, -40.0)                        # referencia que se adapta al volumen del tema (para los medidores)
        self._suave = {0: np.zeros(4, np.float32), 1: np.zeros(4, np.float32)}     # nivel mostrado en cada deck (con ataque/caída)
        self._armar()
        self._restaurar()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)      # en Windows el temporizador normal redondea a ~16/31 ms
        self.timer.start(16)
        pantalla = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1560, int(pantalla.width() * 0.96)), min(960, int(pantalla.height() * 0.94)))
        self.setMinimumSize(900, 640)

    # ------------------------------------------------------------------------------ construcción
    def _armar(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        raiz = QVBoxLayout(central)
        raiz.setContentsMargins(18, 12, 18, 14)
        raiz.setSpacing(10)

        # ---- cabecera
        cab = QHBoxLayout()
        cab.addWidget(Logo())
        cab.addStretch(1)
        lbl = QLabel("Largo de la fusión")
        lbl.setStyleSheet("color:#7f8bb0; font-weight:600;")
        cab.addWidget(lbl)
        self.combo_largo = QComboBox()
        for nombre, _ in LARGOS:
            self.combo_largo.addItem(nombre)
        self.combo_largo.setCurrentIndex(int(self.cfg.get("largo", 0)) % len(LARGOS))
        self.combo_largo.currentIndexChanged.connect(self._cambio_largo)
        self.combo_largo.setMinimumWidth(230)
        cab.addWidget(self.combo_largo)
        b_aj = QPushButton("⚙  Ajustes")
        b_aj.setToolTip("Pausa del cruce, rampas y largo de la fusión")
        b_aj.clicked.connect(self._abrir_ajustes)
        cab.addWidget(b_aj)
        raiz.addLayout(cab)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(14)
        raiz.addLayout(cuerpo, 1)

        # ---- cola (izquierda)
        izq = Vidrio()
        izq.setMinimumWidth(250)
        izq.setMaximumWidth(392)
        li = QVBoxLayout(izq)
        li.setContentsMargins(14, 14, 14, 14)
        li.setSpacing(10)
        fila = QHBoxLayout()
        t = QLabel("COLA")
        t.setStyleSheet("font-size:13px; font-weight:800; letter-spacing:4px; color:#eaf4ff;")
        self.lbl_cuenta = QLabel("0 temas")
        self.lbl_cuenta.setStyleSheet("color:#7f8bb0; font-weight:600;")
        fila.addWidget(t)
        fila.addStretch(1)
        fila.addWidget(self.lbl_cuenta)
        li.addLayout(fila)
        fila2 = QHBoxLayout()
        b1 = QPushButton("➕  Archivos")
        b1.clicked.connect(self._elegir_archivos)
        b2 = QPushButton("📁  Carpeta")
        b2.clicked.connect(self._elegir_carpeta)
        fila2.addWidget(b1)
        fila2.addWidget(b2)
        li.addLayout(fila2)
        self.lista = ListaCola()
        self.lista.motor = self.motor
        self.lista.archivos_soltados.connect(self._agregar)
        self.lista.orden_cambiado.connect(self._orden_manual)
        self.lista.reproducir_desde.connect(self._reproducir_desde)
        self.lista.quitar_fila.connect(self._quitar)
        li.addWidget(self.lista, 1)
        fila3 = QHBoxLayout()
        self.b_orden = QPushButton("🧠  Ordenar ahora")
        self.b_orden.setToolTip("La cola se ordena sola (BPM de menor a mayor, después tonalidad y nombre) a medida que se analizan los temas.\n"
                                "Este botón la ordena ahora mismo. Si suena un tema, solo se ordena lo que viene después.")
        self.b_orden.clicked.connect(self._orden_inteligente)
        b4 = QPushButton("🗑")
        b4.setToolTip("Vaciar la cola")
        b4.setFixedWidth(46)
        b4.clicked.connect(self._vaciar)
        fila3.addWidget(self.b_orden, 1)
        fila3.addWidget(b4)
        li.addLayout(fila3)
        cuerpo.addWidget(izq, 3)

        # ---- centro
        centro = QVBoxLayout()
        centro.setSpacing(10)
        cuerpo.addLayout(centro, 8)
        fila_n = QHBoxLayout()
        fila_n.setSpacing(12)
        self.deck_a = TarjetaDeck()
        self.deck_b = TarjetaDeck()
        for d in (self.deck_a, self.deck_b):
            d.setMinimumWidth(190)
            d.setMaximumWidth(268)
        for d in (self.deck_a, self.deck_b):
            d.buscar.connect(self.motor.buscar)           # clic o arrastre sobre la onda del deck: salta a ese punto del tema
        self.nucleo = Nucleo()
        fila_n.addWidget(self.deck_a, 3)
        fila_n.addWidget(self.nucleo, 5)
        fila_n.addWidget(self.deck_b, 3)
        centro.addLayout(fila_n, 1)
        self.matriz = MatrizFusion()
        self.matriz.setMaximumHeight(204)
        centro.addWidget(self.matriz)
        self.linea = LineaMezcla()
        self.linea.setMaximumHeight(96)
        self.linea.buscar.connect(self.motor.buscar_global)
        centro.addWidget(self.linea)

        # ---- transporte
        tr = QHBoxLayout()
        tr.setSpacing(12)
        self.b_ant = QPushButton("⏮")
        self.b_ant.setFixedSize(52, 46)
        self.b_ant.clicked.connect(self.motor.anterior)
        self.b_play = BotonPlay()
        self.b_play.clic.connect(self._alternar)
        self.b_sig = QPushButton("⏭")
        self.b_sig.setFixedSize(52, 46)
        self.b_sig.clicked.connect(self.motor.siguiente)
        self.b_ya = QPushButton("⚡  Mezclar")
        self.b_ya.setObjectName("primario")
        self.b_ya.setMinimumHeight(46)
        self.b_ya.setToolTip("Mezcla el tema que suena con el siguiente desde donde va: busca un punto cercano, calcula la mezcla y la mete (el tema sigue sonando mientras tanto). Atajo: F2 (funciona aunque NEXUS esté minimizado)")
        self.b_ya.clicked.connect(self._fusionar_ya)
        self.b_bien = QPushButton("👍")
        self.b_mal = QPushButton("👎")
        for b_, v_, tip in ((self.b_bien, 1, "Me gustó esta mezcla: NEXUS repite ese estilo en pares parecidos"),
                            (self.b_mal, -1, "No me gustó esta mezcla: NEXUS la rehace con otra postura")):
            b_.setFixedSize(46, 46)
            b_.setToolTip(tip)
            b_.clicked.connect(lambda _c=False, v=v_: self._votar(v))
        self.lbl_estado = QLabel("")
        self.lbl_estado.setStyleSheet("color:#7f8bb0; font-weight:600;")
        self.lbl_estado.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lv = QLabel("🔊")
        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(self.cfg.get("volumen", 90)))
        self.vol.setMinimumWidth(70)
        self.vol.setMaximumWidth(150)
        self.vol.valueChanged.connect(lambda v: setattr(self.motor, "volumen", (v / 100.0) ** 1.6))
        self.motor.volumen = (self.vol.value() / 100.0) ** 1.6
        self.motor._vol_act = self.motor.volumen
        for w in (self.b_ant, self.b_play, self.b_sig, self.b_ya, self.b_bien, self.b_mal):
            tr.addWidget(w)
        self._ver_votos()
        tr.addSpacing(10)
        tr.addWidget(self.lbl_estado, 1)
        tr.addWidget(lv)
        tr.addWidget(self.vol)
        centro.addLayout(tr)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._alternar)
        QShortcut(QKeySequence("F"), self, activated=self._fusionar_ya)
        self._tecla = TeclaGlobal(self)                          # F2 mezcla desde cualquier programa, aunque NEXUS esté minimizado
        self._tecla.pulsada.connect(self._fusionar_ya)
        if not self._tecla.activa:                               # si otro programa (p. ej. AutoMix) ya tiene F2, al menos funciona con la ventana en primer plano
            QShortcut(QKeySequence(Qt.Key.Key_F2), self, activated=self._fusionar_ya)
        QShortcut(QKeySequence("N"), self, activated=self.motor.siguiente)
        QShortcut(QKeySequence("P"), self, activated=self.motor.anterior)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0, T.BG0)
        g.setColorAt(0.55, QColor("#070b1c"))
        g.setColorAt(1, QColor("#0d0a22"))
        p.fillRect(self.rect(), QBrush(g))
        for (x, y, c) in ((0.15, 0.1, T.CYAN), (0.9, 0.95, T.MAGENTA)):
            r = QRadialGradient(self.width() * x, self.height() * y, self.width() * 0.45)
            r.setColorAt(0, T.con_alfa(c, 0.07))
            r.setColorAt(1, T.con_alfa(c, 0.0))
            p.fillRect(self.rect(), QBrush(r))
        p.end()

    # ------------------------------------------------------------------------------ acciones
    def _restaurar(self) -> None:
        rutas = [r for r in self.cfg.get("cola", []) if isinstance(r, str) and os.path.isfile(r)]
        if rutas:
            self._agregar(rutas, guardar=False)

    def _guardar(self) -> None:
        self.cfg.update(cola=[p.ruta for p in self.lista.pistas()], volumen=self.vol.value(), largo=self.combo_largo.currentIndex())
        guardar_config(self.cfg)

    def _agregar(self, rutas: list[str], guardar: bool = True) -> None:
        vistos = {os.path.basename(p.ruta).casefold() for p in self.lista.pistas()}     # mismo nombre de archivo = duplicado
        nuevas = []
        for r in rutas:
            n = os.path.basename(r).casefold()
            if n in vistos:
                continue
            vistos.add(n)
            nuevas.append(r)
        for p in self.motor.agregar(nuevas):
            self.lista.agregar_pista(p)
        if guardar:
            self._guardar()

    def _elegir_archivos(self) -> None:
        patron = "Audio (" + " ".join("*" + e for e in EXTENSIONES) + ")"
        rutas, _ = QFileDialog.getOpenFileNames(self, "Agregar temas", str(Path.home() / "Music"), patron)
        if rutas:
            self._agregar(rutas)

    def _elegir_carpeta(self) -> None:
        ruta = QFileDialog.getExistingDirectory(self, "Agregar una carpeta de temas", str(Path.home() / "Music"))
        if ruta:
            rutas = expandir_rutas([ruta])
            if not rutas:
                QMessageBox.information(self, "NEXUS", "No encontré temas de audio en esa carpeta.")
                return
            self._agregar(rutas)

    def _orden_manual(self) -> None:
        self._firma_orden = self._firma_actual()        # lo que acaba de ordenar el usuario a mano se respeta hasta que lleguen análisis nuevos
        self.motor.establecer_orden(self.lista.pistas())
        self._guardar()

    def _orden_inteligente(self) -> None:
        self._ordenar_lista(forzar=True)

    def _firma_actual(self) -> tuple:
        return (sum(1 for p in self.lista.pistas() if p.analisis is not None), self.lista.count())

    def _controlar_orden(self) -> None:
        """Cada segundo: si llegaron temas o análisis nuevos, la cola se reordena sola. Si suena un tema, solo se ordena lo que viene
        después, y no se toca cuando falta poco para que empiece la mezcla (así no se cambia el tema siguiente justo antes de mezclar).
        Con colas grandes, como mucho una vez cada 10 s: reordenar cuesta."""
        if QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            return
        m = self.motor
        c = m.cur
        if m.sonando and c is not None and (c.tipo == "fus" or (c.s1 - (c.s0 + m.off)) / SR < 25.0):
            return
        firma = self._firma_actual()
        if firma == self._firma_orden:
            return
        if self.lista.count() >= 300 and time.perf_counter() - self._t_orden < 10.0:
            return
        self._ordenar_lista(forzar=False)

    @staticmethod
    def _energia_tema(p) -> float | None:
        """Energía media del tema (solo los compases con música); None si todavía no se analizó."""
        a = p.analisis
        if a is None or len(a.energia) == 0:
            return None
        e = np.asarray(a.energia, np.float64)
        e = e[e > 0.06]
        return float(e.mean()) if len(e) else None

    def _ordenar_lista(self, forzar: bool) -> None:
        """Ordena la cola como una tabla: BPM de menor a mayor, después tonalidad y nombre; los temas sin analizar, al final.
        Si suena un tema solo se ordena lo que viene después de él (y, si se reordena sola, tampoco los 2 siguientes: A, B y el próximo no cambian)."""
        self._t_orden = time.perf_counter()
        self._firma_orden = self._firma_actual()
        pistas = self.lista.pistas()
        if len(pistas) < 2:
            return
        m = self.motor
        fijas: list = []
        if m.cur is not None and m.sonando:                                  # lo que ya sonó y el tema actual no se tocan
            ref = m.cur.pb if m.cur.tipo == "fus" else m.cur.pista
            if ref in pistas:
                k = pistas.index(ref)
                fijas, pistas = pistas[:k + 1], pistas[k + 1:]
                if not forzar:                                                   # al reordenarse sola, tampoco se mueven los 2 que siguen (el B y el próximo)
                    fijas, pistas = fijas + pistas[:2], pistas[2:]
        datos = [{"bpm": (p.analisis.bpm if p.analisis else None), "cam": (p.analisis.camelot if p.analisis else None), "nombre": p.nombre,
                  "en": self._energia_tema(p)} for p in pistas]
        idx = orden_cola(datos)
        nuevas = fijas + [pistas[i] for i in idx]
        if [id(p) for p in nuevas] == [id(p) for p in self.lista.pistas()]:
            return
        barra = self.lista.verticalScrollBar().value()
        self.lista.poner_pistas(nuevas)
        self.lista.verticalScrollBar().setValue(barra)
        m.establecer_orden(nuevas)
        self._guardar()

    def _quitar(self, fila: int) -> None:
        pista = self.lista.item(fila).data(Qt.ItemDataRole.UserRole)
        self.lista.takeItem(fila)
        self.motor.quitar(pista)
        self._guardar()

    def _vaciar(self) -> None:
        if not self.lista.count():
            return
        if QMessageBox.question(self, "NEXUS", "¿Vaciar toda la cola?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.motor.pausar()
            self.motor.cur = None
            self.motor.futuro = []
            self.lista.clear()
            self.motor.establecer_orden([])
            self._guardar()

    def _alternar(self) -> None:
        if not self.lista.count():
            return
        self.motor.alternar() if self.motor.cur is not None else self.motor.reproducir(max(0, self.lista.currentRow()))

    def _reproducir_desde(self, fila: int) -> None:
        """Doble clic: si no suena nada, reproduce ese tema; si ya hay uno sonando, lo deja como SIGUIENTE (deck en espera)
        y la mezcla con el que suena se recalcula."""
        pista = self.lista.item(fila).data(Qt.ItemDataRole.UserRole)
        if self.motor.cur is not None and self.motor.sonando:
            if self.motor.poner_siguiente(pista):
                barra = self.lista.verticalScrollBar().value()
                self.lista.poner_pistas(list(self.motor.pistas))
                self.lista.verticalScrollBar().setValue(barra)
                self._firma_orden = self._firma_actual()
                self._guardar()
                self.motor.mensaje = f"«{pista.nombre}» quedó como siguiente: recalculando la mezcla…"
            return
        self.motor.reproducir(fila, 0.0)

    def _fusionar_ya(self) -> None:
        if self.motor.cur is None and self.lista.count():
            self.motor.reproducir(max(0, self.lista.currentRow()))
        self.motor.fusionar_ya()                          # el avance y los avisos salen en la barra de estado (motor.aviso_ya)

    def _ver_votos(self) -> None:
        ver = float(self.motor.ajustes.get("i_gustos", 0.0)) >= 0.5 and float(self.motor.ajustes.get("auto", 1.0)) >= 0.5
        self.b_bien.setVisible(ver)
        self.b_mal.setVisible(ver)

    def _votar(self, voto: int) -> None:
        msg = self.motor.votar(voto)
        self.cfg["gustos"] = list(self.motor.gustos)
        self._guardar()
        self.motor._aviso(msg)

    def _abrir_ajustes(self) -> None:
        dlg = DialogoAjustes(self.motor.ajustes, LARGOS, self.combo_largo.currentIndex(), self)
        dlg.aplicado.connect(self._aplicar_ajustes)
        dlg.exec()

    def _aplicar_ajustes(self, aj: dict, indice_largo: int) -> None:
        """Guarda los ajustes y rehace las fusiones que todavía no suenan (la que está sonando no se toca)."""
        self.motor.ajustes = dict(aj)
        self._ver_votos()
        self.cfg["mezcla"] = dict(aj)
        if indice_largo != self.combo_largo.currentIndex():
            self.combo_largo.setCurrentIndex(indice_largo)            # dispara _cambio_largo (que ya invalida)
        else:
            self.motor.invalidar_fusiones()
        self._guardar()

    def _cambio_largo(self, i: int) -> None:
        self.motor.largos = LARGOS[i][1]
        self.motor.invalidar_fusiones()
        self._guardar()

    # ------------------------------------------------------------------------------ espectro en vivo
    def _bandas_vivas(self) -> np.ndarray:
        x = self.motor.ultimo_bloque(2048)
        X = np.abs(np.fft.rfft(x * self._ventana_fft)) / 512.0
        self._X = X
        if self._bandas_idx is None:
            fr = np.fft.rfftfreq(2048, 1 / SR)
            bordes = np.geomspace(35.0, 16000.0, N_BANDAS + 1)
            self._fr = fr
            self._centros = np.sqrt(bordes[:-1] * bordes[1:])
            self._bandas_idx = np.searchsorted(fr, bordes)
            self._peso = ((self._centros / 1000.0) ** 0.45).astype(np.float32)
        idx = self._bandas_idx
        acum = np.concatenate([[0.0], np.cumsum(X)])
        ancho = idx[1:] - idx[:-1]
        media = (acum[idx[1:]] - acum[idx[:-1]]) / np.maximum(1, ancho)
        interp = np.interp(self._centros, self._fr, X)                 # graves: pocas muestras por banda → se interpola
        v = np.where(ancho >= 2, media, interp).astype(np.float32) * self._peso
        db = 20 * np.log10(v + 1e-7)
        return np.clip((db + 72.0) / 52.0, 0.0, 1.0)

    def _niveles_vivos(self, dt: float) -> np.ndarray:
        """Nivel EN VIVO de las 4 zonas del tema que suena (graves, cuerpo, presencia, brillo), medido del audio real
        en cada cuadro: así los medidores del deck se mueven con la música durante todo el tema, no solo en la mezcla."""
        if self._bandas_idx is None:
            return np.zeros(4, np.float32)
        fr = np.fft.rfftfreq(2048, 1 / SR)
        ix = np.searchsorted(fr, [25.0, 150.0, 900.0, 4000.0, 16000.0])
        X = self._X
        e = np.array([np.sqrt(np.sum(X[ix[k]:max(ix[k] + 1, ix[k + 1])] ** 2)) for k in range(4)])
        db = 20 * np.log10(e + 1e-7)
        self._ref_db = np.maximum(db, self._ref_db - 1.5 * dt)                 # el techo baja despacio si el tema se calma
        self._ref_db = np.maximum(self._ref_db, -45.0)
        return np.clip((db - (self._ref_db - 28.0)) / 28.0, 0.0, 1.0).astype(np.float32) * 0.95

    def _suavizar(self, slot: int, objetivo: np.ndarray, dt: float) -> np.ndarray:
        """Ataque rápido y caída más lenta: las barras se mueven fluidas en vez de saltar de un valor a otro."""
        act = self._suave[slot]
        k = np.where(objetivo > act, 1 - math.exp(-dt / 0.025), 1 - math.exp(-dt / 0.14))
        act = act + (objetivo - act) * k
        self._suave[slot] = act.astype(np.float32)
        return self._suave[slot]

    # ------------------------------------------------------------------------------ cuadro de animación
    def _tick(self) -> None:
        ahora = time.perf_counter()
        dt = min(0.1, ahora - self._t_tick)
        self._t_tick = ahora
        m = self.motor
        if ahora - self._t_rev > 0.25:
            self._t_rev = ahora
            m.revisar()
            if ahora - self._t_chk_orden > 1.0:
                self._t_chk_orden = ahora
                self._controlar_orden()
            self.lista.viewport().update()
            self._estado_texto()
        if ahora - self._t_bloques > 0.5:
            self._t_bloques = ahora
            self._bloques = m.bloques()
        snap = m.snapshot()
        self.b_play.poner(m.sonando)
        self.lista.i_actual = (snap["i_b"] if snap["tipo"] == "fus" else snap["i_a"])
        self.lista.sonando = m.sonando
        self.lbl_cuenta.setText(f"{self.lista.count()} tema{'s' if self.lista.count() != 1 else ''}")
        pistas = m.pistas
        tipo = snap["tipo"]
        fus = snap["fusion"] if tipo == "fus" else snap["proxima"]
        u = snap["u"] if tipo == "fus" else -1.0
        u_vis = self._u_visual(snap["fusion"], u) if tipo == "fus" else u          # cursor que se queda quieto en la línea blanca
        e = EstadoNucleo()
        bandas = self._bandas_vivas() if m.sonando else np.zeros(N_BANDAS)
        e.bandas, e.sonando = bandas, m.sonando
        kick = float(bandas[:6].mean())
        self._kick_avg += (kick - self._kick_avg) * (1 - math.exp(-dt * 3))
        e.kick = float(np.clip((kick - self._kick_avg) * 5.0, 0.0, 1.0)) if m.sonando else 0.0

        pa, pb = snap["pa"], snap["pb"]
        ia, ib = snap["i_a"], snap["i_b"]
        ana = pa.analisis if pa is not None else None
        anb = pb.analisis if pb is not None else None
        if pa is not None:
            e.col_sale = T.color_deck(ia)
            e.col_entra = T.color_deck(ib if ib >= 0 else ia + 1)
            if ana:
                e.bpm_sale, e.cam_sale = ana.bpm, ana.camelot
            if tipo == "fus" and anb:
                e.bpm_entra, e.cam_entra = anb.bpm, anb.camelot
                e.u = u_vis
                e.t_swap = snap["fusion"].t_swap
                f = snap["fusion"]
                fr = int((u * f.n) / SR * 50)
                if 0 <= fr < len(f.viz_bandas):
                    v = f.viz_bandas[fr].astype(np.float32)
                    sh = v[:, 1] / (v[:, 0] + v[:, 1] + 1e-6)
                    e.share = np.where(v.sum(axis=1) < 1e-3, u, sh)
                e.swap_en = max(0.0, (f.t_swap - u) * f.duracion)
            bpm_now = e.bpm_sale if tipo != "fus" else (e.bpm_sale * (1 - min(1, max(0, (u - 0.18) / 0.5))) + e.bpm_entra * min(1, max(0, (u - 0.18) / 0.5)))
            if m.sonando and bpm_now > 0:
                self._beats += dt * bpm_now / 60.0
            e.beats = self._beats
        else:
            e.subtitulo = "Arrastrá temas a la cola y apretá ▶"
        self.nucleo.actualizar(e)

        # ---- decks A (pares) y B (impares)
        self._decks(snap, tipo, u, dt)

        # ---- matriz de fusión
        self._matriz(snap, tipo, u_vis, fus)

        # ---- línea de tiempo
        pos_lin = m.pos_global(self._bloques) if m.cur is not None else 0.0
        if tipo == "fus":
            pos_lin += (u_vis - u) * snap["fusion"].duracion
        self.linea.poner(self._bloques, pistas, pos_lin)

    def _u_visual(self, f, u: float) -> float:
        """Posición del cursor que se dibuja: copia el movimiento del fader. Se queda quieto en la línea blanca durante los segundos
        de 'Congelar' (el sonido sigue) y después avanza a la misma velocidad que el fader."""
        dur = f.duracion
        seg, k = f.congelado
        if seg <= 0 or dur <= 0 or u < f.t_swap:
            return u
        hold = seg / dur
        if u < f.t_swap + hold:
            return f.t_swap
        return min(1.0, f.t_swap + (u - f.t_swap - hold) / max(0.3, k))

    def _decks(self, snap: dict, tipo, u: float, dt: float = 0.033) -> None:
        m = self.motor
        slots = {0: dict(pista=None, indice=0, rol="", pos=0.0, niveles=np.zeros(4), gan=np.zeros(4), activo=False, nota="Deck A\nlibre"),
                 1: dict(pista=None, indice=1, rol="", pos=0.0, niveles=np.zeros(4), gan=np.zeros(4), activo=False, nota="Deck B\nlibre")}
        pistas = m.pistas

        def marcas(idx, rol=""):
            """Rango (segundos del tema) del enganche que le corresponde a ese deck: si está sonando, el que sale hacia el siguiente;
            si todavía no sonó, el que entra desde el anterior. El resto de la onda queda libre para moverse con el mouse."""
            res = []
            if not (0 <= idx < len(pistas)):
                return res
            if rol == "SALIENDO":
                quiere = "sale"
            elif rol == "ENTRANDO":
                quiere = "entra"
            else:
                quiere = "sale" if idx <= m._indice_actual() else "entra"
            for a, b, tipo_m in ((idx, idx + 1, "sale"), (idx - 1, idx, "entra")):
                if tipo_m != quiere:
                    continue
                if a < 0 or b >= len(pistas):
                    continue
                f = m._ya.get((pistas[a].id, pistas[b].id))
                if f is None:
                    f = m.fusiones.get((pistas[a].id, pistas[b].id))
                if not hasattr(f, "a_inicio"):
                    f = m._hist.get((pistas[a].id, pistas[b].id))
                if not hasattr(f, "a_inicio"):
                    continue
                if tipo_m == "sale":
                    res.append((f.a_inicio / SR, (f.a_inicio + f.n) / SR, tipo_m))
                else:
                    res.append((max(0.0, (f.b_continua - f.n) / SR), f.b_continua / SR, tipo_m))
            return res

        def poner(pista, idx, rol, pos, niv, gan, activo, siglas=None):
            if pista is None or idx < 0:
                return
            niv = self._suavizar(idx % 2, np.asarray(niv, np.float32), dt)
            slots[idx % 2] = dict(pista=pista, indice=idx, rol=rol, pos=pos, niveles=niv, gan=gan, activo=activo, nota="",
                                  siglas=siglas or ("BAT", "BAJ", "VOZ", "AMB"), marcas=marcas(idx, rol))

        if snap["tipo"] is None:
            for i in range(min(2, len(pistas))):
                poner(pistas[i], i, "CARGADO", 0.0, np.zeros(4), np.zeros(4), False)
        else:
            sona = m.sonando
            if tipo == "fus":
                f = snap["fusion"]
                pos_f = min(len(f.viz_nivel) - 1.0, (u * f.n) / SR * 50)       # cuadro con decimales: se interpola entre dos cuadros
                fr = int(pos_f)
                fr1 = min(len(f.viz_nivel) - 1, fr + 1)
                t = pos_f - fr
                niv = (f.viz_nivel[fr].astype(np.float32) * (1 - t) + f.viz_nivel[fr1].astype(np.float32) * t)
                gan = (f.viz_gan[fr].astype(np.float32) * (1 - t) + f.viz_gan[fr1].astype(np.float32) * t)
                poner(snap["pa"], snap["i_a"], "SALIENDO", snap["pos_a"], niv[:, 0], gan[:, 0], sona, f.siglas)
                poner(snap["pb"], snap["i_b"], "ENTRANDO", snap["pos_b"], niv[:, 1], gan[:, 1], sona, f.siglas)
            else:
                p = snap["pa"]
                niv = self._niveles_vivos(dt) if sona else np.zeros(4, np.float32)
                poner(p, snap["i_a"], "SONANDO" if sona else "EN PAUSA", snap["pos_a"], niv, np.ones(4, np.float32), sona, SIGLAS_NATURAL)
                if snap["pb"] is not None:
                    poner(snap["pb"], snap["i_b"], "SIGUIENTE", 0.0, np.zeros(4), np.zeros(4), False, SIGLAS_NATURAL)
        self.deck_a.poner(**slots[0])
        self.deck_b.poner(**slots[1])

    def _matriz(self, snap: dict, tipo, u: float, fus) -> None:
        m = self.motor
        pistas = m.pistas
        if tipo is None or snap["pa"] is None:
            self.matriz.poner(fusion=None, u=-1.0, titulo="MATRIZ DE FUSIÓN", detalle="", calculando=False,
                              mensaje="Cuando suene la mezcla vas a ver acá cómo se reparten cada capa los dos temas" if pistas else "")
            return
        i = snap["i_b"] if tipo == "fus" else snap["i_a"]
        sale = T.color_deck(snap["i_a"])
        entra = T.color_deck(snap["i_a"] + 1) if tipo != "fus" else T.color_deck(snap["i_b"])
        if fus is not None:
            plan = fus.plan
            if tipo == "fus":
                titulo = "FUSIÓN EN CURSO · " + plan.resumen()
                detalle = plan.razon
            else:
                falta = max(0.0, fus.a_inicio / SR - snap["pos_a"])
                titulo = "PRÓXIMA FUSIÓN · " + plan.resumen()
                detalle = f"empieza en {T.tiempo_txt(falta)}  ·  {plan.razon}"
            self.matriz.poner(fusion=fus, u=u, titulo=titulo, detalle=detalle, col_sale=sale, col_entra=entra, calculando=False, mensaje="")
        else:
            hay_sig = tipo == "solo" and snap["pb"] is not None
            self.matriz.poner(fusion=None, u=-1.0, titulo="PRÓXIMA FUSIÓN" if hay_sig else "MATRIZ DE FUSIÓN", detalle="", col_sale=sale, col_entra=entra,
                              calculando=hay_sig, mensaje=("" if hay_sig else "Agregá otro tema a la cola para que haya fusión"))

    def _estado_texto(self) -> None:
        m = self.motor
        analizando = sum(1 for p in m.pistas if p.estado == "analizando")
        calculando = sum(1 for v in m.fusiones.values() if v == "calculando")
        partes = []
        if analizando:
            partes.append(f"🔬 analizando {analizando} tema{'s' if analizando != 1 else ''}")
            if m.guardia.pausado:
                partes.append("⏸ poca memoria libre: espero para no trabar la PC")
        listas, total = m.preparacion()
        if total and listas < total and not analizando:
            partes.append(f"🧬 preparando mezclas {listas}/{total}")
        elif calculando:
            partes.append("🧬 calculando fusión")
        if m.aviso_ya:
            partes.insert(0, m.aviso_ya)
        if m.error_audio:
            partes.append("⚠ " + m.error_audio)
        elif m.mensaje:
            partes.append("⚠ " + m.mensaje)
        if m.fin_lista:
            partes.append("fin de la cola")
        self.lbl_estado.setText("   ·   ".join(partes))

    def closeEvent(self, e) -> None:
        self._guardar()
        try:
            self._tecla.cerrar()
        except Exception:                                    # noqa: BLE001
            pass
        self.timer.stop()
        self.motor.cerrar()
        super().closeEvent(e)


def abrir() -> int:
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(T.HOJA)
    app.setFont(T.fuente(13, 500))
    v = VentanaNexus()
    v.show()
    return app.exec()
