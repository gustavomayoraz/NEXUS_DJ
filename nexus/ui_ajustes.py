"""Ventana de ajustes del sistema de mezcla: largo de la fusión, pausa del cruce y rampas."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from . import ui_tema as T
from .config import AJUSTES_MEZCLA, IDEAS_MEZCLA, LIMITES_AJUSTES
from .fusion import disposicion, ventana_bajo

PASO = 0.5                                                   # segundos por paso de cada control


class VistaReparto(QWidget):
    """Dibujo del reparto de la fusión: rampa de entrada · pausa · rampa de salida (según los valores elegidos)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(92)
        self.ajustes = dict(AJUSTES_MEZCLA)
        self.compases = 8
        self.bpm = 126.0

    def poner(self, ajustes: dict, compases: int) -> None:
        self.ajustes, self.compases = ajustes, compases
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        N = 4.0 * self.compases
        P, Q, ri, ro = disposicion(SimpleNamespace(L=self.compases), SimpleNamespace(bpm=self.bpm), self.ajustes)
        seg = 60.0 / self.bpm
        x0, x1, y0, y1 = 8.0, w - 8.0, 30.0, 58.0
        X = lambda b: x0 + (x1 - x0) * b / N
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 14))
        p.drawRoundedRect(QRectF(x0, y0, x1 - x0, y1 - y0), 8, 8)
        tramos = [(0.0, P - ri, T.CYAN, 0.10, "A solo"), (P - ri, P, T.VERDE, 0.40, "entrada"), (P, Q, T.AMBAR, 0.55, "PAUSA"),
                  (Q, Q + ro, T.MAGENTA, 0.40, "salida"), (Q + ro, N, T.MAGENTA, 0.10, "B solo")]
        p.setFont(T.fuente(10, 700))
        for a, b, col, al, nombre in tramos:
            if b - a < 0.05:
                continue
            p.setBrush(T.con_alfa(col, al))
            p.drawRoundedRect(QRectF(X(a) + 1, y0 + 2, X(b) - X(a) - 2, y1 - y0 - 4), 6, 6)
            p.setPen(T.BLANCO)
            if X(b) - X(a) > 46:
                p.drawText(QRectF(X(a), y0, X(b) - X(a), y1 - y0), Qt.AlignmentFlag.AlignCenter, nombre)
            if nombre in ("entrada", "PAUSA", "salida"):
                p.setPen(T.GRIS)
                p.drawText(QRectF(X(a) - 20, y1 + 4, X(b) - X(a) + 40, 16), Qt.AlignmentFlag.AlignCenter, f"{(b - a) * seg:0.1f} s")
            p.setPen(Qt.PenStyle.NoPen)
        b0, b1 = ventana_bajo(SimpleNamespace(L=self.compases), SimpleNamespace(bpm=self.bpm), self.ajustes)
        xb = X((b0 + b1) / 2.0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.con_alfa(T.BLANCO, 0.55))
        p.drawRoundedRect(QRectF(X(b0), y0 + 2, max(3.0, X(b1) - X(b0)), y1 - y0 - 4), 3, 3)
        p.setPen(QPen(T.BLANCO, 1.2, Qt.PenStyle.DashLine))
        p.drawLine(xb, y0 - 8, xb, y1 + 2)
        p.setPen(T.GRIS)
        p.setFont(T.fuente(10, 600))
        p.drawText(QRectF(xb - 80, 2, 160, 14), Qt.AlignmentFlag.AlignCenter, "cruce del bajo")
        p.drawText(QRectF(x0, 2, 200, 14), Qt.AlignmentFlag.AlignLeft, f"fusión de {self.compases} compases · {N * seg:0.0f} s")
        p.end()


class FilaControl(QWidget):
    """Un deslizador con su nombre, su valor en segundos y una explicación corta."""
    cambio = Signal()

    def __init__(self, titulo: str, ayuda: str, clave: str, parent=None) -> None:
        super().__init__(parent)
        self.clave = clave
        lo, hi = LIMITES_AJUSTES[clave]
        self.sl = QSlider(Qt.Orientation.Horizontal)
        self.sl.setRange(int(lo / PASO), int(hi / PASO))
        self.val = QLabel()
        self.val.setMinimumWidth(64)
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val.setStyleSheet("font-weight:800; color:#ffd166;")
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        t = QLabel(titulo)
        t.setStyleSheet("font-weight:700;")
        a = QLabel(ayuda)
        a.setWordWrap(True)
        a.setStyleSheet("color:#7f8bb0; font-size:11px;")
        lay.addWidget(t, 0, 0)
        lay.addWidget(self.val, 0, 1)
        lay.addWidget(self.sl, 1, 0, 1, 2)
        lay.addWidget(a, 2, 0, 1, 2)
        self.sl.valueChanged.connect(self._mover)
        self.auto_en_cero = clave.startswith("rampa_")

    def _mover(self, _v: int) -> None:
        v = self.valor()
        self.val.setText("Auto" if (self.auto_en_cero and v <= 0) else ("sin pausa" if (self.clave == "pausa" and v <= 0) else f"{v:0.1f} s"))
        self.cambio.emit()

    def valor(self) -> float:
        return self.sl.value() * PASO

    def poner(self, v: float) -> None:
        self.sl.setValue(int(round(v / PASO)))
        self._mover(0)


class DialogoAjustes(QDialog):
    """Ajustes del sistema de mezcla. `aplicado(ajustes, indice_largo)` se emite al apretar «Aplicar»."""
    aplicado = Signal(dict, int)

    def __init__(self, ajustes: dict, largos: list, indice_largo: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚙ Ajustes de mezcla")
        self.setMinimumWidth(560)
        self._largos = largos
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(8)
        tit = QLabel("AJUSTES DE MEZCLA")
        tit.setStyleSheet("font-size:14px; font-weight:800; letter-spacing:4px;")
        lay.addWidget(tit)

        fila = QHBoxLayout()
        t = QLabel("Largo de la fusión (de inicio a fin)")
        t.setStyleSheet("font-weight:700;")
        self.combo = QComboBox()
        for nombre, _ in largos:
            self.combo.addItem(nombre)
        self.combo.setCurrentIndex(indice_largo)
        fila.addWidget(t, 1)
        fila.addWidget(self.combo)
        lay.addLayout(fila)
        n = QLabel("Cuánto dura el enganche completo, desde que B empieza a sonar hasta que A desaparece del todo.")
        n.setWordWrap(True)
        n.setStyleSheet("color:#7f8bb0; font-size:11px;")
        lay.addWidget(n)

        fila_p = QHBoxLayout()
        tp = QLabel("Postura de la mezcla")
        tp.setStyleSheet("font-weight:700;")
        self.combo_auto = QComboBox()
        self.combo_auto.addItem("Automática · NEXUS elige cómo mezclar cada par de temas (recomendada)")
        self.combo_auto.addItem("Manual · uso los controles de abajo")
        fila_p.addWidget(tp, 1)
        fila_p.addWidget(self.combo_auto)
        lay.addLayout(fila_p)
        self.nota_auto = QLabel("En automática NEXUS mide cada pareja (tempo, armonía, choque de voces, bajo) y adopta la postura que mejor suena: "
                                "larga y suave, estándar, cuidando las voces o firme y corta. La que usó se ve en el detalle de cada fusión.")
        self.nota_auto.setWordWrap(True)
        self.nota_auto.setStyleSheet("color:#7f8bb0; font-size:11px;")
        lay.addWidget(self.nota_auto)

        fila_m = QHBoxLayout()
        tm = QLabel("Tipo de fusión")
        tm.setStyleSheet("font-weight:700;")
        self.combo_motor = QComboBox()
        self.combo_motor.addItem("Natural · mezcla el audio original (recomendada)", "natural")
        self.combo_motor.addItem("Por capas · separa batería, bajo, voz y ambiente (experimental)", "capas")
        fila_m.addWidget(tm, 1)
        fila_m.addWidget(self.combo_motor)
        lay.addLayout(fila_m)
        ti = QLabel("Ideas de mezcla (tildá las que quieras sumar)")
        ti.setStyleSheet("font-weight:700; margin-top:6px;")
        lay.addWidget(ti)
        self.ideas: dict[str, QCheckBox] = {}
        for clave, (texto, lista) in IDEAS_MEZCLA.items():
            cb = QCheckBox(texto if lista else texto + "  — en preparación")
            cb.setProperty("base", cb.text())
            cb.setEnabled(lista)
            cb.setStyleSheet("font-size:12px;")
            self.ideas[clave] = cb
            lay.addWidget(cb)
        self.f_pausa = FilaControl("Tiempo de espera en el cruce (pausa)", "El cruce se queda quieto este tiempo con los dos temas sonando juntos y alineados; "
                                   "recién después se hace el cambio de bajo y de batería.", "pausa")
        self.f_voz = FilaControl("Rampa de la voz", "Cuánto dura el relevo de las voces. Las voces de A y B nunca suenan juntas: la de B espera a que termine la de A, y entre las dos queda un pequeño hueco.", "voz_rampa")
        self.f_bajo = FilaControl("Rampa del bajo", "Cuánto dura el cruce de los graves. Cae en la mitad de la pausa, sobre una línea de compás, y nunca suenan "
                                  "dos bajos a pleno a la vez. Más corto = cambio más firme; más largo = más suave (con riesgo de que se enturbie).", "bajo_rampa")
        self.f_cong = FilaControl("Congelar en la línea blanca", "Cuando el cursor toca la línea blanca punteada (la mitad del cruce de graves) los faders quedan quietos este "
                                  "tiempo, con los dos temas sonando juntos; después sigue el relevo. 0 = no congelar.", "congelar")
        for f in (self.f_pausa, self.f_voz, self.f_bajo, self.f_cong):
            lay.addWidget(f)
            f.cambio.connect(self._repintar)
        self.vista = VistaReparto()
        lay.addWidget(self.vista)
        self.combo.currentIndexChanged.connect(self._repintar)
        self.combo_auto.currentIndexChanged.connect(self._auto_cambio)
        self.combo_motor.currentIndexChanged.connect(self._reglas_ideas)

        botones = QHBoxLayout()
        bd = QPushButton("Valores por defecto")
        bd.clicked.connect(self._defecto)
        bc = QPushButton("Cancelar")
        bc.clicked.connect(self.reject)
        ba = QPushButton("Aplicar")
        ba.setObjectName("primario")
        ba.clicked.connect(self._aplicar)
        botones.addWidget(bd)
        botones.addStretch(1)
        botones.addWidget(bc)
        botones.addWidget(ba)
        lay.addLayout(botones)
        self.poner(ajustes)

    def _auto_cambio(self, *_a) -> None:
        """En automática se ocultan los controles que decide NEXUS (el congelado depende del usuario, no del par)."""
        auto = self.combo_auto.currentIndex() == 0
        for f in (self.f_pausa, self.f_voz, self.f_bajo):
            f.setVisible(not auto)
        self.vista.setVisible(not auto)
        self._reglas_ideas()
        self._repintar()
        self.adjustSize()

    def _reglas_ideas(self, *_a) -> None:
        """Bloquea las ideas que no se ajustan a lo elegido: algunas solo tienen sentido con la postura Automática y otra solo con
        el tipo de fusión Natural. La marca que el usuario puso se conserva (si vuelve a habilitarse, sigue tildada)."""
        auto = self.combo_auto.currentIndex() == 0
        natural = self.combo_motor.currentData() != "capas"
        bloqueos = {"i_rejilla": (not auto, "no aplica con postura Manual (solo con Automática)"),
                    "i_gustos": (not auto, "no aplica con postura Manual (solo con Automática)"),
                    "i_armonia": (not natural, "no aplica con el tipo «Por capas» (esa fusión ya afina sola)")}
        for k, cb in self.ideas.items():
            base = cb.property("base")
            bloq, motivo = bloqueos.get(k, (False, ""))
            if not IDEAS_MEZCLA[k][1]:
                continue
            cb.setEnabled(not bloq)
            cb.setText(f"{base}  — {motivo}" if bloq else base)

    def poner(self, aj: dict) -> None:
        self.combo_auto.setCurrentIndex(0 if float(aj.get("auto", 1.0)) >= 0.5 else 1)
        self.combo_motor.setCurrentIndex(max(0, self.combo_motor.findData(aj.get("motor", "natural"))))
        for k, cb in self.ideas.items():
            cb.setChecked(float(aj.get(k, 0.0)) >= 0.5)
        self.f_pausa.poner(aj["pausa"])
        self.f_voz.poner(aj["voz_rampa"])
        self.f_bajo.poner(aj.get("bajo_rampa", AJUSTES_MEZCLA["bajo_rampa"]))
        self.f_cong.poner(aj.get("congelar", AJUSTES_MEZCLA["congelar"]))
        self._auto_cambio()

    def valores(self) -> dict:
        return dict(auto=1.0 if self.combo_auto.currentIndex() == 0 else 0.0, motor=self.combo_motor.currentData(), pausa=self.f_pausa.valor(), rampa_in=0.0, rampa_out=0.0, voz_rampa=self.f_voz.valor(), bajo_rampa=self.f_bajo.valor(), congelar=self.f_cong.valor(), **{k: (1.0 if cb.isChecked() else 0.0) for k, cb in self.ideas.items()})

    def _compases(self) -> int:
        tupla = self._largos[self.combo.currentIndex()][1]
        return int(tupla[0])

    def _repintar(self, *_a) -> None:
        self.vista.poner(self.valores(), self._compases())

    def _defecto(self) -> None:
        self.poner(AJUSTES_MEZCLA)
        self.combo.setCurrentIndex(0)

    def _aplicar(self) -> None:
        self.aplicado.emit(self.valores(), self.combo.currentIndex())
        self.accept()
