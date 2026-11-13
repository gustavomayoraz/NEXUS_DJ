"""Motor de reproducción de NEXUS: cola de temas, análisis y fusiones calculados en segundo plano y salida de audio.

El hilo de audio (callback) solo copia muestras ya preparadas; todo lo pesado (análisis, separación de capas,
alineación, fusión) ocurre en un proceso hijo de baja prioridad y se entrega listo antes de que haga falta.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import trabajo
from . import basedatos
from .analisis import VERSION as VERSION_ANALISIS, Analisis, leer_cache
from .audio_io import cargar_estereo
from .config import AJUSTES_MEZCLA, SR
from .memoria import EST_FUSION_MB, Guardia, calcular_perfil, estimar_analisis_mb
from .fusion import VERSION_FUSION, Fusion, desempaquetar, desviacion_tempo, fusion_simple, limitador
from .planificador import SOLO_MIN, SOLO_OK, Plan, planificar

try:                                                    # sounddevice puede faltar o no tener dispositivo: el motor sigue funcionando
    import sounddevice as sd
except Exception:                                       # noqa: BLE001
    sd = None

FADE_SIN_FUSION = int(1.6 * SR)


class Pista:
    """Un tema de la cola."""
    _contador = 0

    def __init__(self, ruta: str) -> None:
        Pista._contador += 1
        self.id = Pista._contador
        self.ruta = str(ruta)
        self.nombre = Path(ruta).stem
        self.analisis: Analisis | None = None
        self.estado = "pendiente"            # pendiente · analizando · listo · error
        self.error = ""
        self.audio: np.ndarray | None = None
        self._cargando = False

    @property
    def n_muestras(self) -> int:
        if self.audio is not None:
            return int(self.audio.shape[1])
        return int(self.analisis.duracion * SR) if self.analisis else 0

    @property
    def gan(self) -> float:
        return 10 ** (self.analisis.ganancia_db / 20) if self.analisis else 1.0


class Tramo:
    """Un trozo de la línea de reproducción: un tema normal ('solo') o una fusión ('fus')."""
    __slots__ = ("tipo", "pista", "fus", "pa", "pb", "s0", "s1", "fade_in", "fade_out")

    def __init__(self, tipo: str, pista: Pista | None = None, s0: int = 0, s1: int = 0, fus: Fusion | None = None,
                 pa: Pista | None = None, pb: Pista | None = None, fade_in: int = 0, fade_out: int = 0) -> None:
        self.tipo, self.pista, self.fus, self.pa, self.pb = tipo, pista, fus, pa, pb
        self.s0, self.s1, self.fade_in, self.fade_out = s0, s1, fade_in, fade_out

    @property
    def largo(self) -> int:
        return self.fus.n if self.tipo == "fus" else max(0, self.s1 - self.s0)


def _crear_pool(trabajadores: int = 1, tareas_por_hijo: int = 6, inicializador=None):
    """Procesos hijos para los trabajos pesados; si no se puede, hilos (menos fluido pero funciona)."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    try:
        import multiprocessing as mp
        # cada hijo se renueva tras unos pocos trabajos: así la memoria que va acumulando (librosa, numba, fragmentación) vuelve al sistema
        return ProcessPoolExecutor(max_workers=trabajadores, mp_context=mp.get_context("spawn"), initializer=inicializador or trabajo.iniciar_hijo,
                                   max_tasks_per_child=tareas_por_hijo)
    except Exception:                                   # noqa: BLE001
        return ThreadPoolExecutor(max_workers=trabajadores)


UMBRAL_NATURAL_YA = 0.14
TEMAS_PRIORITARIOS = 5                                  # el que suena y los 4 siguientes: sus mezclas se preparan antes que cualquier otra cosa


class Motor:
    def __init__(self) -> None:
        self.pistas: list[Pista] = []
        self.lock = threading.RLock()
        self.lock_cb = threading.Lock()                         # protege (cur, off, futuro) frente al hilo de audio; se retiene microsegundos
        # dos grupos de procesos: el análisis de toda la cola no puede hacer esperar a la fusión que va a sonar enseguida
        import os
        self.perfil = calcular_perfil()                            # reparto de hilos/procesos según la RAM y los núcleos de esta PC
        os.environ["NEXUS_HILOS"] = str(self.perfil.hilos)         # los procesos hijo lo heredan (hilos dentro de cada análisis)
        self.guardia = Guardia(self.perfil.reserva_mb)             # frena el trabajo en segundo plano si queda poca RAM libre
        self._max_an = self.perfil.trab_an
        self._an_calentado = False                                # se vuelve True cuando un análisis termina bien
        self._prio_pend = False                                   # hay mezclas de los próximos temas sin preparar: el resto espera
        self._prio_ids: set = set()
        self._lej_fu: set = set()                                 # mezclas lejanas que están calculándose ahora (máximo una a la vez)
        self.pool_an = _crear_pool(self._max_an, self.perfil.tareas_por_hijo)
        self._cerrando = False
        self._lock_an = threading.Lock()
        self._pend_an: list[Pista] = []                           # temas esperando análisis (se eligen por cercanía a lo que suena)
        self._en_vuelo = 0
        self._urgentes: set[int] = set()
        self.pool_fu = _crear_pool(self.perfil.trab_fu, self.perfil.tareas_por_hijo)
        from concurrent.futures import ThreadPoolExecutor
        self.pool_ya = _crear_pool(1, 50, trabajo.iniciar_hijo_normal)          # proceso propio para el ⚡ (no espera detrás de las mezclas de la cola)
        try:
            self.pool_ya.submit(trabajo.iniciar_hijo_normal)        # lo deja arrancado y listo
        except Exception:                                           # noqa: BLE001
            pass
        self.pool_db = ThreadPoolExecutor(max_workers=2)         # planificar y buscar en la base: rápido y fuera del hilo de la interfaz
        self._estado: dict[tuple[int, int], str] = {}            # par -> buscando | calculando | hecha | error
        self._reintentos: dict[tuple[int, int], int] = {}
        self._hist: dict[tuple[int, int], object] = {}      # últimas fusiones vistas (solo para dibujar el pasado)
        self.fusiones: dict[tuple[int, int], object] = {}      # (id_a, id_b) -> Fusion | "calculando" | "error"
        self._ya: dict[tuple[int, int], Fusion] = {}            # mezclas hechas "desde donde va" con el botón ⚡ (tienen prioridad)
        self._ya_tok = 0                                        # identifica el último pedido de ⚡ (los anteriores se descartan)
        self._aviso_t = 0.0
        self._lock_log = threading.Lock()
        self._t_vigia = 0.0
        self._t_ya_nat = 4.0                                    # segundos que tarda una mezcla ⚡ con el motor natural (se afina con cada una)
        self._t_ya_cap = 30.0                                   # ídem con el motor de capas (tempos muy distintos): mucho más lento
        self._ya_t0 = 0.0
        self._adapt: dict[tuple[int, int], Fusion] = {}        # mezclas "adaptadas": cuando la prevista ya quedó atrás, se busca otro punto de salida
        self._adapt_pedidos: dict[tuple[int, int], int] = {}   # par -> desde qué muestra se pidió (para no repetir)
        self._ya_pre: set = set()                               # pares (A,B) cuyos puntos de salida rápida ya se pidieron calcular
        self._ya_grillas: dict = {}                             # (A,B) -> planes de los puntos de salida rápida
        self._ya_hasta = 0.0                                    # mientras dure el cálculo del ⚡ no se arrancan trabajos nuevos en segundo plano
        self._ya_calculando = False
        self.aviso_ya = ""                                      # estado del pedido ⚡ para mostrar en la interfaz
        self.planes: dict[tuple[int, int], Plan] = {}
        self.cur: Tramo | None = None
        self.off = 0
        self.futuro: list[Tramo] = []
        self.sonando = False
        self.volumen = 0.9
        self._vol_act = 0.9
        self.largos: tuple[int, ...] = (16, 8)
        self._simples: dict = {}
        threading.Thread(target=lambda: basedatos.importar_json_viejos(VERSION_ANALISIS), daemon=True).start()   # pasa los .json viejos a la base
        self._simples_en_curso: set = set()
        self.gustos: list = []                                     # votos 👍/👎 del usuario: {"r": rasgos, "p": postura, "v": ±1}
        self.ajustes: dict = dict(AJUSTES_MEZCLA)                 # pausa y rampas del cruce (se pasan a cada fusión)
        self.stream = None
        self.error_audio = ""
        self.mensaje = ""
        self.fin_lista = False
        self.pico = (0.0, 0.0)
        self.anillo = np.zeros(8192, np.float32)
        self.anillo_pos = 0
        self._rebuild = False
        self._cerrando = False
        self._cola_trabajo = 0
        self._gen = 0                                           # sube cada vez que cambian los ajustes de fusión (descarta resultados viejos)

    # ------------------------------------------------------------------ cola y análisis
    def agregar(self, rutas: list[str]) -> list[Pista]:
        nuevas = []
        with self.lock:
            for r in rutas:
                p = Pista(r)
                self.pistas.append(p)
                nuevas.append(p)
        for p in nuevas:
            self._analizar(p)
        self._rebuild = True
        return nuevas

    def _analizar(self, p: Pista) -> None:
        c = leer_cache(p.ruta)
        if c is not None:
            p.analisis, p.estado = c, "listo"
            return
        p.estado = "analizando"
        with self._lock_an:
            if p not in self._pend_an:
                self._pend_an.append(p)
        self._despachar_an()

    def _elegir_an(self) -> Pista:
        """(con _lock_an tomado) El tema pendiente más importante: los urgentes, después los que vienen justo después del que suena."""
        i0 = self._indice_actual()
        orden = {q.id: k for k, q in enumerate(list(self.pistas))}

        def clave(q: Pista):
            k = orden.get(q.id, 10 ** 9)
            return (q.id not in self._urgentes, k < i0, abs(k - i0))
        return min(self._pend_an, key=clave)

    def _limite_an(self) -> int:
        """Análisis a la vez. Hasta que uno termine bien se hace de a uno: la primera vez, los procesos compilan (numba) y
        guardan su caché; si dos lo hicieran juntos podían pisarse y caerse."""
        return self._max_an if self._an_calentado else 1

    def _despachar_an(self) -> None:
        """Mantiene ocupados los procesos de análisis, pero sin encolar todo de golpe: así el orden se decide al momento
        (con 400 temas en la cola, el que suena y los que siguen se analizan primero)."""
        while not self._cerrando:
            if time.time() < self._ya_hasta:                 # se está calculando la mezcla del botón ⚡: tiene toda la PC
                return
            with self._lock_an:
                if self._en_vuelo >= self._limite_an() or not self._pend_an:
                    return
                p = self._elegir_an()
                if self._prio_pend and p.id not in self._prio_ids and p.id not in self._urgentes:
                    return                                    # primero las mezclas de los temas que van a sonar; el resto del análisis espera
                necesita = estimar_analisis_mb(p.ruta)
                if self._en_vuelo > 0 and not self.guardia.hay_lugar(necesita):    # poca RAM: se espera a que termine el que está en curso
                    return
                if self._en_vuelo == 0 and self.guardia.critico():                 # ni uno solo es seguro ahora: se espera a que haya aire
                    self.guardia.pausado = True
                    return
                self._pend_an.remove(p)
                self._urgentes.discard(p.id)
                self._en_vuelo += 1
            try:
                fut = self.pool_an.submit(trabajo.analizar_ruta, p.ruta)
            except Exception as e:                           # noqa: BLE001
                with self._lock_an:
                    self._en_vuelo -= 1
                if self._rehacer_pool("an", e):
                    with self._lock_an:
                        self._pend_an.append(p)
                    continue
                p.estado, p.error = "error", str(e)
                continue
            self._evento(f"análisis enviado: {p.ruta}")
            fut.add_done_callback(lambda f, p=p: self._analisis_listo(p, f))

    def _rehacer_pool(self, cual: str, e: Exception) -> bool:
        """Si un proceso hijo murió (por ejemplo por falta de memoria) el pool queda inutilizable: se crea uno nuevo."""
        from concurrent.futures.process import BrokenProcessPool
        if not isinstance(e, BrokenProcessPool):
            return False
        try:
            if cual == "an":
                viejo, self.pool_an = self.pool_an, _crear_pool(self._max_an, self.perfil.tareas_por_hijo)
            else:
                viejo, self.pool_fu = self.pool_fu, _crear_pool(self.perfil.trab_fu, self.perfil.tareas_por_hijo)
            # se cierra en otro hilo: hacerlo desde el hilo que atiende al propio pool lo dejaba bloqueado para siempre
            threading.Thread(target=lambda: viejo.shutdown(wait=False, cancel_futures=True), daemon=True).start()
        except Exception:                                    # noqa: BLE001
            return False
        return True

    def _analisis_listo(self, p: Pista, fut: Future) -> None:
        with self._lock_an:
            self._en_vuelo = max(0, self._en_vuelo - 1)
        try:
            p.analisis = Analisis.de_dict(fut.result())
            p.estado = "listo"
            self._an_calentado = True
        except Exception as e:                               # noqa: BLE001
            self._evento(f"análisis falló ({p.ruta}): {e!r}")
            self._rehacer_pool("an", e)
            n = getattr(p, "_reint", 0)
            if n < 2 and not self._cerrando:                 # fallo pasajero (memoria, disco ocupado…): se reintenta más tarde
                p._reint = n + 1
                p.estado = "analizando"

                def volver(p=p) -> None:
                    with self._lock_an:
                        if p not in self._pend_an:
                            self._pend_an.append(p)
                    self._despachar_an()
                threading.Timer(6.0, volver).start()
            else:
                p.estado = "error"
                p.error = str(e) or e.__class__.__name__
        self._rebuild = True
        self._despachar_an()

    def poner_siguiente(self, p: Pista) -> bool:
        """Pone el tema `p` justo después del que suena, para que sea el próximo en entrar; la mezcla se recalcula sola."""
        with self.lock:
            if self.cur is None or p not in self.pistas:
                return False
            ref = self.cur.pb if self.cur.tipo == "fus" else self.cur.pista
            if p is ref:
                return False
            self.pistas.remove(p)
            j = self.pistas.index(ref) + 1 if ref in self.pistas else 0
            self.pistas.insert(j, p)
            sig = self.pistas[j + 1] if j + 1 < len(self.pistas) else None
            for q in (p, sig):
                if q is not None and q.analisis is None:
                    self._urgentes.add(q.id)
                    if q.estado == "error":
                        q._reint = 0
                        self._analizar(q)
            self._rebuild = True
        self._despachar_an()
        return True

    def establecer_orden(self, pistas: list[Pista]) -> None:
        with self.lock:
            self.pistas = list(pistas)
            self._rebuild = True

    def quitar(self, p: Pista) -> None:
        with self.lock:
            if p in self.pistas:
                self.pistas.remove(p)
            self._rebuild = True

    # ------------------------------------------------------------------ fusiones
    def _aj_render(self) -> dict:
        """Ajustes que se mandan a calcular una fusión (con los votos si está tildado «aprender de mis gustos»; no cambian la firma)."""
        aj = dict(self.ajustes)
        if self.gustos and float(aj.get("i_gustos", 0.0)) >= 0.5:
            aj["_gustos"] = list(self.gustos[-200:])
        return aj

    def votar(self, voto: int) -> str:
        """👍 (+1) / 👎 (−1) a la mezcla que suena o a la última que sonó. Con 👎 y «aprender de mis gustos» tildado, esa mezcla se rehace."""
        c = self.cur
        if c is None:
            return "No hay ninguna mezcla para evaluar"
        with self.lock:
            if c.tipo == "fus":
                a, b, f = c.pa, c.pb, c.fus
            else:
                i = self._indice_actual()
                if i <= 0 or i >= len(self.pistas):
                    return "Todavía no sonó ninguna mezcla"
                a, b = self.pistas[i - 1], self.pistas[i]
                f = self._ya.get((a.id, b.id))
                if not hasattr(f, "plan"):
                    f = self.fusiones.get((a.id, b.id))
                if not hasattr(f, "plan"):
                    f = self._hist.get((a.id, b.id))
            g = getattr(getattr(f, "plan", None), "detalle", {}).get("gusto") if f is not None else None
            if not g:
                return "Esta mezcla no tiene postura automática para evaluar"
            self.gustos.append(dict(r=g["r"], p=g["p"], v=1 if voto > 0 else -1))
            del self.gustos[:-300]
            if voto < 0 and float(self.ajustes.get("i_gustos", 0.0)) >= 0.5:
                clave = (a.id, b.id)
                try:
                    plan = self.planes.get(clave)
                    ca, cb = basedatos.clave_tema(a.ruta), basedatos.clave_tema(b.ruta)
                    if plan is not None and ca and cb:
                        basedatos.borrar_fusion(ca, cb, basedatos.firma_fusion(asdict(plan), dict(self.ajustes), VERSION_FUSION))
                except Exception:                              # noqa: BLE001
                    pass
                for d in (self.fusiones, self._ya, self.planes, self._estado, self._hist):
                    d.pop(clave, None)
                self._rebuild = True
        return ("👍 Anotado: voy a repetir este estilo en pares parecidos" if voto > 0 else
                "👎 Anotado: esta mezcla se rehace con otra postura" if float(self.ajustes.get("i_gustos", 0.0)) >= 0.5 else
                "👎 Anotado (tildá «Aprender de mis gustos» en Ajustes para que cambie)")

    def _vz(self) -> bool:
        """¿Está tildada la idea «voces por frases»?"""
        return float(self.ajustes.get("i_voces", 0.0)) >= 0.5

    def _ini(self) -> float | None:
        """Dónde empieza la mezcla en todos los temas (0..1), o None si está en Auto."""
        try:
            v = float(self.ajustes.get("inicio_pct", 0.0))
        except (TypeError, ValueError):
            return None
        return v / 100.0 if v > 0.0 else None

    def _est(self) -> bool:
        """¿Está tildada la idea «mezclar por estructura»?"""
        return float(self.ajustes.get("i_estructura", 0.0)) >= 0.5

    def _fusion(self, a: Pista, b: Pista) -> Fusion | None:
        if a in self.pistas and self.pistas.index(a) < self._indice_actual():        # par que ya sonó: se muestra la mezcla que de verdad sonó (p. ej. la ⚡)
            h = self._hist.get((a.id, b.id))
            if h is not None:
                return h
        f = self._ya.get((a.id, b.id))
        if f is None:
            g = self.fusiones.get((a.id, b.id))
            f = g if isinstance(g, Fusion) else None
        if f is not None:
            self._hist[(a.id, b.id)] = f                   # historial: la línea sigue mostrando mezclas de pares que ya sonaron
            if len(self._hist) > 6:
                self._hist.pop(next(iter(self._hist)), None)
            return f
        h = self._hist.get((a.id, b.id))
        if h is not None and self.pistas.index(a) < self._indice_actual() if a in self.pistas else False:
            return h
        return self._cruce_simple(a, b)

    def _cruce_simple(self, a: Pista, b: Pista) -> Fusion | None:
        """Si la fusión todavía no está lista, se usa un cruce solapado simple: nunca un corte seco ni un silencio entre temas.
        Se prepara en un hilo aparte (la primera vez devuelve None; al terminar pide reconstruir la línea)."""
        if a.audio is None or b.audio is None:
            return None
        clave = (a.id, b.id)
        f = self._simples.get(clave)
        if f is not None or clave in self._simples_en_curso:
            return f or None                               # False = no se pudo construir (no se reintenta)
        self._simples_en_curso.add(clave)

        def construir() -> None:
            try:
                self._simples[clave] = fusion_simple(a.audio, a.gan, b.audio, b.gan, seg=6.0, bpm=a.analisis.bpm if a.analisis else 100.0)
                if len(self._simples) > 8:
                    self._simples.pop(next(iter(self._simples)), None)
            except (ValueError, MemoryError, AttributeError, TypeError):
                self._simples[clave] = False
            finally:
                self._simples_en_curso.discard(clave)
                self._rebuild = True

        threading.Thread(target=construir, daemon=True).start()
        return None

    def estado_fusion(self, a: Pista, b: Pista) -> str:
        clave = (a.id, b.id)
        f = self.fusiones.get(clave)
        if isinstance(f, Fusion):
            return "lista"
        est = self._estado.get(clave)
        if est == "hecha":                                # ya está calculada y guardada en la base
            return "lista"
        if est == "error" or f == "error":
            return "error"
        if not (a.analisis and b.analisis):
            return "sin"
        return "calculando" if (est in ("buscando", "calculando") or f == "calculando") else "pendiente"

    def preparacion(self) -> tuple[int, int]:
        """(mezclas listas, mezclas totales) de la cola: sirve para mostrar el avance de la preparación."""
        with self.lock:
            total = max(0, len(self.pistas) - 1)
            listas = sum(1 for i in range(total) if self._estado.get((self.pistas[i].id, self.pistas[i + 1].id)) in ("hecha", "error"))
        return listas, total

    def _ids_cerca(self) -> set:
        i0 = self._indice_actual()
        return {p.id for p in self.pistas[max(0, i0 - 1):i0 + 3]}

    def _actualizar_prioridad(self) -> None:
        """Calcula si falta preparar algo de lo que va a sonar pronto (análisis de los próximos temas o sus mezclas). Mientras falte,
        el análisis del resto de la cola y las mezclas lejanas esperan: todos los recursos van a lo que suena primero."""
        with self.lock:
            i0 = self._indice_actual()
            ventana = self.pistas[i0:i0 + TEMAS_PRIORITARIOS]
            pend = False
            for k, p in enumerate(ventana):
                if p.estado == "error":
                    continue
                if p.analisis is None:
                    pend = True
                    continue
                if k + 1 < len(ventana):
                    q = ventana[k + 1]
                    if q.analisis is None or q.estado == "error":
                        continue
                    est = self._estado.get((p.id, q.id))
                    if est in (None, "buscando", "calculando"):
                        pend = True
            self._prio_ids = {p.id for p in ventana}
            self._prio_pend = pend

    def _programar(self) -> None:
        """Prepara en segundo plano las fusiones de TODA la cola (primero la del tema actual y las que siguen).
        Cada una se busca en la base de datos; solo si no está se calcula, y queda guardada para siempre.
        Acá no se hace nada pesado: solo se reparten los trabajos (este método corre en el hilo de la interfaz)."""
        if time.time() < self._ya_hasta:
            return
        trabajos = []
        self._actualizar_prioridad()
        with self.lock:
            i0 = self._indice_actual()
            cerca = self._ids_cerca()
            for i in sorted(range(len(self.pistas) - 1), key=lambda k: (k < i0, abs(k - i0))):
                a, b = self.pistas[i], self.pistas[i + 1]
                if not (a.analisis and b.analisis):
                    continue
                clave = (a.id, b.id)
                est = self._estado.get(clave)
                if a.id in cerca and b.id in cerca:       # las cercanas tienen que estar en memoria, listas para sonar
                    if clave in self.fusiones or est in ("buscando", "calculando", "error"):
                        continue
                    self.fusiones[clave] = "calculando"
                elif (est is not None or (not (a.id in self._prio_ids and b.id in self._prio_ids) and (self._prio_pend or self._lej_fu))
                      or not self.guardia.hay_lugar(EST_FUSION_MB)):      # las lejanas se preparan solo si sobra memoria y nada urgente espera
                    continue
                self._estado[clave] = "buscando"
                trabajos.append((a.ruta, a.analisis, b.ruta, b.analisis, clave))
            self._podar()
        for t in trabajos:
            try:
                self.pool_db.submit(self._preparar, *t, self._gen)
            except RuntimeError:                             # el motor se está cerrando
                return
        self._actualizar_prioridad()
        self._ya_precalcular()

    def _preparar(self, ruta_a: str, A: Analisis, ruta_b: str, B: Analisis, clave: tuple[int, int], gen: int) -> None:
        """(hilo) Planifica el cruce, lo busca en la base de datos y, si no está, pide calcularlo y guardarlo."""
        if gen != self._gen or self._cerrando:
            return
        try:
            plan = planificar(A, B, self.largos, estructura=self._est(), voces=self._vz(), inicio=self._ini())
            self.planes[clave] = plan
            ca, cb = basedatos.clave_tema(ruta_a), basedatos.clave_tema(ruta_b)
            if ca is None or cb is None:
                raise OSError("no se puede leer uno de los archivos")
            ajustes = self._aj_render()
            firma = basedatos.firma_fusion(asdict(plan), ajustes, VERSION_FUSION)
            if basedatos.existe_fusion(ca, cb, firma):
                self._estado[clave] = "hecha"
                self._cargar_si_cerca(clave, ca, cb, firma, gen)
                return
            if not (clave[0] in self._ids_cerca() and clave[1] in self._ids_cerca()) and not self.guardia.hay_lugar(EST_FUSION_MB):
                self._estado.pop(clave, None)                  # mezcla lejana y poca memoria: se deja para más tarde
                return
            lejana = not (clave[0] in self._prio_ids and clave[1] in self._prio_ids)
            if lejana and (self._prio_pend or self._lej_fu):      # algo más urgente espera o ya hay una lejana calculándose: se deja para después
                self._estado.pop(clave, None)
                return
            self._estado[clave] = "calculando"
            if lejana:
                self._lej_fu.add(clave)
            try:
                fut = self.pool_fu.submit(trabajo.renderizar_par, ruta_a, A.a_dict(), ruta_b, B.a_dict(), asdict(plan), ajustes, ca, cb, firma)
            except Exception as e:                           # noqa: BLE001
                self._lej_fu.discard(clave)
                if not self._rehacer_pool("fu", e):
                    raise
                self._estado.pop(clave, None)                # el pool se renovó: este par se reintenta en el próximo ciclo
                self.fusiones.pop(clave, None)
                return
            fut.add_done_callback(lambda f, k=clave, g=gen: self._fusion_lista(k, f, g, ca, cb, firma))
        except Exception as e:                               # noqa: BLE001
            self._estado[clave] = "error"
            self.fusiones[clave] = "error"
            self.mensaje = f"No se pudo preparar la fusión: {e}"
            self._rebuild = True

    def _cargar_si_cerca(self, clave: tuple[int, int], ca: str, cb: str, firma: str, gen: int) -> None:
        """(hilo) Trae de la base la fusión guardada. Si el par todavía está lejos en la cola, queda en la base y se trae al acercarse."""
        with self.lock:
            if gen != self._gen:
                return
            ids = self._ids_cerca()
        if clave[0] not in ids or clave[1] not in ids:
            return
        try:
            blob = basedatos.leer_fusion(ca, cb, firma)
            if blob is None:
                raise ValueError("no está en la base")
            f = desempaquetar(blob)
        except ValueError:                                   # dañada o borrada: se descarta y se vuelve a calcular (máximo 2 veces)
            basedatos.borrar_fusion(ca, cb, firma)
            n = self._reintentos.get(clave, 0) + 1
            self._reintentos[clave] = n
            if n > 2:
                self._estado[clave] = "error"
                self.fusiones[clave] = "error"
                self.mensaje = "No se pudo leer la fusión guardada en la base de datos"
            else:
                self._estado.pop(clave, None)
                self.fusiones.pop(clave, None)
            return
        if gen == self._gen:
            self.fusiones[clave] = f
            self._rebuild = True

    def _fusion_lista(self, clave: tuple[int, int], fut: Future, gen: int, ca: str, cb: str, firma: str) -> None:
        """Se llama cuando termina el cálculo: la fusión ya quedó guardada en la base; si hace falta ya mismo, se trae a memoria."""
        self._lej_fu.discard(clave)
        if gen != self._gen:
            return
        try:
            res = fut.result()
        except Exception as e:                               # noqa: BLE001
            if self._rehacer_pool("fu", e):
                self._estado.pop(clave, None)
                self.fusiones.pop(clave, None)
                return
            self._estado[clave] = "error"
            self.fusiones[clave] = "error"
            self.mensaje = f"Falló la fusión: {e}"
            self._rebuild = True
            return
        self._estado[clave] = "hecha"
        if isinstance(res, Fusion):                          # no se pudo guardar en la base: se usa igual si está cerca
            with self.lock:
                ids = self._ids_cerca()
            if clave[0] in ids and clave[1] in ids:
                self.fusiones[clave] = res
                self._rebuild = True
            return
        try:
            self.pool_db.submit(self._cargar_si_cerca, clave, ca, cb, firma, gen)
        except RuntimeError:
            pass

    def invalidar_fusiones(self) -> None:
        """Descarta las fusiones calculadas (p. ej. al cambiar el largo preferido)."""
        self._evento("invalidar_fusiones()")
        with self.lock:
            self._gen += 1
            self.fusiones = {k: v for k, v in self.fusiones.items() if k in self._claves_en_uso()}
            self.planes.clear()
            self._ya.clear()
            self._adapt.clear()
            self._adapt_pedidos.clear()
            self._ya_pre.clear()
            self._ya_grillas.clear()
            self._ya_tok += 1
            self._aviso("")
            self._estado.clear()
            self._reintentos.clear()
            self._rebuild = True

    def _claves_en_uso(self) -> set:
        usadas = set()
        if self.cur is not None and self.cur.tipo == "fus":
            usadas.add((self.cur.pa.id, self.cur.pb.id))
        return usadas

    def _podar(self) -> None:
        i0 = self._indice_actual()
        ids_cerca = {p.id for p in self.pistas[max(0, i0 - 1):i0 + 3]}
        for k in list(self.fusiones):
            if not (k[0] in ids_cerca and k[1] in ids_cerca):
                self.fusiones.pop(k, None)
        for k in list(self._adapt):
            if not (k[0] in ids_cerca and k[1] in ids_cerca):
                self._adapt.pop(k, None)
                self._adapt_pedidos.pop(k, None)
        for k in list(self._ya):                            # la mezcla pedida con ⚡ vale mientras su tema A siga siendo el actual
            if k[0] not in ids_cerca or k[1] not in ids_cerca or (self._ya[k] is not None and not self._ya_vigente(k)):
                self._evento(f"⚡ quitada por poda: cerca={k[0] in ids_cerca and k[1] in ids_cerca} vigente={self._ya_vigente(k)}")
                self._ya.pop(k, None)
        # audio decodificado: solo el actual y los siguientes que indique el perfil (con poca RAM, uno)
        mant = {p.id for p in self.pistas[max(0, i0):i0 + 1 + self.perfil.audio_siguientes]}
        for p in self.pistas:
            if p.id not in mant and p.audio is not None and not (self.cur and self.cur.pista is p):
                p.audio = None

    # ------------------------------------------------------------------ audio decodificado
    def _asegurar_audio(self, p: Pista) -> None:
        if p.audio is not None or p._cargando:
            return
        p._cargando = True

        def tarea() -> None:
            try:
                p.audio = cargar_estereo(p.ruta)
            except Exception as e:                       # noqa: BLE001
                p.estado, p.error = "error", str(e)
            finally:
                p._cargando = False
                self._rebuild = True
        threading.Thread(target=tarea, daemon=True).start()

    # ------------------------------------------------------------------ línea de reproducción
    def _indice_actual(self) -> int:
        c = self.cur
        if c is None:
            return getattr(self, "_i_inicio", 0)
        p = c.pb if c.tipo == "fus" else c.pista
        return self.pistas.index(p) if p in self.pistas else min(getattr(self, "_i_ult", 0), max(0, len(self.pistas) - 1))

    def _cadena(self, i: int, s_ini: int, fade_in: int = 0) -> list[Tramo]:
        tr: list[Tramo] = []
        while i < len(self.pistas):
            p = self.pistas[i]
            sig = self.pistas[i + 1] if i + 1 < len(self.pistas) else None
            fus = self._fusion_desde(p, sig, s_ini) if sig is not None else None
            if fus is not None and s_ini < fus.a_inicio - int(0.25 * SR):
                tr.append(Tramo("solo", p, s_ini, fus.a_inicio, fade_in=fade_in))
                tr.append(Tramo("fus", fus=fus, pa=p, pb=sig))
                s_ini, fade_in = fus.b_continua, 0
                i += 1
                continue
            fin = p.n_muestras
            tr.append(Tramo("solo", p, s_ini, fin, fade_in=fade_in, fade_out=FADE_SIN_FUSION))
            i, s_ini, fade_in = i + 1, 0, (FADE_SIN_FUSION if i + 1 < len(self.pistas) else 0)
        return tr

    def _reconstruir(self) -> None:
        with self.lock:
            self._rebuild = False
            if not self.pistas:
                self.cur, self.futuro, self.off = None, [], 0
                self.sonando = False
                return
            cur = self.cur
            if cur is None:
                return
            if cur.tipo == "fus":
                i = self.pistas.index(cur.pb) if cur.pb in self.pistas else 0
                nueva = self._cadena(i, cur.fus.b_continua)
                with self.lock_cb:
                    self.futuro = nueva
            else:
                if cur.pista in self.pistas:
                    i = self.pistas.index(cur.pista)
                else:
                    i = min(getattr(self, "_i_ult", 0), len(self.pistas) - 1)
                    cur.pista = self.pistas[i]
                self._i_ult = i
                pos = cur.s0 + self.off
                nueva = self._cadena(i, pos)
                if nueva and nueva[0].tipo == "solo":
                    with self.lock_cb:
                        cur.s1, cur.fade_out = nueva[0].s1, nueva[0].fade_out
                        self.futuro = nueva[1:]
                else:
                    with self.lock_cb:
                        self.futuro = nueva
            for t in self.futuro[:2]:
                p = t.pb if t.tipo == "fus" else t.pista
                if p is not None:
                    self._asegurar_audio(p)
            if cur.pista is not None:
                self._asegurar_audio(cur.pista)
            self._programar()

    def _vigilar_ya(self) -> None:
        """Cada segundo comprueba que la línea realmente tenga la mezcla ⚡ puesta donde prometió; si no, la rearma y lo registra."""
        ahora = time.time()
        if not self._ya or ahora - self._t_vigia < 1.0:
            return
        self._t_vigia = ahora
        c = self.cur
        if c is None or c.tipo != "solo" or c.pista not in self.pistas:
            return
        i = self.pistas.index(c.pista)
        if i + 1 >= len(self.pistas):
            return
        f = self._ya.get((c.pista.id, self.pistas[i + 1].id))
        if f is None:
            return
        pos = (c.s0 + self.off) / SR
        faltan = f.a_inicio / SR - pos
        ok = abs(c.s1 - f.a_inicio) < 2 and bool(self.futuro) and self.futuro[0].tipo == "fus" and self.futuro[0].fus is f
        if not ok and faltan > 0.5:
            self._evento(f"⚠ vigía: la línea NO tiene la mezcla ⚡ (s1={c.s1 / SR:.1f}s, a_inicio={f.a_inicio / SR:.1f}s, futuro={[t.tipo for t in self.futuro[:3]]}): se rearma")
            self._rebuild = True
        if faltan > 0:
            self.aviso_ya = f"⚡ La mezcla entra en {faltan:.0f} s"
            self._aviso_t = ahora

    def revisar(self) -> None:
        """Lo llama la interfaz con un temporizador: reconstruye la línea si algo cambió."""
        self._vigilar_ya()
        if self._ya_calculando:
            self.aviso_ya = f"⚡ Calculando la mezcla… {int(time.time() - self._ya_t0)} s"
            self._aviso_t = time.time()
        elif self._ya_hasta and time.time() >= self._ya_hasta:
            self._ya_hasta = 0.0
            self._rebuild = True
        if self.aviso_ya and time.time() - self._aviso_t > 12.0 and not self.aviso_ya.startswith(("⚡ Calculando", "⚡ Buscando", "⚡ Se pasó")):
            self.aviso_ya = ""                                  # los avisos finales del ⚡ se borran solos
        if self._rebuild:
            self._reconstruir()
        elif self.cur is None and self.pistas:
            self._programar()
        if self._pend_an:
            self._actualizar_prioridad()
            self._despachar_an()                                 # por si el análisis estaba frenado por falta de memoria o por esperar las mezclas urgentes

    # ------------------------------------------------------------------ control
    def reproducir(self, i: int | None = None, seg: float = 0.0) -> None:
        with self.lock:
            if not self.pistas:
                return
            if i is None and self.cur is not None and not self.fin_lista:
                self.sonando = True
                self._abrir_stream()
                return
            i = 0 if i is None else max(0, min(i, len(self.pistas) - 1))
            self._i_inicio = i
            self.buscar(i, seg)
            self.sonando = True
        self._abrir_stream()

    def pausar(self) -> None:
        self.sonando = False

    def alternar(self) -> None:
        if self.sonando:
            self.pausar()
        else:
            self.reproducir()

    def buscar(self, i: int, seg: float) -> None:
        """Salta al segundo `seg` del tema número `i` (si hay fusión lista, no pasa del punto donde empieza)."""
        with self.lock:
            p = self.pistas[i]
            self._asegurar_audio(p)
            s = int(max(0.0, seg) * SR)
            sig = self.pistas[i + 1] if i + 1 < len(self.pistas) else None
            fus = self._fusion(p, sig) if sig is not None else None
            if fus is not None:
                s = min(s, max(0, fus.a_inicio - int(0.3 * SR)))
            self.fin_lista = False
            nuevo = Tramo("solo", p, s, p.n_muestras, fade_in=int(0.03 * SR))
            with self.lock_cb:
                self.cur = nuevo
                self.off = 0
            self._i_ult = i
            self._rebuild = True
        self._reconstruir()

    # ------------------------------------------------------------------ cuando la mezcla prevista ya quedó atrás
    def _fusion_desde(self, p: Pista, sig: Pista, s_ini: int) -> Fusion | None:
        """Mezcla de `p` hacia `sig` que todavía esté por delante de la muestra `s_ini`. Si la prevista ya quedó atrás (pasa con
        temas cortos: el tema recién vuelve a sonar solo después de la mezcla anterior), pide buscar OTRO punto de salida
        más adelante y mientras tanto usa el cruce simple al final del tema: nunca termina sin mezclar."""
        f = self._fusion(p, sig)
        if f is None or s_ini < f.a_inicio - int(0.25 * SR):
            return f
        ad = self._adapt.get((p.id, sig.id))
        if ad is not None and s_ini < ad.a_inicio - int(0.25 * SR):
            return ad
        self._pedir_adaptada(p, sig, s_ini)
        return self._cruce_simple(p, sig)

    def _pedir_adaptada(self, a: Pista, b: Pista, s_ini: int) -> None:
        clave = (a.id, b.id)
        previo = self._adapt_pedidos.get(clave)
        if previo is not None and s_ini <= previo + int(0.5 * SR):
            return
        if not (a.analisis and b.analisis) or self._cerrando:
            return
        self._adapt_pedidos[clave] = s_ini
        try:
            self.pool_db.submit(self._adaptar, a, b, s_ini, self._gen)
        except RuntimeError:
            pass

    def _adaptar(self, a: Pista, b: Pista, s_ini: int, gen: int) -> None:
        """(hilo) Busca un punto de salida de `a` después de `s_ini` y deja lista esa mezcla (se guarda en la base)."""
        clave = (a.id, b.id)
        try:
            if gen != self._gen or self._cerrando:
                return
            A, B = a.analisis, b.analisis
            pos = self._ya_pos(a)
            tiempos = np.array([A.tiempo_compas(k) for k in range(A.n_compases)])
            comp = 240.0 / max(40.0, float(A.bpm))             # segundos que dura un compás
            plan = None
            for solo in (SOLO_OK, SOLO_MIN):                   # el tema suena solo un rato (cómodo; si no alcanza, el mínimo) antes de empezar a salir
                t_ref = s_ini / SR + solo * comp
                if pos is not None:                              # ya está sonando: hay que dejar tiempo para calcular
                    t_ref = max(t_ref, pos + max(self._t_ya_nat, self._t_ya_cap) * 1.3 + 6.0)
                k0 = int(np.searchsorted(tiempos, t_ref))
                plan = planificar(A, B, (8,), a_desde=k0, estructura=self._est(), voces=self._vz()) or planificar(A, B, (4,), a_desde=k0, estructura=self._est(), voces=self._vz())
                if plan is not None:
                    break
            if plan is None:                                 # no queda lugar: queda el cruce simple al final del tema
                return
            ca, cb = basedatos.clave_tema(a.ruta), basedatos.clave_tema(b.ruta)
            if ca is None or cb is None:
                return
            aj = self._ajustes_ya()
            firma = basedatos.firma_fusion(asdict(plan), aj, VERSION_FUSION)
            if basedatos.existe_fusion(ca, cb, firma):
                self._adaptada_lista(clave, ca, cb, firma, None, gen)
                return
            lo_a, hi_a, lo_b, hi_b = trabajo.ventana_par(A, B, plan)
            fut = self.pool_fu.submit(trabajo.renderizar_par, a.ruta, A.a_dict(), b.ruta, B.a_dict(), asdict(plan), aj, ca, cb, firma,
                                      self._recorte_memoria(a, lo_a, hi_a), self._recorte_memoria(b, lo_b, hi_b))
            fut.add_done_callback(lambda f, k=clave, x=ca, y=cb, z=firma, g=gen: self._adaptada_lista(k, x, y, z, f, g))
        except Exception as e:                               # noqa: BLE001
            self._adapt_pedidos.pop(clave, None)
            self.mensaje = f"No se pudo adaptar la mezcla: {e}"

    def _adaptada_lista(self, clave: tuple[int, int], ca: str, cb: str, firma: str, fut: Future | None, gen: int) -> None:
        if gen != self._gen:
            return
        try:
            fus = fut.result() if fut is not None else None
            if not isinstance(fus, Fusion):
                blob = basedatos.leer_fusion(ca, cb, firma)
                if blob is None:
                    raise ValueError("no está en la base")
                fus = desempaquetar(blob)
        except Exception:                                    # noqa: BLE001
            self._adapt_pedidos.pop(clave, None)
            return
        self._adapt[clave] = fus
        self._rebuild = True

    # ------------------------------------------------------------------ ⚡ puntos de salida rápida (se dejan guardados en la base)
    PASO_YA = 8                                              # compases entre un punto de salida rápida y el siguiente

    def _ya_planes(self, a: Pista, b: Pista) -> list[Plan]:
        """Planes de mezcla corta (8 compases) con salida en varios puntos del tema A (siempre los mismos para el mismo par)."""
        clave = (a.id, b.id)
        ps = self._ya_grillas.get(clave)
        if ps is None:
            A, B = a.analisis, b.analisis
            ps, vistos = [], set()
            k = max(8, int(0.15 * A.n_compases))
            while k < A.n_compases - 12:
                pl = planificar(A, B, (8,), a_desde=k, estructura=self._est(), voces=self._vz())
                if pl is not None and pl.a0 not in vistos:
                    vistos.add(pl.a0)
                    ps.append(pl)
                k += self.PASO_YA
            ps.sort(key=lambda q: q.a0)
            self._ya_grillas[clave] = ps
        return ps

    def _ajustes_ya(self) -> dict:
        aj = self._aj_render()
        aj["motor"] = "natural"                              # el más rápido (solo cae al de capas si los tempos son muy distintos)
        aj["umbral_natural"] = UMBRAL_NATURAL_YA              # en el ⚡ se prioriza la velocidad: el motor natural alcanza hasta ~14 % de diferencia de tempo
        return aj

    def _ya_precalcular(self) -> None:
        """Mientras suena el tema, prepara en segundo plano mezclas cortas con salida en varios puntos y las guarda en la base:
        así el botón ⚡ solo tiene que buscar la que viene y se escucha al instante."""
        c = self.cur
        if c is None or c.tipo != "solo" or c.pista not in self.pistas or time.time() < self._ya_hasta or self._prio_pend:
            return
        i = self.pistas.index(c.pista)
        if i + 1 >= len(self.pistas):
            return
        a, b = self.pistas[i], self.pistas[i + 1]
        clave = (a.id, b.id)
        if not (a.analisis and b.analisis) or clave in self._ya_pre or not self.guardia.hay_lugar(EST_FUSION_MB):
            return
        self._ya_pre.add(clave)
        try:
            self.pool_db.submit(self._ya_pre_tarea, a, b, self._gen)
        except RuntimeError:
            self._ya_pre.discard(clave)

    def _ya_pre_tarea(self, a: Pista, b: Pista, gen: int) -> None:
        """(hilo) Pide calcular y guardar cada punto de salida rápida que todavía no esté en la base."""
        try:
            ca, cb = basedatos.clave_tema(a.ruta), basedatos.clave_tema(b.ruta)
            if ca is None or cb is None:
                return
            A, B = a.analisis, b.analisis
            aj = self._ajustes_ya()
            for pl in self._ya_planes(a, b):
                if gen != self._gen or self._cerrando:
                    return
                firma = basedatos.firma_fusion(asdict(pl), aj, VERSION_FUSION)
                if basedatos.existe_fusion(ca, cb, firma):
                    continue
                lo_a, hi_a, lo_b, hi_b = trabajo.ventana_par(A, B, pl)
                self.pool_fu.submit(trabajo.renderizar_par, a.ruta, A.a_dict(), b.ruta, B.a_dict(), asdict(pl), aj, ca, cb, firma,
                                    self._recorte_memoria(a, lo_a, hi_a), self._recorte_memoria(b, lo_b, hi_b))
        except Exception:                                    # noqa: BLE001
            self._ya_pre.discard((a.id, b.id))               # se reintenta en el próximo ciclo

    def _ya_buscar_guardada(self, a: Pista, b: Pista, pos: float, tok: int) -> bool:
        """(hilo) Busca en la base la mezcla corta más cercana que todavía no pasó. Si está, la mete en la línea al instante."""
        ca, cb = basedatos.clave_tema(a.ruta), basedatos.clave_tema(b.ruta)
        if ca is None or cb is None:
            return False
        A = a.analisis
        aj = self._ajustes_ya()
        for pl in self._ya_planes(a, b):
            t_entra = A.tiempo_compas(pl.a0)
            if t_entra < pos + 3.0:                          # ya pasó (o no da tiempo de cargarla)
                continue
            if t_entra > pos + 30.0:                         # demasiado lejos: mejor calcular una más cercana
                break
            firma = basedatos.firma_fusion(asdict(pl), aj, VERSION_FUSION)
            if not basedatos.existe_fusion(ca, cb, firma):
                continue
            try:
                blob = basedatos.leer_fusion(ca, cb, firma)
                fus = desempaquetar(blob) if blob is not None else None
            except ValueError:
                fus = None
            if fus is None:
                continue
            with self.lock:
                p_act = self._ya_pos(a)
                if tok != self._ya_tok or p_act is None or p_act * SR >= fus.a_inicio - int(0.6 * SR):
                    continue
                self._ya[(a.id, b.id)] = fus
                self._evento(f"⚡ instalada (guardada) a_inicio={fus.a_inicio / SR:.1f}s pos={p_act:.1f}s")
                self._aviso(f"⚡ Mezcla lista (ya guardada): entra en {fus.a_inicio / SR - p_act:.0f} s")
                self._rebuild = True
            return True
        return False

    # ------------------------------------------------------------------ ⚡ mezclar desde donde va
    MARGEN_YA = (0, 1, 2)                                    # cantidad de intentos (cada uno reserva más ventaja que el anterior)

    def _evento(self, texto: str) -> None:
        """Deja una línea en datos/eventos.log (diagnóstico de las mezclas ⚡; se recorta solo si crece)."""
        try:
            from .config import carpeta_datos
            ruta = carpeta_datos() / "eventos.log"
            with self._lock_log:
                if ruta.exists() and ruta.stat().st_size > 300_000:
                    ruta.write_text("", encoding="utf-8")
                with open(ruta, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {texto}\n")
        except Exception:                                    # noqa: BLE001
            pass

    def _aviso(self, texto: str) -> None:
        self.aviso_ya = texto
        self._aviso_t = time.time()

    def _ya_vigente(self, k: tuple[int, int]) -> bool:
        """La mezcla ⚡ de este par sigue valiendo: el tema A suena (o es el anterior de la fusión que está sonando)."""
        c = self.cur
        if c is None:
            return False
        if c.tipo == "fus":
            return (c.pa.id, c.pb.id) == k
        return c.pista is not None and c.pista.id == k[0]

    def fusionar_ya(self) -> bool:
        """⚡ Mezcla el tema que suena con el siguiente desde donde va: busca un punto de salida cercano (unos segundos
        adelante), calcula la mezcla en segundo plano y, cuando está lista, la mete en la línea. Mientras tanto sigue sonando."""
        with self.lock:
            c = self.cur
            if c is None or c.tipo != "solo" or c.pista not in self.pistas:
                self._aviso("⚡ Esperá a que el tema suene solo (no en medio de una mezcla)")
                return False
            i = self.pistas.index(c.pista)
            if i + 1 >= len(self.pistas):
                self._aviso("⚡ No hay un tema siguiente para mezclar")
                return False
            a, b = self.pistas[i], self.pistas[i + 1]
            if not (a.analisis and b.analisis):
                self._aviso("⚡ Falta terminar el análisis de los dos temas")
                return False
            self._ya_tok += 1
            tok = self._ya_tok
            self._ya.pop((a.id, b.id), None)
            self._evento(f"⚡ pedido: «{a.nombre}» → «{b.nombre}» pos={(c.s0 + self.off) / SR:.1f}s tok={tok}")
            self._aviso("⚡ Buscando dónde meter la mezcla…")
        self._rebuild = True
        try:
            self.pool_db.submit(self._ya_preparar, a, b, tok, 0)
        except RuntimeError:
            return False
        return True

    def _ya_pos(self, a: Pista) -> float | None:
        """Segundo actual de `a` si es el tema que suena solo; si no, None."""
        c = self.cur
        if c is None or c.tipo != "solo" or c.pista is not a:
            return None
        return (c.s0 + self.off) / SR

    def _ya_preparar(self, a: Pista, b: Pista, tok: int, intento: int) -> None:
        """(hilo) Elige el compás de salida un poco más adelante de lo que suena y pide calcular esa mezcla."""
        try:
            if tok != self._ya_tok or self._cerrando:
                return
            pos = self._ya_pos(a)
            if pos is None:
                self._aviso("")
                return
            if intento == 0 and self._ya_buscar_guardada(a, b, pos, tok):
                return
            A, B = a.analisis, b.analisis
            tiempos = np.array([A.tiempo_compas(k) for k in range(A.n_compases)])
            plan = self._plan_ya(A, B, pos, tiempos, intento)     # mezcla corta (8 compases) con salida a tiempo para calcularla
            if plan is None:
                self._aviso("⚡ Ya no queda lugar para una mezcla en este tema: entra la siguiente normal")
                return
            ajustes = self._ajustes_ya()
            lo_a, hi_a, lo_b, hi_b = trabajo.ventana_par(A, B, plan)
            aud_a, aud_b = self._recorte_memoria(a, lo_a, hi_a), self._recorte_memoria(b, lo_b, hi_b)    # si ya está decodificado, no se vuelve a leer del disco
            self._ya_t0 = time.time()
            self._ya_hasta = self._ya_t0 + 120.0
            self._ya_calculando = True
            fut = self.pool_ya.submit(trabajo.renderizar_par, a.ruta, A.a_dict(), b.ruta, B.a_dict(), asdict(plan), ajustes, "", "", "", aud_a, aud_b)
            self._aviso(f"⚡ Calculando la mezcla… (entra en ~{tiempos[plan.a0] - pos:.0f} s)")
            fut.add_done_callback(lambda f, aa=a, bb=b, t=tok, n=intento: self._ya_lista(aa, bb, f, t, n))
        except Exception as e:                               # noqa: BLE001
            if tok == self._ya_tok:
                self._ya_termino()
            self._aviso(f"⚡ No se pudo preparar la mezcla: {e}")

    def _est_ya(self, A, B, plan) -> float:
        """Segundos que se estima que tarda calcular esa mezcla (según el motor que le tocaría)."""
        try:
            desv = desviacion_tempo(A, B, plan)
        except Exception:                                    # noqa: BLE001
            desv = 1.0
        return self._t_ya_nat if desv <= UMBRAL_NATURAL_YA else self._t_ya_cap

    def _plan_ya(self, A, B, pos: float, tiempos, intento: int):
        """Plan de mezcla corta cuyo punto de salida deja tiempo para calcularla: primero con el margen del motor rápido y, si ese plan
        necesita el lento, otra vez con el margen del lento (así el punto de salida no queda demasiado cerca)."""
        plan = None
        est = self._t_ya_nat
        for _ in range(2):
            t_min = pos + (est * 1.05 + 1.5) * (1.0 + 0.5 * intento)     # lo justo para calcularla: la mezcla entra en el primer compás posible; si no llega, reintenta con más margen
            k0 = int(np.searchsorted(tiempos, t_min))
            plan = planificar(A, B, (8,), a_desde=k0, estructura=self._est(), voces=self._vz(), cerca=True) or planificar(A, B, (4,), a_desde=k0, estructura=self._est(), voces=self._vz(), cerca=True)
            if plan is None:
                return None
            nuevo = self._est_ya(A, B, plan)
            if nuevo <= est:
                break
            est = nuevo
        return plan

    def _recorte_memoria(self, p: Pista, lo: int, hi: int):
        """Trozo [lo, hi) del audio ya decodificado de `p` (como AudioParcial), o None si no está en memoria."""
        aud = p.audio
        if aud is None:
            return None
        n = int(aud.shape[1])
        lo, hi = max(0, min(lo, n)), max(0, min(hi, n))
        if hi <= lo:
            return None
        from .audio_io import AudioParcial
        return AudioParcial(n, lo, np.ascontiguousarray(aud[:, lo:hi]))

    def _ya_termino(self) -> None:
        self._ya_calculando = False
        self._ya_hasta = 0.0
        self._rebuild = True
        try:
            self.pool_db.submit(self._despachar_an)             # retoma el trabajo en segundo plano
        except RuntimeError:
            pass

    def _registrar_ya(self, fus: Fusion, medido: float, intento: int) -> None:
        """Deja en datos/ya_tiempos.log cuánto tardó cada mezcla ⚡ y en qué (sirve para afinar la velocidad)."""
        try:
            from .config import carpeta_datos
            t = getattr(fus, "tiempos", {})
            linea = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} total={medido:.1f}s decodificar={t.get('decodificar')}s "
                     f"calcular={t.get('calcular')}s motor={t.get('motor')} L={fus.plan.L} intento={intento + 1} "
                     f"cpu={__import__('os').cpu_count()} ram_libre={self.perfil.reserva_mb}\n")
            with open(carpeta_datos() / "ya_tiempos.log", "a", encoding="utf-8") as f:
                f.write(linea)
        except Exception:                                    # noqa: BLE001
            pass

    def _ya_lista(self, a: Pista, b: Pista, fut: Future, tok: int, intento: int) -> None:
        """Cuando termina el cálculo: si todavía hay tiempo antes del punto de salida, entra a la línea; si no, reintenta más adelante."""
        if tok != self._ya_tok or self._cerrando:             # lo reemplazó un pedido más nuevo: ese es el que manda
            self._evento(f"⚡ resultado descartado: lo reemplazó un pedido más nuevo (tok {tok} ≠ {self._ya_tok})")
            return
        self._ya_termino()
        try:
            fus = fut.result()
        except Exception as e:                               # noqa: BLE001
            try:
                self.pool_ya.shutdown(wait=False, cancel_futures=True)
            except Exception:                                # noqa: BLE001
                pass
            self.pool_ya = _crear_pool(1, 50, trabajo.iniciar_hijo_normal)    # se renueva para el próximo intento
            self._aviso(f"⚡ Falló el cálculo de la mezcla: {e}")
            return
        if not isinstance(fus, Fusion):
            self._aviso("⚡ No se pudo preparar la mezcla")
            return
        medido = time.time() - self._ya_t0
        if getattr(fus, "motor", "") == "natural":            # cuánto tarda en esta PC con cada motor (para reservar el margen justo)
            self._t_ya_nat = medido if medido > self._t_ya_nat else 0.5 * self._t_ya_nat + 0.5 * medido
        else:
            self._t_ya_cap = medido if medido > self._t_ya_cap else 0.5 * self._t_ya_cap + 0.5 * medido
        self._registrar_ya(fus, medido, intento)
        with self.lock:
            pos = self._ya_pos(a)
            if pos is None:                                    # cambió el tema o ya empezó otra mezcla: se descarta
                self._evento("⚡ descartada: ya no suena ese tema solo")
                self._aviso("")
                return
            if pos * SR < fus.a_inicio - int(0.6 * SR):
                self._ya[(a.id, b.id)] = fus
                self._evento(f"⚡ instalada (calculada) a_inicio={fus.a_inicio / SR:.1f}s pos={pos:.1f}s")
                self._aviso(f"⚡ Mezcla lista: entra en {fus.a_inicio / SR - pos:.0f} s")
                self._rebuild = True
                return
        if intento + 1 < len(self.MARGEN_YA):                  # llegó tarde (el tema ya pasó el punto): se busca uno más adelante
            self._aviso("⚡ Se pasó el punto, busco uno más adelante…")
            try:
                self.pool_db.submit(self._ya_preparar, a, b, tok, intento + 1)
            except RuntimeError:
                pass
        else:
            self._aviso("⚡ No llegué a calcularla a tiempo: entra la mezcla normal")

    def siguiente(self) -> None:
        with self.lock:
            i = self._indice_actual()
            if i + 1 < len(self.pistas):
                self.buscar(i + 1, 0.0)

    def anterior(self) -> None:
        with self.lock:
            i = self._indice_actual()
            c = self.cur
            if c is not None and c.tipo == "solo" and (c.s0 + self.off) / SR > 4.0:
                self.buscar(i, 0.0)
            else:
                self.buscar(max(0, i - 1), 0.0)

    # ------------------------------------------------------------------ salida de audio
    def _abrir_stream(self) -> None:
        if self.stream is not None or sd is None:
            if sd is None and not self.error_audio:
                self.error_audio = "No está instalada la librería de audio (sounddevice)."
            return
        try:
            self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32", latency="high", callback=self._callback)
            self.stream.start()
            self.error_audio = ""
        except Exception as e:                           # noqa: BLE001
            self.stream = None
            self.error_audio = f"No se pudo abrir la salida de audio: {e}"

    def _callback(self, out: np.ndarray, frames: int, tiempo, estado) -> None:
        out[:] = 0.0
        if not self.sonando or self.cur is None:
            self._vol_act = self.volumen
            return
        if not self.lock_cb.acquire(False):                # la interfaz está cambiando la línea justo ahora: este bloque va en silencio
            return
        try:
            self._llenar(out, frames)
        finally:
            self.lock_cb.release()
        # volumen con rampa (sin "clicks") y datos para los medidores
        v0, v1 = self._vol_act, self.volumen
        if v0 != v1:
            out *= np.linspace(v0, v1, frames, dtype=np.float32)[:, None]
            self._vol_act = v1
        else:
            out *= v1
        mono = out.mean(axis=1)
        p = self.anillo_pos
        n = len(mono)
        if p + n <= len(self.anillo):
            self.anillo[p:p + n] = mono
        else:
            k = len(self.anillo) - p
            self.anillo[p:] = mono[:k]
            self.anillo[:n - k] = mono[k:]
        self.anillo_pos = (p + n) % len(self.anillo)
        self.pico = (float(np.abs(out[:, 0]).max()), float(np.abs(out[:, 1]).max()))

    def _llenar(self, out: np.ndarray, frames: int) -> None:
        lleno = 0
        guardas = 0
        while lleno < frames and guardas < 8:
            guardas += 1
            t = self.cur
            if t is None:
                break
            n = min(frames - lleno, t.largo - self.off)
            if n <= 0:
                if not self._avanzar():
                    self.sonando = False
                    self.fin_lista = True
                    break
                continue
            if t.tipo == "fus":
                out[lleno:lleno + n] = t.fus.audio[:, self.off:self.off + n].T
            else:
                aud = t.pista.audio
                if aud is None:                           # el tema todavía se está decodificando: se espera sin avanzar
                    self._rebuild = True
                    break
                a = t.s0 + self.off
                blk = aud[:, a:a + n]
                m = blk.shape[1]
                if m > 0:
                    seg = blk.T * t.pista.gan
                    if t.fade_in and self.off < t.fade_in:
                        r = np.minimum(1.0, (np.arange(m) + self.off) / t.fade_in).astype(np.float32)
                        seg = seg * r[:, None]
                    resta = t.largo - self.off
                    if t.fade_out and resta <= t.fade_out + m:
                        r = np.clip((resta - np.arange(m)) / t.fade_out, 0.0, 1.0).astype(np.float32)
                        seg = seg * r[:, None]
                    out[lleno:lleno + m] = limitador(seg)
            lleno += n
            self.off += n

    def _avanzar(self) -> bool:
        f = self.futuro
        if not f:
            return False
        self.cur = f[0]
        self.futuro = f[1:]
        self.off = 0
        self._rebuild = True
        if self._ya or self._ya_calculando:
            self._evento(f"avanza → {self.cur.tipo}")
        return True

    # ------------------------------------------------------------------ información para la interfaz
    def ultimo_bloque(self, n: int = 2048) -> np.ndarray:
        p = self.anillo_pos
        a = np.concatenate([self.anillo[p:], self.anillo[:p]])
        return a[-n:]

    def snapshot(self) -> dict:
        """Estado instantáneo para dibujar (se llama ~60 veces por segundo; debe ser barato)."""
        c, off = self.cur, self.off
        s = dict(sonando=self.sonando, tipo=None, pa=None, pb=None, pos_a=0.0, pos_b=0.0, u=0.0, fusion=None,
                 proxima=None, pico=self.pico, fin=self.fin_lista, i_a=-1, i_b=-1)
        if c is None:
            return s
        try:
            if c.tipo == "fus":
                f = c.fus
                u = min(1.0, off / max(1, f.n))
                s.update(tipo="fus", pa=c.pa, pb=c.pb, u=u, fusion=f,
                         pos_a=(f.a_inicio + off) / SR, pos_b=(f.b_continua - (f.n - off)) / SR,
                         i_a=self.pistas.index(c.pa) if c.pa in self.pistas else -1,
                         i_b=self.pistas.index(c.pb) if c.pb in self.pistas else -1)
            else:
                p = c.pista
                i = self.pistas.index(p) if p in self.pistas else -1
                s.update(tipo="solo", pa=p, pos_a=(c.s0 + off) / SR, i_a=i)
                if 0 <= i < len(self.pistas) - 1:
                    sig = self.pistas[i + 1]
                    s["pb"], s["i_b"] = sig, i + 1
                    s["proxima"] = self._fusion(p, sig)
        except ValueError:
            pass
        return s

    def bloques(self) -> list[dict]:
        """Estructura de toda la mezcla (tramos normales y fusiones) con su tiempo absoluto, para dibujar la línea."""
        res, t = [], 0.0
        with self.lock:
            pistas = list(self.pistas)
        s_ini = 0
        c = self.cur
        ia = self._indice_actual() if c is not None else 0
        ci = ia if (c is not None and c.tipo == "solo") else -1
        for i, p in enumerate(pistas):
            sig = pistas[i + 1] if i + 1 < len(pistas) else None
            if i == ci and c.s1 > c.s0:                       # el tema que suena: se dibuja el tramo real (de dónde entró a dónde sale), no el teórico
                s_ini = c.s0
            fus = (self._fusion_desde(p, sig, s_ini) if i >= ia else self._fusion(p, sig)) if sig is not None else None      # igual que la línea real: la mezcla tiene que quedar adelante de donde entra el tema
            if fus is not None and s_ini >= fus.a_inicio - int(0.25 * SR):
                fus = None
            fin = fus.a_inicio if fus else p.n_muestras
            if i == ci and c.s1 > c.s0:
                fin = c.s1
            dur = max(0, fin - s_ini) / SR
            res.append(dict(tipo="solo", i=i, t0=t, t1=t + dur, s0=s_ini / SR, s1=fin / SR))
            t += dur
            if fus is not None:
                res.append(dict(tipo="fus", i=i, t0=t, t1=t + fus.duracion, fus=fus))
                t += fus.duracion
                s_ini = fus.b_continua
            else:
                s_ini = 0
        return res

    def pos_global(self, bloques: list[dict] | None = None) -> float:
        c = self.cur
        if c is None:
            return 0.0
        for b in (bloques if bloques is not None else self.bloques()):
            if c.tipo == "solo" and b["tipo"] == "solo" and b["i"] == self._indice_actual():
                return b["t0"] + max(0.0, (c.s0 + self.off) / SR - b["s0"])
            if c.tipo == "fus" and b["tipo"] == "fus" and b.get("fus") is c.fus:
                return b["t0"] + self.off / SR
        return 0.0

    def buscar_global(self, t: float) -> None:
        for b in self.bloques():
            if b["t0"] <= t <= b["t1"]:
                if b["tipo"] == "solo":
                    self.buscar(b["i"], b["s0"] + (t - b["t0"]))
                else:
                    with self.lock:
                        a, p = self.pistas[b["i"]], self.pistas[b["i"] + 1]
                        nuevo = Tramo("fus", fus=b["fus"], pa=a, pb=p)
                        with self.lock_cb:
                            self.cur = nuevo
                            self.off = int((t - b["t0"]) * SR)
                        self._rebuild = True
                    self._reconstruir()
                return

    def cerrar(self) -> None:
        self._cerrando = True
        self.sonando = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:                                # noqa: BLE001
            pass
        self.stream = None
        for pool in (self.pool_an, self.pool_fu, self.pool_ya, self.pool_db):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
                for proc in list(getattr(pool, "_processes", {}).values()):       # que no queden procesos trabajando tras cerrar
                    proc.terminate()
            except Exception:                            # noqa: BLE001
                pass
