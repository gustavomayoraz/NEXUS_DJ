"""Cerebro de la fusión: elige DÓNDE sale un tema, DÓNDE entra el otro, cuántos compases dura y cuánto hay que
subir o bajar el tono del que entra para que armonicen.

Se puntúan todas las combinaciones (compás de salida de A × compás de entrada de B × duración) con:
armonía (correlación de cromas con desplazamiento de tono), tempo, energía, solape de voces, frases de 8 compases
y la posición en cada tema (salir tarde de A, entrar temprano en B).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analisis import Analisis, NOTAS


SOLO_OK = 16               # compases de solo «cómodos»: si el punto elegido no se los deja a un tema, ese tema suena casi entero
SOLO_MIN = 8               # compases mínimos que un tema suena solo entre el fin de su entrada y el comienzo de su salida (con «dónde empieza la mezcla»)


@dataclass
class Plan:
    a0: int                    # compás de A donde empieza la fusión
    b0: int                    # compás de B donde entra
    L: int                     # compases de la fusión medidos en la rejilla de A
    k: float                   # 1 = mismo pulso; 2 / 0.5 = B se toma al doble / mitad de tiempo
    shift: float               # semitonos que se le suben a B mientras armoniza con A (incluye afinación)
    modo: str                  # "profunda" o "limpia"
    puntaje: float
    armonia: float             # correlación de cromas lograda (0..1)
    razon: str = ""
    detalle: dict = field(default_factory=dict)

    @property
    def Lb(self) -> int:
        """Compases de B que consume la fusión."""
        return max(1, int(round(self.L / self.k)))

    @property
    def n_beats(self) -> int:
        return 4 * self.L

    def resumen(self) -> str:
        s = f"{self.shift:+.1f} st" if abs(self.shift) >= 0.05 else "tono igual"
        return f"{'Fusión profunda' if self.modo == 'profunda' else 'Fusión limpia'} · {self.L} compases · B {s}"


def _suave(x: np.ndarray, t: int) -> np.ndarray:
    return np.convolve(x, np.ones(t) / t, mode="same")


def _factor_tempo(bpm_a: float, bpm_b: float) -> float:
    r = bpm_b / bpm_a
    mejor, kmej = abs(np.log2(r)), 1.0
    for k in (2.0, 0.5):
        v = abs(np.log2(r * k))
        if v < mejor - 0.12:
            mejor, kmej = v, k
    return kmej


def _tope_L(X: Analisis, largos: tuple[int, ...]) -> int:
    """Fusión más larga con la que X puede ENTRAR y todavía le queda lugar para sonar solo (SOLO_MIN) y salir con una fusión de 8 compases."""
    aud = np.where(X.energia > 0.06)[0]
    b0 = int(aud[0]) if len(aud) else 0
    hasta = int(X.ultimo_compas)
    ok = [L for L in sorted(largos) if b0 + int(np.ceil(L * 1.25)) + SOLO_MIN + 4 + 8 <= hasta]
    return max(ok) if ok else min(largos)


def _sale_desde(X: Analisis, largos: tuple[int, ...]) -> int:
    """Primer compás en que X puede empezar a salir sin pisar su propia entrada (más SOLO_MIN compases solo)."""
    aud = np.where(X.energia > 0.06)[0]
    b0 = int(aud[0]) if len(aud) else 0
    return max(int(0.30 * X.n_compases), 8, b0 + int(np.ceil(_tope_L(X, largos) * 1.25)) + SOLO_MIN)


def planificar(A: Analisis, B: Analisis, largos: tuple[int, ...] = (16, 8), forzar: dict | None = None,
               a_desde: int | None = None, estructura: bool = False, voces: bool = False, inicio: float | None = None,
               cerca: bool = False) -> Plan | None:
    """Devuelve el mejor Plan de fusión de A hacia B.

    Con `a_desde` (compás de A), la fusión tiene que empezar en ese compás o después (se prefiere lo más cercano):
    sirve para mezclar "desde donde va" el tema. Si no queda lugar para ninguna fusión, devuelve None.
    Con `cerca` (solo junto a `a_desde`, para el ⚡), la mezcla se busca lo más pegada posible a ese compás (unos 5 compases como mucho), sin exigir
    el solo mínimo previo: el usuario pidió mezclar ya.
    Con `inicio` (0..1), la fusión empieza en ese punto de A (por ejemplo 0.5 = a la mitad del tema), con un margen de ±4 compases
    para caer en una frase.
    Con `estructura`, además se premia que la fusión empiece y termine en cambios de sección y se castiga pisar un estribillo con otro."""
    nA, nB = A.n_compases, B.n_compases
    if a_desde is not None and not cerca:                             # mezcla rápida (no ⚡): aun así A tiene que terminar de entrar y sonar solo un rato
        aud_a = np.where(A.energia > 0.06)[0]
        a_desde = max(int(a_desde), (int(aud_a[0]) if len(aud_a) else 0) + int(np.ceil(_tope_L(A, (16, 8)) * 1.25)) + SOLO_MIN + 4)
    k = _factor_tempo(A.bpm, B.bpm)
    razon_t = abs(np.log2(B.bpm * k / A.bpm))
    cum_a = np.vstack([np.zeros((1, 12)), np.cumsum(A.chroma, axis=0)])
    cum_b = np.vstack([np.zeros((1, 12)), np.cumsum(B.chroma, axis=0)])
    cum_eA = np.concatenate([[0], np.cumsum(A.energia)])
    cum_eB = np.concatenate([[0], np.cumsum(B.energia)])
    # primer compás "fuerte" de cada tema: referencia de las frases
    pa = int(np.argmax(A.energia > 0.7)) if (A.energia > 0.7).any() else 0
    pb = int(np.argmax(B.energia > 0.7)) if (B.energia > 0.7).any() else 0
    audible_b = np.where(B.energia > 0.06)[0]
    b_min = int(audible_b[0]) if len(audible_b) else 0
    if estructura:
        from .estructura import analizar as _estr
        estA, estB = _estr(A), _estr(B)
        cum_fA = np.concatenate([[0], np.cumsum(estA.fuerte)])
        cum_fB = np.concatenate([[0], np.cumsum(estB.fuerte)])
    obj_a = obj_b = None
    if inicio is not None and a_desde is None:
        def _obj(X, nX):                                            # el punto que eligió el usuario (20 % … 100 %): se respeta tal cual, 100 % = el tema suena entero
            return max(8, int(round(float(inicio) * nX)))
        obj_a, obj_b = _obj(A, nA), _obj(B, nB)
    tope_B = _tope_L(B, largos)
    tope_A = _tope_L(A, largos)
    sale_B = _sale_desde(B, largos)
    mejor: tuple | None = None
    largos_it = largos
    if obj_b is not None:                                          # con «dónde empieza la mezcla»: B tiene que terminar de entrar (y sonar solo) antes de su propia salida;
        todos = sorted(set(largos) | {8, 4}, reverse=True)         # si la fusión pedida no entra en ese espacio, se usa una más corta para respetar el punto elegido
        caben = [L for L in todos if b_min + max(1, int(round(L / k))) + SOLO_MIN + 4 <= obj_b]
        if caben:
            largos_it = tuple(L for L in largos if L in caben) or (caben[0],)
        else:
            largos_it = (min(todos),)
    for L in largos_it:
        Lb = max(1, int(round(L / k)))
        if a_desde is None and (L > tope_B or L > tope_A):                       # B es corto: con esta fusión no le queda lugar para sonar solo
            continue
        a_ini = _sale_desde(A, largos) if a_desde is None else max(0, int(a_desde))
        if a_desde is None:                                        # tiempo parejo: un tema corto no sale enseguida, suena casi entero
            a_ini = max(a_ini, min(max(0, min(A.ultimo_compas - L, nA - L - 2)), _sale_desde(A, largos) + SOLO_OK - SOLO_MIN))
        a_fin = min(A.ultimo_compas - L, nA - L - 2)
        obj = None
        if a_desde is not None:
            a_fin = min(a_fin, a_ini + (5 if cerca else 32))      # cerca de donde va: ⚡ no más de ~12 compases adelante, el resto ~32
        elif obj_a is not None and a_fin >= 0:                     # el usuario eligió dónde empieza la mezcla: ±4 compases alrededor de ese punto
            obj = int(min(max(0, obj_a), a_fin))
            a_ini, a_fin = max(0, obj - 4), min(a_fin, obj + 4)
        b_fin = min(int(0.45 * nB), nB - Lb - 2)
        if a_desde is None:
            b_fin = min(b_fin, max(b_min, sale_B - SOLO_MIN - Lb))
        if obj_b is not None:                                      # B tiene que terminar de entrar y sonar solo un rato antes de empezar su propia salida
            b_fin = min(b_fin, max(b_min, obj_b - 4 - Lb - SOLO_MIN))
        if a_fin < a_ini or b_fin < b_min:
            continue
        ia = np.arange(a_ini, a_fin + 1)
        ib = np.arange(b_min, b_fin + 1)
        # --- armonía: cromas medios de cada zona, con los 7 desplazamientos posibles
        ca = (cum_a[ia + L] - cum_a[ia]);  ca /= np.linalg.norm(ca, axis=1, keepdims=True) + 1e-9
        cb = (cum_b[ib + Lb] - cum_b[ib]); cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9
        corr = np.stack([ca @ np.roll(cb, s, axis=1).T for s in range(-3, 4)])          # (7, na, nb) ; roll>0 = sube B
        costo_s = 0.12 * np.abs(np.arange(-3, 4))[:, None, None]
        pen = corr - costo_s
        s_idx = np.argmax(pen, axis=0)
        mejor_corr = np.take_along_axis(corr, s_idx[None], axis=0)[0]
        arm = 5.0 * (np.take_along_axis(pen, s_idx[None], axis=0)[0] - 0.6)                  # ≈ −3 … +2
        # --- energía
        eA = (cum_eA[ia + L] - cum_eA[ia]) / L                    # (na,)
        eB0 = (cum_eB[ib + min(4, Lb)] - cum_eB[ib]) / min(4, Lb)
        eB1 = (cum_eB[ib + Lb] - cum_eB[ib]) / Lb
        e_cont = -1.4 * np.abs(eA[:, None] - eB1[None, :])        # que el nivel no pegue un salto
        e_sale = 1.2 * (1.0 - np.clip(eA, 0, 1))[:, None] + np.zeros((1, len(ib)))      # A sale en un tramo sereno
        e_entra = 0.9 * (1.0 - np.clip(eB0, 0, 1))[None, :] + np.zeros((len(ia), 1))    # B entra desde su intro
        # --- voces
        vA = np.stack([A.voz[a:a + L] for a in ia])                # (na, L)
        pos_b = np.clip(np.round(np.arange(L) / k).astype(int), 0, Lb - 1)
        vB = np.stack([B.voz[b:b + Lb][pos_b] for b in ib])        # (nb, L)
        voz_sol = -3.0 * (vA @ vB.T) / L
        if voces:                                                  # voces por frases: A tiene un hueco de voz justo a mitad de la fusión (ahí se hace el relevo)
            pres_a = np.clip((vA - 0.35) / 0.30, 0.0, 1.0)
            hueco = 1.0 - pres_a[:, int(0.4 * L):int(0.7 * L) + 1].mean(axis=1)
            voz_sol = voz_sol + 1.2 * hueco[:, None]
        # --- posición y frases
        if obj is not None:                                        # lo más cerca posible del punto elegido
            pos_a = -0.25 * np.abs(ia - obj)[:, None] + np.zeros((1, len(ib)))
        elif a_desde is None:
            pos_a = 1.0 * (ia / nA)[:, None] + np.zeros((1, len(ib)))
        else:                                                      # lo más cerca posible del punto actual
            pos_a = -(0.4 if cerca else 0.08) * (ia - a_ini)[:, None] + np.zeros((1, len(ib)))
        pos_b0 = 0.5 * (b_min == ib)[None, :] + np.zeros((len(ia), 1))
        fr_a = 0.6 * (((ia - pa) % 8) == 0)[:, None] + np.zeros((1, len(ib)))
        fr_b = 0.5 * (((ib - pb) % 8) == 0)[None, :] + np.zeros((len(ia), 1))
        aud_b = (cum_eB[ib + Lb] - cum_eB[ib]) / Lb > 0.12
        ok = aud_b[None, :] & np.ones((len(ia), 1), bool)
        if estructura:
            def cerca(d, idx):
                v = d[np.clip(idx, 0, len(d) - 1)]
                return 1.0 * (v == 0) + 0.5 * (v == 1)
            b_a = 1.0 * cerca(estA.dist_borde, ia) + 0.5 * cerca(estA.dist_borde, ia + L)             # A sale justo cuando cambia de sección
            b_b = 0.6 * (1.0 * cerca(estB.dist_borde, ib) + 0.5 * cerca(estB.dist_borde, ib + Lb))            # B entra justo al empezar una sección
            fA = (cum_fA[np.clip(ia + L, 0, len(cum_fA) - 1)] - cum_fA[np.clip(ia, 0, len(cum_fA) - 1)]) / L
            fB = (cum_fB[np.clip(ib + Lb, 0, len(cum_fB) - 1)] - cum_fB[np.clip(ib, 0, len(cum_fB) - 1)]) / Lb
            est = b_a[:, None] + b_b[None, :] - 2.0 * np.outer(fA, fB)                           # estribillo sobre estribillo, no
        else:
            est = 0.0
        tot = arm + est + e_cont + e_sale + e_entra + voz_sol + pos_a + pos_b0 + fr_a + fr_b + (0.25 if L == 16 else 0.0)
        tot = tot - 6.0 * razon_t
        tot = np.where(ok, tot, -1e9)
        i, j = np.unravel_index(int(np.argmax(tot)), tot.shape)
        if mejor is None or tot[i, j] > mejor[0]:
            s_int = int(s_idx[i, j]) - 3
            mejor = (float(tot[i, j]), int(ia[i]), int(ib[j]), L, s_int, float(mejor_corr[i, j]),
                     dict(armonia=float(arm[i, j]), energia=float(e_cont[i, j] + e_sale[i, j] + e_entra[i, j]),
                          voz=float(voz_sol[i, j]), tempo=float(-6.0 * razon_t)))
    if mejor is None and a_desde is not None:
        return None
    if mejor is None:
        # temas muy cortos: fusión mínima sobre lo que haya
        L = max(4, min(8, nA // 3, nB // 3))
        a0 = max(0, nA - L - 2)
        return Plan(a0=a0, b0=0, L=L, k=1.0, shift=0.0, modo="limpia", puntaje=-1.0, armonia=0.0, razon="temas cortos")
    pt, a0, b0, L, s_int, corr, det = mejor
    shift = 0.0
    modo = "limpia"
    if corr >= 0.55:
        modo = "profunda" if corr >= 0.72 else "limpia"
        shift = s_int + (A.afinacion - B.afinacion) if s_int != 0 or abs(A.afinacion - B.afinacion) > 0.05 else 0.0
        shift = float(np.clip(shift, -3.4, 3.4))
    plan = Plan(a0=a0, b0=b0, L=L, k=k, shift=shift, modo=modo, puntaje=pt, armonia=corr, detalle=det)
    plan.razon = (f"{A.camelot}→{B.camelot} · {A.bpm:.0f}→{B.bpm:.0f} BPM · armonía {corr:.2f}")
    if voces:
        plan.razon += " · voces por frases"
    if estructura:
        plan.razon += " · por estructura"
    if forzar:
        for kk, v in forzar.items():
            setattr(plan, kk, v)
    return plan


def costo_par(A: Analisis, B: Analisis) -> float:
    """Qué tan mal combinan dos temas en general (menor = mejor): tempo, armonía y nivel de energía."""
    k = _factor_tempo(A.bpm, B.bpm)
    t = abs(np.log2(B.bpm * k / A.bpm))
    arm = max(float(np.dot(A.chroma_total, np.roll(B.chroma_total, s))) - 0.1 * abs(s) for s in range(-2, 3))
    return 8.0 * t + 4.0 * (1.0 - arm)


# ------------------------------------------------------------------------------------------- orden de la cola (como en AutoMix)
def _camelot_tuple(c: str | None):
    try:
        return int(c[:-1]), c[-1].upper()
    except (TypeError, ValueError, IndexError):
        return None


def compat_armonica(c1: str | None, c2: str | None) -> int | None:
    """2 = misma clave o relativa (suenan perfecto juntas) · 1 = vecina en la rueda Camelot (±1) · 0 = chocan · None = sin dato."""
    a, b = _camelot_tuple(c1), _camelot_tuple(c2)
    if not a or not b:
        return None
    if a[0] == b[0]:
        return 2
    if a[1] == b[1] and (abs(a[0] - b[0]) % 12 in (1, 11)):
        return 1
    return 0


CADENA_EXACTA_HASTA = 1500        # hasta esta cantidad de temas la cadena mira TODOS los que quedan; con más, solo los vecinos en BPM
CADENA_VENTANA = 96


def _encadenar(datos: list[dict], listos: list[int], ini: int) -> list[int]:
    """Cadena de mezcla: empieza en `ini` y en cada paso toma el tema que mejor engancha con el anterior
    (clave compatible primero, BPM parecido después). `listos` viene ordenada por BPM."""
    import bisect

    def puntaje_de(cur: int):
        def puntaje(j: int):
            nivel = compat_armonica(datos[cur]["cam"], datos[j]["cam"])
            dif = abs(datos[j]["bpm"] / datos[cur]["bpm"] - 1.0)
            return ((nivel or 0) * 10.0 - dif * 150.0, -datos[j]["bpm"])
        return puntaje

    listos = list(listos)
    cur = ini
    listos.remove(ini)
    cadena = [cur]
    if len(listos) + 1 <= CADENA_EXACTA_HASTA:
        while listos:
            cur = max(listos, key=puntaje_de(cur))
            listos.remove(cur)
            cadena.append(cur)
        return cadena
    bpms = [datos[j]["bpm"] for j in listos]
    v = CADENA_VENTANA
    while listos:
        p = bisect.bisect_left(bpms, datos[cur]["bpm"])
        lo, hi = max(0, p - v), min(len(listos), p + v)
        if lo >= hi:
            lo, hi = max(0, len(listos) - 1), len(listos)
        pf = puntaje_de(cur)
        j = max(range(lo, hi), key=lambda k: pf(listos[k]))
        if pf(listos[j])[0] < 0.0 and (hi - lo) < len(listos):         # nada compatible cerca: se mira más lejos
            lo2, hi2 = max(0, p - 8 * v), min(len(listos), p + 8 * v)
            j = max(range(lo2, hi2), key=lambda k: pf(listos[k]))
        cur = listos.pop(j)
        bpms.pop(j)
        cadena.append(cur)
    return cadena


ARCO = [(0.0, 0.35), (0.25, 0.55), (0.50, 0.75), (0.70, 0.95), (0.85, 0.60), (1.0, 0.80)]      # energía objetivo (0..1) a lo largo del set


def _orden_arco(datos: list[dict], listos: list[int]) -> list[int]:
    """Cadena de mezcla que sigue una curva de energía: arranca tranquilo, sube por tramos hasta un pico (~70 %), respira y cierra arriba.
    En cada paso gana el tema que mejor engancha con el anterior (clave y BPM) y cuya energía más se acerca a la que toca en ese punto."""
    n = len(listos)
    en = np.array([datos[j]["en"] for j in listos], np.float64)
    rango = np.argsort(np.argsort(en)) / max(1, n - 1)                  # energía relativa al resto de la cola (0 = el más sereno, 1 = el más fuerte)
    e_de = {j: float(r) for j, r in zip(listos, rango)}
    objetivo = lambda s: float(np.interp(s / max(1, n - 1), [p for p, _ in ARCO], [v for _, v in ARCO]))
    resto = list(listos)
    cur = min(resto, key=lambda j: abs(e_de[j] - objetivo(0)))
    resto.remove(cur)
    cadena = [cur]
    while resto:
        t = objetivo(len(cadena))

        def puntaje(j: int) -> float:
            nivel = compat_armonica(datos[cur]["cam"], datos[j]["cam"])
            dif = abs(datos[j]["bpm"] / datos[cur]["bpm"] - 1.0)
            return (nivel or 0) * 10.0 - dif * 150.0 - 45.0 * abs(e_de[j] - t)
        cur = max(resto, key=puntaje)
        resto.remove(cur)
        cadena.append(cur)
    return cadena


def orden_cola(datos: list[dict], armonico: bool = False, inicio: int | None = None, arco: bool = False) -> list[int]:
    """Orden de la cola como lo hace AutoMix. `datos[i] = {"bpm": float|None, "cam": str|None, "nombre": str}`.
    Sin armónico: como una tabla, por BPM de menor a mayor (y por tonalidad y nombre si empatan); los que no tienen análisis, al final.
    Con armónico: cadena de mezcla desde `inicio` (o desde el de menor BPM); los que no tienen BPM o clave van al final, por BPM."""
    n = len(datos)

    def clave_bpm(i: int):
        """Como una tabla: BPM (el entero que se ve), después tonalidad Camelot (1A, 1B, 2A…), después nombre."""
        d = datos[i]
        cam = d["cam"] or ""
        try:
            tono = (int(cam[:-1]), cam[-1])
        except ValueError:
            tono = (99, "")
        return (not d["bpm"], round(d["bpm"] or 0.0), tono, d["nombre"].lower())

    if arco:                                                  # arco del set: solo con los temas que tienen BPM, clave y energía
        validos = sorted((i for i in range(n) if datos[i]["bpm"] and datos[i]["cam"] and datos[i].get("en") is not None), key=clave_bpm)
        if len(validos) >= 4 and len(validos) <= CADENA_EXACTA_HASTA:
            sv = set(validos)
            return _orden_arco(datos, validos) + sorted((i for i in range(n) if i not in sv), key=clave_bpm)
    if not armonico:
        return sorted(range(n), key=clave_bpm)
    listos = sorted((i for i in range(n) if datos[i]["bpm"] and datos[i]["cam"]), key=clave_bpm)
    resto = sorted((i for i in range(n) if i not in set(listos)), key=clave_bpm)
    if not listos:
        return resto
    ini = inicio if inicio in listos else listos[0]
    return _encadenar(datos, listos, ini) + resto
