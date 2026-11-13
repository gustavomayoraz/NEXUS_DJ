"""Memoria RAM del equipo (sin librerías extra) para que el trabajo en segundo plano no ahogue la PC."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def memoria_mb() -> tuple[int, int]:
    """(libre, total) en MB. Si no se puede medir devuelve valores altos para no frenar nada."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys // 1048576), int(st.ullTotalPhys // 1048576)
        datos = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for linea in f:
                k, _, v = linea.partition(":")
                datos[k] = int(v.split()[0])
        return datos["MemAvailable"] // 1024, datos["MemTotal"] // 1024
    except (OSError, ValueError, KeyError, AttributeError):
        return 99999, 99999


EST_FUSION_MB = 400           # lo que usa calcular una mezcla (medido: ~250 MB + margen)


def estimar_analisis_mb(ruta: str) -> int:
    """Estimación de la memoria que usará analizar este tema, según el tamaño del archivo (medido: ~3,7 MB por segundo de audio)."""
    try:
        tam = os.path.getsize(ruta)
    except OSError:
        return 1200
    ext = os.path.splitext(ruta)[1].lower()
    seg = tam / 176400.0 if ext in (".wav", ".flac", ".aiff", ".aif") else tam / 16000.0      # sin comprimir / mp3 típico de 128 kbps
    return int(min(6000, 150 + 3.7 * max(60.0, seg)))


@dataclass(frozen=True)
class Perfil:
    """Cómo reparte NEXUS el trabajo según la RAM y los núcleos de esta PC."""
    nombre: str
    ram_mb: int
    nucleos: int
    hilos: int                 # hilos dentro de cada análisis (filtros de mediana)
    trab_an: int               # procesos que analizan temas a la vez
    trab_fu: int               # procesos que calculan mezclas a la vez
    tareas_por_hijo: int       # cada proceso hijo se renueva tras tantas tareas (devuelve la memoria al sistema; cada arranque cuesta 5–7 s de imports, por eso son varias)
    reserva_mb: int            # memoria libre que siempre se deja tranquila al sistema y a los demás programas
    audio_siguientes: int      # temas siguientes que se mantienen decodificados en memoria


def calcular_perfil() -> Perfil:
    _, total = memoria_mb()
    nuc = max(1, os.cpu_count() or 2)
    if total < 6000:
        return Perfil("mínimo", total, nuc, 1, 1, 1, 6, 1000, 1)
    if total < 12000:
        return Perfil("básico", total, nuc, max(1, min(2, nuc // 2)), 1, 1, 8, 1300, 1)
    if total < 20000:
        return Perfil("medio", total, nuc, max(1, min(3, nuc // 2)), 2 if nuc >= 6 else 1, 1, 12, 1800, 2)
    return Perfil("alto", total, nuc, max(1, min(4, nuc // 2)), max(1, min(3, nuc // 2)), 2 if nuc >= 6 else 1, 16, 2500, 2)


class Guardia:
    """Decide si hay memoria para lanzar una tarea más. Con histéresis (se pausa y no se reanuda hasta tener 30 % más de margen)."""

    def __init__(self, reserva_mb: int) -> None:
        self.reserva = reserva_mb
        self.pausado = False

    def hay_lugar(self, necesita_mb: int = 0) -> bool:
        libre, _ = memoria_mb()
        margen = libre - necesita_mb
        self.pausado = margen < (self.reserva * 1.3 if self.pausado else self.reserva)
        return not self.pausado

    def critico(self) -> bool:
        """Queda tan poca memoria libre que ni una sola tarea pesada es segura."""
        return memoria_mb()[0] < 600
