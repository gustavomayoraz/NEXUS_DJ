"""Dependencias de NEXUS guardadas en la carpeta del programa (libs/pyXY): después de la primera vez funciona sin internet.

No usa tkinter: sirve tanto para la ventana de verificación como para el modo consola.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
LIBS = RAIZ / "libs" / f"py{sys.version_info.major}{sys.version_info.minor}"     # una carpeta por versión de Python

# (nombre para pip, nombre para importar): las librerías REALES que usa NEXUS
DEPENDENCIAS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"),
    ("librosa", "librosa"),
    ("PySide6-Essentials", "PySide6"),
]


def activar_libs() -> None:
    """Pone la carpeta libs al frente del buscador de módulos (los procesos hijos la heredan)."""
    LIBS.mkdir(parents=True, exist_ok=True)
    ruta = str(LIBS)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
    importlib.invalidate_caches()


def donde(imp: str) -> str | None:
    """'libs' si el módulo está en la carpeta del programa, 'sistema' si está instalado en otro lado, None si falta."""
    try:
        spec = importlib.util.find_spec(imp)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    origen = spec.origin if spec.origin and spec.origin != "namespace" else None
    rutas = [origen] if origen else list(spec.submodule_search_locations or [])
    for r in rutas:
        try:
            if LIBS in Path(r).resolve().parents or Path(r).resolve() == LIBS:
                return "libs"
        except OSError:
            pass
    return "sistema"


def faltantes() -> list[tuple[str, str]]:
    """Dependencias que todavía no están en la carpeta libs."""
    return [(n, i) for n, i in DEPENDENCIAS if donde(i) != "libs"]


def instalar(nombre: str) -> tuple[bool, str]:
    """Descarga `nombre` (con sus dependencias) dentro de libs. Devuelve (ok, último mensaje de error)."""
    LIBS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", nombre, "--target", str(LIBS), "--only-binary=:all:",
           "--disable-pip-version-check", "--no-warn-script-location", "--quiet"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000 if sys.platform == "win32" else 0)
    except OSError as e:
        return False, str(e)
    if r.returncode != 0:
        lineas = [l for l in (r.stderr or r.stdout or "").strip().splitlines() if l.strip()]
        return False, (lineas[-1] if lineas else f"pip terminó con código {r.returncode}")
    importlib.invalidate_caches()
    return True, ""
