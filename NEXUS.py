"""
NEXUS — mezclador de fusión inteligente de dos decks.

Los temas no se "cruzan": se funden. Cada uno se separa en capas (batería, bajo, voz/melodía, ambiente), se alinean
al mismo pulso, el que entra se afina para armonizar con el que sale, y cada capa entra o sale por separado: el bajo
se releva sobre la línea de compás, la batería cambia en el compás, y las melodías se disuelven con filtros mientras
un procesador espectral baja, banda por banda, lo que choca. El resultado suena como un solo tema.

Uso:  python NEXUS.py      (arrastrá archivos o carpetas a la cola y apretá ▶)
"""
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
import os
# caché del compilador (numba) propio de NEXUS, dentro de libs: los procesos hijo lo heredan
os.environ.setdefault("NUMBA_CACHE_DIR", str(RAIZ / "libs" / "numba_cache"))
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _preparar_dependencias() -> bool:
    """Usa las librerías de la carpeta libs; si falta alguna, abre la ventana que las descarga (solo la primera vez)."""
    from nexus import dependencias as dep
    dep.activar_libs()
    if not dep.faltantes():
        return True                                         # todo está en libs: arranque directo, sin ventana
    try:
        import tkinter  # noqa: F401
    except ImportError:                                     # sin tkinter: se instala por consola
        for nombre, imp in dep.faltantes():
            print(f"Descargando {nombre} a la carpeta libs…")
            ok, msg = dep.instalar(nombre)
            if not ok and dep.donde(imp) is None:
                print(f"No se pudo instalar {nombre}: {msg}")
                return False
        return True
    from nexus.verificacion import verificar_dependencias
    return verificar_dependencias()


def main() -> int:
    if not _preparar_dependencias():
        return 0
    from nexus.ui_principal import abrir
    return abrir()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()                    # necesario en Windows para los procesos de análisis
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:                                   # noqa: BLE001
        detalle = traceback.format_exc()
        try:
            from nexus.config import carpeta_datos
            (carpeta_datos() / "error_nexus.log").write_text(detalle, encoding="utf-8")
        except OSError:
            pass
        print(detalle)
        input("Ocurrió un error (quedó guardado en datos/error_nexus.log). Enter para cerrar…")
