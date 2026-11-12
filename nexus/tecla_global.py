"""Tecla global (funciona aunque NEXUS esté minimizado o en segundo plano): registra F2 en Windows con RegisterHotKey."""
from __future__ import annotations

import sys
import threading

from PySide6.QtCore import QObject, Signal

VK_F2 = 0x71
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
ID_F2 = 0x4E58


class TeclaGlobal(QObject):
    """Emite `pulsada` (en el hilo de la interfaz) cada vez que se aprieta F2 en cualquier parte de Windows."""
    pulsada = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.activa = False                      # True si Windows aceptó registrar F2 (si otro programa ya la tiene, queda False)
        self._tid = 0
        self._listo = threading.Event()
        if sys.platform != "win32":
            return
        threading.Thread(target=self._bucle, daemon=True).start()
        self._listo.wait(2.0)

    def _bucle(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes
            u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
            self._tid = k32.GetCurrentThreadId()
            self.activa = bool(u32.RegisterHotKey(None, ID_F2, MOD_NOREPEAT, VK_F2))
            self._listo.set()
            if not self.activa:
                return
            msg = wintypes.MSG()
            while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam == ID_F2:
                    self.pulsada.emit()
            u32.UnregisterHotKey(None, ID_F2)
        except Exception:                                    # noqa: BLE001
            self.activa = False
        finally:
            self._listo.set()

    def cerrar(self) -> None:
        if self.activa and self._tid:
            try:
                import ctypes
                ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            except Exception:                                # noqa: BLE001
                pass
