"""Ventana de verificación e instalación de dependencias (plantilla VentanaVerificacion de Gustavo).

Solo aparece cuando falta algo: descarga las librerías dentro de la carpeta del programa (libs/) y de ahí en más se usan
sin internet. Si todo está en libs, NEXUS arranca directo sin mostrar esta ventana.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import ttk

from . import dependencias as dep
from .dependencias import DEPENDENCIAS


class VentanaVerificacion:
    def __init__(self) -> None:
        self.iniciar = False                              # True si se pasó a iniciar la aplicación
        self._pending_after_ids: list[str] = []
        self.labels_estado: dict[str, tk.Label] = {}
        self.ventana = tk.Tk()
        self.ventana.title("🔍 Verificación de dependencias")
        self.ventana.resizable(False, False)
        self.ventana.configure(bg="#f8f9fa")
        self.ventana.attributes("-topmost", True)
        self.ventana.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._construir()
        self._ajustar_tamano()
        self._lanzar()

    def _lanzar(self) -> None:
        self._config_seguro(self.Reintentar, state="disabled")
        self._config_seguro(self.Iniciar_app, state="disabled", bg="#95a5a6")
        threading.Thread(target=self._verificar, daemon=True).start()

    # ------------------------------------------------------------------ utilidades seguras con after()
    def _programar(self, ms: int, fn) -> str | None:
        try:
            aid = self.ventana.after(ms, fn)
        except tk.TclError:
            return None
        self._pending_after_ids.append(aid)
        return aid

    def _cancelar_pendientes(self) -> None:
        for aid in self._pending_after_ids:
            try:
                self.ventana.after_cancel(aid)
            except tk.TclError:
                pass
        self._pending_after_ids.clear()

    @staticmethod
    def _config_seguro(widget, **kwargs) -> None:
        try:
            if widget.winfo_exists():
                widget.config(**kwargs)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ construcción
    def _construir(self) -> None:
        v = self.ventana
        tk.Label(v, text="📦 Verificando dependencias", font=("Segoe UI", 14, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(pady=(14, 0))
        tk.Label(v, text="Asegurando que todo esté listo para comenzar", font=("Segoe UI", 9), bg="#f8f9fa", fg="#7f8c8d").pack(pady=(0, 8))

        marco = tk.Frame(v, bg="#f8f9fa")
        marco.pack(fill="x", padx=20)
        for nombre, _imp in DEPENDENCIAS:
            fila = tk.Frame(marco, bg="#f8f9fa")
            fila.pack(fill="x", pady=1)
            tk.Label(fila, text=nombre, font=("Segoe UI", 10, "bold"), width=18, anchor="w", bg="#f8f9fa", fg="#2c3e50").pack(side="left")
            lbl = tk.Label(fila, text="⏳ Pendiente", font=("Segoe UI", 10), bg="#f8f9fa", fg="#7f8c8d", anchor="w")
            lbl.pack(side="left")
            self.labels_estado[nombre] = lbl

        self.barra = ttk.Progressbar(v, mode="determinate", length=380)
        self.barra.pack(pady=(10, 4))
        self.lbl_general = tk.Label(v, text="🔍 Preparando…", font=("Segoe UI", 10), bg="#f8f9fa", fg="#2c3e50", wraplength=370, justify="center")
        self.lbl_general.pack(pady=(0, 6))

        self.frame_botones = tk.Frame(v, bg="#f8f9fa")
        self.frame_botones.pack(pady=(0, 6))
        self.Iniciar_app = tk.Button(self.frame_botones, text="🚀 Iniciar aplicación", font=("Segoe UI", 10, "bold"), fg="white", bg="#95a5a6",
                                     activebackground="#229954", relief="flat", padx=14, pady=6, state="disabled", command=self._iniciar_app)
        self.Iniciar_app.pack(side="left", padx=4)
        self.Reintentar = tk.Button(self.frame_botones, text="🔁 Reintentar", font=("Segoe UI", 10), relief="groove", padx=10, pady=5,
                                    state="disabled", command=self._lanzar)
        self.Reintentar.pack(side="left", padx=4)
        tk.Label(v, text="Se guardan en la carpeta «libs» del programa: la próxima vez funciona sin internet.", font=("Segoe UI", 8),
                 bg="#f8f9fa", fg="#7f8c8d", wraplength=370).pack(pady=(0, 12))

    def _ajustar_tamano(self) -> None:
        """Ancho fijo de 400; el alto se calcula con lo que ocupa el contenido (sin espacio de más ni cortes)."""
        v = self.ventana
        v.update_idletasks()
        alto = v.winfo_reqheight() + 16
        x = (v.winfo_screenwidth() - 400) // 2
        y = (v.winfo_screenheight() - alto) // 2
        v.geometry(f"400x{alto}+{x}+{y}")

    # ------------------------------------------------------------------ verificación (hilo)
    def _verificar(self) -> None:
        total = len(DEPENDENCIAS)
        errores: list[str] = []
        detalle = ""
        for idx, (nombre, imp) in enumerate(DEPENDENCIAS):
            lbl = self.labels_estado[nombre]
            self._programar(0, lambda n=nombre: self._config_seguro(self.lbl_general, text=f"🔍 Verificando {n}..."))
            if dep.donde(imp) == "libs":
                estado = "✅ OK"
            else:
                self._programar(0, lambda l=lbl: self._config_seguro(l, text="⏳ Instalando...", fg="#e67e22"))
                self._programar(0, lambda n=nombre: self._config_seguro(self.lbl_general, text=f"📦 Descargando {n} a la carpeta libs..."))
                ok, msg = dep.instalar(nombre)
                if ok and dep.donde(imp) == "libs":
                    estado = "✅ Instalado"
                elif dep.donde(imp) == "sistema":
                    estado = "✅ OK (del sistema)"                      # no se pudo bajar, pero ya está instalada en Python
                else:
                    estado = "❌ Error"
                    errores.append(nombre)
                    detalle = msg
            color = "#27ae60" if estado.startswith("✅") else "#c0392b"
            self._programar(0, lambda l=lbl, e=estado, c=color: self._config_seguro(l, text=e, fg=c))
            self._programar(0, lambda i=idx: self._config_seguro(self.barra, value=((i + 1) / total) * 100))
        if errores:
            resumen = "⚠️ No se pudo instalar: " + ", ".join(errores) + (f"\n{detalle[:160]}" if detalle else "")
            self._programar(0, lambda: self._config_seguro(self.lbl_general, text=resumen))
            self._programar(0, lambda: self._config_seguro(self.Reintentar, state="normal"))
            self._programar(0, lambda: self._config_seguro(self.Iniciar_app, state="normal", bg="#e67e22"))
        else:
            self._programar(0, lambda: self._config_seguro(self.lbl_general, text="✅ Todo listo para comenzar"))
            self._programar(0, lambda: self._config_seguro(self.Iniciar_app, state="normal", bg="#27ae60"))
            self._programar(1200, self._iniciar_app)                      # sin errores: arranca solo

    # ------------------------------------------------------------------ cierre
    def _iniciar_app(self) -> None:
        if self.iniciar:
            return
        self._cancelar_pendientes()
        self.iniciar = True
        self.ventana.quit()
        self.ventana.destroy()

    def _cerrar(self) -> None:
        self._cancelar_pendientes()
        sys.exit(0)                                       # la X corta toda la aplicación, no solo la ventana

    def mostrar(self) -> bool:
        self.ventana.mainloop()
        return self.iniciar


def verificar_dependencias() -> bool:
    """Abre la ventana (solo si falta algo), espera a que termine y devuelve True para continuar con la aplicación."""
    return VentanaVerificacion().mostrar()
