"""
NEXUS — Instalador (lanzador).

Prepara todo lo que NEXUS necesita en UNA carpeta, sin que tengas que instalar Python:
  1. Python 3.14 "embebido" (portable, de python.org) + pip.
  2. El código de NEXUS (desde GitHub).
  3. Las librerías (numpy, scipy, soundfile, sounddevice, librosa, PySide6) dentro de libs\\py314,
     que es la carpeta donde NEXUS las busca.
Después crea los accesos directos y abre NEXUS. Si ya está todo instalado, abre NEXUS directo (sin ventana).

Uso:
  NexusInstalador.py                 instala en %LOCALAPPDATA%\\NEXUS
  NexusInstalador.py --portable      instala en una carpeta "NEXUS" junto a este programa (también si existe portable.txt)
  NexusInstalador.py --carpeta RUTA  instala en esa carpeta
  NexusInstalador.py --actualizar    vuelve a bajar el código de NEXUS desde GitHub
  NexusInstalador.py --ventana       muestra la ventana aunque ya esté todo instalado
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Optional

# ----------------------------------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------------------------------
REPO_ZIP = "https://github.com/gustavomayoraz/NEXUS_DJ/archive/refs/heads/main.zip"
PY_VERSIONES = ["3.14.10", "3.14.9", "3.14.8"]          # se prueba en orden: si una ya no está en python.org, se usa la siguiente
PY_EMBED_URL = "https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
VERSION_PY = "3.14"                                      # NEXUS busca sus librerías en libs\py314
CARPETA_CODIGO = "codigo_fuente"                         # carpeta del repositorio donde está el código de NEXUS (si no existe, se usa la raíz)
EXCLUIR_AL_ACTUALIZAR = {"datos", "libs", "config", "python", "_descargas", ".git"}

# (nombre, import_name): lista de ESTE programa (las mismas librerías que usa NEXUS en nexus/dependencias.py)
DEPENDENCIAS = [("numpy", "numpy"), ("scipy", "scipy"), ("soundfile", "soundfile"),
                ("sounddevice", "sounddevice"), ("librosa", "librosa"), ("PySide6", "PySide6")]
PIP_NOMBRE = {"PySide6": "PySide6-Essentials"}            # nombre del paquete en pip cuando no coincide (Essentials: solo Widgets/Core/Gui)

# Requisitos extra que no son librerías pip (fila propia en la ventana)
FILA_PYTHON = "Python 3.14"
FILA_CODIGO = "Código NEXUS"

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008


# ----------------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------------
def entorno_limpio() -> dict:
    """Entorno sin variables PYTHON*: así ninguna otra versión de Python instalada (3.10, 3.15...) se mete en la app."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}


def calcular_raiz(argv: list) -> Path:
    """Carpeta donde se instala NEXUS."""
    if "--carpeta" in argv and argv.index("--carpeta") + 1 < len(argv):
        return Path(argv[argv.index("--carpeta") + 1]).expanduser().resolve()
    aqui = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    if "--portable" in argv or (aqui / "portable.txt").exists():
        return aqui / "NEXUS"
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".local" / "share") / "NEXUS"


# ----------------------------------------------------------------------------------------------------
# Lógica de instalación (sin ventana)
# ----------------------------------------------------------------------------------------------------
class Instalacion:
    """Todo lo que se verifica e instala. No toca la ventana: devuelve resultados y avisa el avance con un callback."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = raiz
        self.py_dir = raiz / "python"
        self.py_exe = self.py_dir / "python.exe"
        self.pyw_exe = self.py_dir / "pythonw.exe"
        self.offline = raiz / "libs" / "py314"                    # NEXUS busca sus librerías en libs\py314
        self.cache = raiz / "datos"
        self.proc_actual: Optional[subprocess.Popen] = None
        self.cancelado = False

    # ---------- verificaciones (rápidas, sin ejecutar Python) ----------
    def python_ok(self) -> bool:
        return (self.py_exe.is_file() and self.pyw_exe.is_file() and any(self.py_dir.glob("python314*.dll"))
                and (self.py_dir / "Lib" / "site-packages" / "pip").is_dir())

    def codigo_ok(self) -> bool:
        return (self.raiz / "NEXUS.py").is_file() and (self.raiz / "nexus" / "ui_principal.py").is_file()

    def _marca(self, pip_name: str) -> Path:
        return self.offline / (".instalando_" + pip_name)       # misma marca que usa NEXUS: existe SOLO mientras se instala

    def dep_ok(self, nombre: str, import_name: str) -> bool:
        pip_name = PIP_NOMBRE.get(nombre, nombre)
        try:
            if self._marca(pip_name).exists():
                return False                                   # instalación interrumpida: incompleta
            return ((self.offline / import_name).is_dir() or any(self.offline.glob(import_name + ".*"))
                    or (self.offline / (import_name + ".py")).exists())
        except OSError:
            return False

    def todo_ok(self) -> bool:
        return self.python_ok() and self.codigo_ok() and all(self.dep_ok(n, i) for n, i in DEPENDENCIAS)

    # ---------- descargas ----------
    def _opener(self) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener()

    def descargar(self, url: str, destino: Path, avance: Optional[Callable[[Optional[int]], None]] = None) -> None:
        """Baja `url` a `destino` (archivo .part que se renombra al terminar). avance(porcentaje o None)."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_name(destino.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "NEXUS-Instalador"})
        try:
            with self._opener().open(req, timeout=30) as r, open(parcial, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                hecho = 0
                ultimo = -1
                while True:
                    if self.cancelado:
                        raise RuntimeError("cancelado")
                    bloque = r.read(65536)
                    if not bloque:
                        break
                    f.write(bloque)
                    hecho += len(bloque)
                    if avance:
                        pct = int(100 * hecho / total) if total else None
                        if pct != ultimo:                                  # solo cuando cambia el porcentaje (no por cada bloque)
                            ultimo = pct if pct is not None else -1
                            avance(pct)
            os.replace(parcial, destino)
        finally:
            parcial.unlink(missing_ok=True)

    def _correr(self, cmd: list, **kw) -> subprocess.CompletedProcess:
        """Corre un comando (se puede cancelar) y devuelve el resultado con la salida de texto."""
        flags = CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc_actual = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            text=True, errors="replace", creationflags=flags, env=entorno_limpio(), **kw)
        try:
            salida, _ = self.proc_actual.communicate()
            return subprocess.CompletedProcess(cmd, self.proc_actual.returncode, salida or "")
        finally:
            self.proc_actual = None

    # ---------- Python embebido ----------
    @staticmethod
    def parchar_pth(py_dir: Path) -> None:
        """En el Python embebido 'import site' viene comentado: sin eso no funciona pip. Se lo descomenta."""
        for pth in py_dir.glob("python*._pth"):
            texto = pth.read_text(encoding="utf-8")
            if "#import site" in texto:
                pth.write_text(texto.replace("#import site", "import site"), encoding="utf-8")

    def instalar_python(self, avance: Callable[[str], None]) -> None:
        desc = self.raiz / "_descargas"
        zip_py = None
        ultimo = None
        for v in PY_VERSIONES:
            destino = desc / f"python-{v}-embed-amd64.zip"
            try:
                avance(f"⬇️ Descargando Python {v}...")
                self.descargar(PY_EMBED_URL.format(v=v), destino, lambda p: avance(f"⬇️ Descargando Python {v}... {p}%" if p is not None else f"⬇️ Descargando Python {v}..."))
                zip_py = destino
                break
            except urllib.error.HTTPError as e:
                ultimo = e
                if e.code != 404:
                    raise
        if zip_py is None:
            raise RuntimeError(f"No se encontró el Python embebido en python.org ({ultimo})")
        avance("📦 Descomprimiendo Python...")
        if self.py_dir.exists():
            shutil.rmtree(self.py_dir, ignore_errors=True)
        with zipfile.ZipFile(zip_py) as z:
            z.extractall(self.py_dir)
        self.parchar_pth(self.py_dir)
        get_pip = desc / "get-pip.py"
        avance("⬇️ Descargando pip...")
        self.descargar(GET_PIP_URL, get_pip)
        avance("📦 Instalando pip...")
        cmd = [str(self.py_exe), str(get_pip), "--no-warn-script-location", "--disable-pip-version-check"]
        r = self._correr(cmd, cwd=str(self.py_dir))
        if r.returncode != 0 or not self.python_ok():
            raise RuntimeError("No se pudo instalar pip en el Python embebido:\n" + r.stdout[-1500:])
        shutil.rmtree(desc, ignore_errors=True)

    # ---------- código de NEXUS ----------
    def instalar_codigo(self, avance: Callable[[str], None]) -> None:
        desc = self.raiz / "_descargas"
        zip_cod = desc / "nexus-main.zip"
        avance("⬇️ Descargando NEXUS desde GitHub...")
        self.descargar(REPO_ZIP, zip_cod, lambda p: avance(f"⬇️ Descargando NEXUS... {p}%" if p is not None else "⬇️ Descargando NEXUS..."))
        raiz_abs = self.raiz.resolve()
        with zipfile.ZipFile(zip_cod) as z:
            # el código puede estar en la raíz del repositorio o dentro de la carpeta "codigo_fuente"
            quitar = 2 if any(Path(i.filename).parts[1:2] == (CARPETA_CODIGO,) for i in z.infolist()) else 1
            for info in z.infolist():
                todas = Path(info.filename).parts
                if quitar == 2 and todas[1:2] != (CARPETA_CODIGO,):
                    continue                                           # con carpeta codigo_fuente: solo se baja lo que está adentro
                partes = todas[quitar:]                                # se saca la carpeta de arriba (NEXUS_DJ-main) y, si existe, codigo_fuente
                if not partes or info.is_dir() or partes[0] in EXCLUIR_AL_ACTUALIZAR:
                    continue
                destino = (raiz_abs.joinpath(*partes)).resolve()
                if raiz_abs not in destino.parents:                     # nada de rutas que se salgan de la carpeta
                    continue
                destino.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as origen, open(destino, "wb") as f:
                    shutil.copyfileobj(origen, f)
        shutil.rmtree(desc, ignore_errors=True)
        if not self.codigo_ok():
            raise RuntimeError("El código bajado de GitHub no tiene NEXUS.py / nexus\\ui_principal.py")

    # ---------- librerías ----------
    def instalar_dependencia(self, nombre: str) -> tuple:
        """pip install --target libs\\py314 <paquete>, igual que lo hace NEXUS. Devuelve (ok, detalle)."""
        pip_name = PIP_NOMBRE.get(nombre, nombre)
        marca = self._marca(pip_name)
        ok, det = False, None
        try:
            self.offline.mkdir(parents=True, exist_ok=True)
            reparar = marca.exists()
            marca.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            cmd = [str(self.py_exe), "-m", "pip", "install", "--target", str(self.offline), "--disable-pip-version-check"]
            if reparar:
                cmd.append("--upgrade")
            r = self._correr(cmd + [pip_name])
            ok = (r.returncode == 0)
            det = None if ok else (r.stdout or "pip no devolvió detalle.").strip()[-1500:]
            if ok:
                marca.unlink(missing_ok=True)
        except (OSError, RuntimeError) as e:
            ok, det = False, str(e)
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            with open(self.cache / "dependencias.log", "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + pip_name + "  " + ("OK" if ok else "ERROR\n" + str(det)) + "\n")
        except OSError:
            pass
        return ok, det

    def reparar_permisos(self) -> None:
        """pip (Python 3.13+) deja los archivos con un permiso restringido que impide leerlos desde un inicio normal.
        Se les devuelve el permiso heredado de la carpeta (icacls /reset), una sola vez por instalación."""
        if os.name != "nt":
            return
        marca = self.raiz / "config" / "permisos_ok"
        if marca.exists():
            return
        try:
            r = subprocess.run(["icacls", str(self.raiz), "/reset", "/T", "/C", "/Q"], capture_output=True, text=True,
                               creationflags=CREATE_NO_WINDOW, timeout=600, env=entorno_limpio())
            if r.returncode == 0:
                marca.parent.mkdir(parents=True, exist_ok=True)
                marca.write_text("ok", encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ---------- accesos directos y arranque ----------
    def registrar_desinstalador(self) -> None:
        """Agrega NEXUS a 'Aplicaciones y características' (Panel de control > Desinstalar un programa), solo para el usuario actual."""
        if os.name != "nt":
            return
        marca = self.raiz / "config" / "desinstalador_ok"
        if marca.exists():
            return
        r = str(self.raiz)
        script = (
            "@echo off\r\n"
            'if not "%~1"=="x" (copy /y "%~f0" "%TEMP%\\nexus_desinstalar.cmd" >nul & start "" "%TEMP%\\nexus_desinstalar.cmd" x & exit /b)\r\n'
            "title Desinstalar NEXUS\r\n"
            f'echo Se va a borrar NEXUS: {r}\r\n'
            "echo (tus analisis y las mezclas guardadas de esa carpeta tambien se borran)\r\n"
            "choice /m \"Desinstalar NEXUS\"\r\n"
            "if errorlevel 2 exit /b 0\r\n"
            f"powershell -NoProfile -Command \"Get-Process | Where-Object {{ $_.Path -like '{r}\\*' }} | Stop-Process -Force\" >nul 2>nul\r\n"
            "ping -n 3 127.0.0.1 >nul\r\n"
            f'rmdir /s /q "{r}"\r\n'
            'del "%USERPROFILE%\\Desktop\\NEXUS.lnk" 2>nul\r\n'
            'del "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\NEXUS.lnk" 2>nul\r\n'
            'reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\NEXUS" /f >nul 2>nul\r\n'
            "echo Listo, NEXUS fue desinstalado.\r\n"
            "pause\r\n"
        )
        try:
            ruta = self.raiz / "config" / "desinstalar.cmd"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(script.encode("cp1252", errors="replace"))
            clave = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\NEXUS"
            valores = [("DisplayName", "REG_SZ", "NEXUS"), ("Publisher", "REG_SZ", "Gustavo Mayoraz"),
                       ("InstallLocation", "REG_SZ", r), ("DisplayIcon", "REG_SZ", str(self.raiz / "docs" / "nexus.ico") if (self.raiz / "docs" / "nexus.ico").exists() else str(self.pyw_exe)),
                       ("UninstallString", "REG_SZ", f'"{ruta}"'),
                       ("NoModify", "REG_DWORD", "1"), ("NoRepair", "REG_DWORD", "1")]
            for nombre, tipo, dato in valores:
                res = subprocess.run(["reg", "add", clave, "/v", nombre, "/t", tipo, "/d", dato, "/f"], capture_output=True,
                                     creationflags=CREATE_NO_WINDOW, timeout=30)
                if res.returncode != 0:
                    return
            marca.write_text("ok", encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            pass

    def crear_accesos(self) -> None:
        """Acceso directo en el Escritorio y en el menú Inicio (apuntan directo al Python de la carpeta: abre sin demora)."""
        if os.name != "nt":
            return
        marca = self.raiz / "config" / "accesos_creados"
        if marca.exists():
            return
        def ps(s: str) -> str:
            return "'" + str(s).replace("'", "''") + "'"
        ico = self.raiz / "docs" / "nexus.ico"                 # icono de NEXUS (viene con el código); si falta, queda el de Python
        icono_ps = f"$l.IconLocation = {ps(str(ico) + ',0')}; " if ico.exists() else ""
        guion = (
            "$w = New-Object -ComObject WScript.Shell; "
            "foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { "
            "$l = $w.CreateShortcut((Join-Path $d 'NEXUS.lnk')); "
            f"$l.TargetPath = {ps(self.pyw_exe)}; $l.Arguments = {ps(chr(34) + str(self.raiz / 'NEXUS.py') + chr(34))}; "
            f"$l.WorkingDirectory = {ps(self.raiz)}; {icono_ps}$l.Description = 'NEXUS - fusión inteligente de temas'; $l.Save() }}")
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", guion], capture_output=True,
                               text=True, creationflags=CREATE_NO_WINDOW, timeout=60)
            if r.returncode == 0:
                marca.parent.mkdir(parents=True, exist_ok=True)
                marca.write_text("ok", encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            pass

    def lanzar_app(self) -> None:
        flags = (CREATE_NO_WINDOW | DETACHED_PROCESS) if os.name == "nt" else 0
        subprocess.Popen([str(self.pyw_exe if self.pyw_exe.exists() else self.py_exe), str(self.raiz / "NEXUS.py")],
                         cwd=str(self.raiz), creationflags=flags, close_fds=True, env=entorno_limpio())


# ----------------------------------------------------------------------------------------------------
# Ventana (plantilla VentanaVerificacion)
# ----------------------------------------------------------------------------------------------------
class VentanaVerificacion:
    ANCHO = 400

    def __init__(self, inst: Instalacion, actualizar_codigo: bool = False) -> None:
        self.inst = inst
        self.actualizar_codigo = actualizar_codigo
        self.iniciar = False
        self.errores: dict = {}
        self._pending_after_ids: list = []

        self.ventana = tk.Tk()
        self.ventana.title("🔍 Verificación de dependencias")
        self.ventana.resizable(False, False)
        self.ventana.configure(bg="#f8f9fa")
        self.ventana.attributes("-topmost", True)
        self.ventana.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        cont = tk.Frame(self.ventana, bg="#f8f9fa", padx=10, pady=10)
        cont.pack(fill="both", expand=True)

        # 1. Título
        tk.Label(cont, text="📦 Verificando dependencias", font=("Segoe UI", 14, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(anchor="w")
        tk.Label(cont, text="Asegurando que todo esté listo para comenzar", font=("Segoe UI", 9), bg="#f8f9fa", fg="#7f8c8d").pack(anchor="w", pady=(0, 8))

        # 2. Lista: filas extra (Python y código, que no son librerías pip) + las dependencias
        self.labels_estado: dict = {}
        self.filas = [(FILA_PYTHON, "python"), (FILA_CODIGO, "codigo")] + [(n, "pip") for n, _i in DEPENDENCIAS]
        for nombre, _tipo in self.filas:
            fila = tk.Frame(cont, bg="#f8f9fa")
            fila.pack(fill="x", pady=1)
            tk.Label(fila, text=nombre, font=("Segoe UI", 9, "bold"), bg="#f8f9fa", fg="#2c3e50", width=14, anchor="w").pack(side="left")
            lbl = tk.Label(fila, text="⏳ Pendiente", font=("Segoe UI", 9), bg="#f8f9fa", fg="#7f8c8d", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            lbl.bind("<Button-1>", lambda _e, n=nombre: self._detalle(n))
            self.labels_estado[nombre] = lbl

        # 3. Barra
        self.barra = ttk.Progressbar(cont, mode="determinate", length=380)
        self.barra.pack(pady=(8, 4))

        # 4. Estado general
        self.lbl_general = tk.Label(cont, text="🔍 Verificando...", font=("Segoe UI", 9), bg="#f8f9fa", fg="#2c3e50",
                                    anchor="w", justify="left", wraplength=370)
        self.lbl_general.pack(fill="x", pady=(0, 6))

        # 5. Botones
        self.frame_botones = tk.Frame(cont, bg="#f8f9fa")
        self.frame_botones.pack(fill="x", pady=(0, 6))
        self.Iniciar_app = tk.Button(self.frame_botones, text="🚀 Iniciar aplicación", font=("Segoe UI", 9, "bold"), bg="#95a5a6", fg="white",
                                     relief="flat", padx=10, pady=5, state="disabled", command=self._iniciar_app)
        self.Iniciar_app.pack(side="left")
        self.btn_reintentar = tk.Button(self.frame_botones, text="🔄 Reintentar", font=("Segoe UI", 9, "bold"), bg="#95a5a6", fg="white",
                                        relief="flat", padx=10, pady=5, state="disabled", command=self._reintentar)
        self.btn_reintentar.pack(side="left", padx=(8, 0))

        # Tamaño: ancho fijo, alto según el contenido; centrada en pantalla
        self.ventana.update_idletasks()
        alto = self.ventana.winfo_reqheight() + 16
        x = (self.ventana.winfo_screenwidth() - self.ANCHO) // 2
        y = (self.ventana.winfo_screenheight() - alto) // 2
        self.ventana.geometry(f"{self.ANCHO}x{alto}+{x}+{y}")

        self._programar(150, self._lanzar)             # el hilo arranca YA con el mainloop andando (desde otro hilo, after() falla si todavía no está)

    # ---------- utilidades de la plantilla ----------
    def _programar(self, ms: int, fn: Callable[[], None]):
        """after() que guarda el id: se cancelan al cerrar y no quedan callbacks colgados."""
        caja: list = []

        def correr() -> None:
            if caja and caja[0] in self._pending_after_ids:
                self._pending_after_ids.remove(caja[0])               # ya se ejecutó: deja de estar pendiente
            fn()
        try:
            id_ = self.ventana.after(ms, correr)
        except (tk.TclError, RuntimeError):
            return None
        caja.append(id_)
        self._pending_after_ids.append(id_)
        return id_

    def _cancelar_pendientes(self) -> None:
        for id_ in self._pending_after_ids:
            try:
                self.ventana.after_cancel(id_)
            except (tk.TclError, RuntimeError):
                pass
        self._pending_after_ids.clear()

    @staticmethod
    def _config_seguro(widget, **kwargs) -> None:
        """widget.config(...) solo si el widget sigue vivo (la ventana pudo cerrarse con un after() todavía en cola)."""
        try:
            if widget.winfo_exists():
                widget.config(**kwargs)
        except (tk.TclError, RuntimeError):
            pass

    def _ui(self, widget, **kwargs) -> None:
        """Actualización de la interfaz desde el hilo de verificación."""
        self._programar(0, lambda: self._config_seguro(widget, **kwargs))

    def _detalle(self, nombre: str) -> None:
        if self.errores.get(nombre):
            messagebox.showerror(f"Error en '{nombre}'", self.errores[nombre], parent=self.ventana)

    # ---------- verificación (hilo) ----------
    def _lanzar(self) -> None:
        self.inst.cancelado = False
        self.errores.clear()
        for nombre, _t in self.filas:
            self._config_seguro(self.labels_estado[nombre], text="⏳ Pendiente", fg="#7f8c8d")
        self._config_seguro(self.barra, value=0)
        self._config_seguro(self.Iniciar_app, state="disabled", bg="#95a5a6")
        self._config_seguro(self.btn_reintentar, state="disabled", bg="#95a5a6")
        threading.Thread(target=self._trabajo, daemon=True).start()

    def _paso(self, nombre: str, tipo: str, hay_python: bool) -> str:
        """Verifica e instala un requisito. Devuelve 'ya', 'nuevo' o 'error'."""
        lbl = self.labels_estado[nombre]
        general = self.lbl_general
        if tipo == "python":
            ok = self.inst.python_ok()
        elif tipo == "codigo":
            ok = self.inst.codigo_ok() and not self.actualizar_codigo
        else:
            ok = self.inst.dep_ok(nombre, dict(DEPENDENCIAS)[nombre])
        if ok:
            self._ui(lbl, text="✅ Instalado", fg="#27ae60")
            return "ya"
        if tipo == "pip" and not hay_python:
            self.errores[nombre] = "No se puede instalar sin el Python embebido."
            self._ui(lbl, text="❌ Sin Python -- click para ver el detalle", fg="#e74c3c")
            return "error"
        self._ui(lbl, text="⏳ Instalando...", fg="#e67e22")
        self._ui(general, text=f"📦 Instalando {nombre} (puede tardar unos minutos)...")
        try:
            if tipo == "python":
                self.inst.instalar_python(lambda t: self._ui(general, text=t))
            elif tipo == "codigo":
                self.inst.instalar_codigo(lambda t: self._ui(general, text=t))
            else:
                ok2, det = self.inst.instalar_dependencia(nombre)
                if not ok2 or not self.inst.dep_ok(nombre, dict(DEPENDENCIAS)[nombre]):
                    raise RuntimeError(det or "El paquete se instaló pero no aparece en libs.")
        except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as e:
            self.errores[nombre] = str(e)
            self._ui(lbl, text="❌ Error -- click acá para ver el detalle", fg="#e74c3c")
            return "error"
        self._ui(lbl, text="✅ Instalado", fg="#27ae60")
        return "nuevo"

    def _trabajo(self) -> None:
        total = len(self.filas)
        nuevos = ya = fallos = 0
        hay_python = True
        for idx, (nombre, tipo) in enumerate(self.filas):
            if self.inst.cancelado:
                return
            self._ui(self.lbl_general, text=f"🔍 Verificando {nombre}...")
            r = self._paso(nombre, tipo, hay_python)
            if r == "ya":
                ya += 1
            elif r == "nuevo":
                nuevos += 1
            else:
                fallos += 1
                if tipo == "python":
                    hay_python = False
            self._ui(self.barra, value=((idx + 1) / total) * 100)
        if fallos == 0:
            self._ui(self.lbl_general, text="🔧 Ajustando permisos de los archivos...")
            self.inst.reparar_permisos()
            self.inst.crear_accesos()
            self.inst.registrar_desinstalador()
            self._programar(0, lambda: self._terminar(True, f"✅ Todo listo: {nuevos} instalado(s), {ya} ya estaban."))
        else:
            self._programar(0, lambda: self._terminar(False, f"❌ Hubo {fallos} error(es): tocá el ❌ para ver el detalle y reintentá."))

    def _terminar(self, ok: bool, resumen: str) -> None:
        self._config_seguro(self.lbl_general, text=resumen)
        if ok:
            self._config_seguro(self.Iniciar_app, state="normal", bg="#27ae60")        # el mismo botón: gris -> verde
        else:
            self._config_seguro(self.btn_reintentar, state="normal", bg="#e67e22")

    def _reintentar(self) -> None:
        self._lanzar()

    # ---------- botones ----------
    def _iniciar_app(self) -> None:
        self._cancelar_pendientes()
        self.iniciar = True
        self.ventana.quit()
        self.ventana.destroy()

    def _al_cerrar(self) -> None:
        self._cancelar_pendientes()
        self.inst.cancelado = True
        p = self.inst.proc_actual
        if p is not None:
            try:
                p.kill()
            except OSError:
                pass
        sys.exit(0)                                    # corta todo el programa, no solo la ventana

# ----------------------------------------------------------------------------------------------------
# Programa
# ----------------------------------------------------------------------------------------------------
def main(argv: list) -> int:
    raiz = calcular_raiz(argv)
    inst = Instalacion(raiz)
    if inst.todo_ok() and "--ventana" not in argv and "--actualizar" not in argv:       # todo instalado: ni abrir la ventana
        inst.reparar_permisos()
        inst.crear_accesos()
        inst.registrar_desinstalador()
        inst.lanzar_app()
        return 0
    win = VentanaVerificacion(inst, actualizar_codigo="--actualizar" in argv)
    win.ventana.mainloop()
    if win.iniciar:
        inst.lanzar_app()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
