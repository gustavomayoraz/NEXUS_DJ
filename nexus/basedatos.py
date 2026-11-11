"""Base de datos de NEXUS (SQLite, un solo archivo: datos/nexus.db). Cada tema se analiza UNA sola vez y queda guardado.

El tema se reconoce por su contenido (tamaño + primeros y últimos bytes), no por el nombre ni la carpeta: si lo renombrás o
lo movés, NO se vuelve a analizar. SQLite viene con Python, no hace falta instalar nada.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zlib
from pathlib import Path

from .config import carpeta_datos

_LEIDA = 4 * 65536


def archivo_db() -> Path:
    return carpeta_datos() / "nexus.db"


def _conectar() -> sqlite3.Connection:
    con = sqlite3.connect(str(archivo_db()), timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""CREATE TABLE IF NOT EXISTS analisis (
        clave TEXT PRIMARY KEY, ruta TEXT, nombre TEXT, tamano INTEGER, version INTEGER,
        bpm REAL, tonalidad TEXT, duracion REAL, sonoridad_db REAL, fecha TEXT, datos BLOB NOT NULL)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_analisis_ruta ON analisis(ruta)")
    con.execute("""CREATE TABLE IF NOT EXISTS fusiones (
        clave_a TEXT, clave_b TEXT, firma TEXT, nombre_a TEXT, nombre_b TEXT, tamano INTEGER, fecha TEXT, datos BLOB NOT NULL,
        PRIMARY KEY (clave_a, clave_b, firma))""")
    return con


def clave_tema(ruta: str) -> str | None:
    """Huella del archivo: tamaño + 256 KB del principio + 256 KB del final. None si no se puede leer."""
    try:
        tam = os.path.getsize(ruta)
        h = hashlib.sha1(str(tam).encode())
        with open(ruta, "rb") as f:
            h.update(f.read(_LEIDA))
            if tam > 2 * _LEIDA:
                f.seek(tam - _LEIDA)
                h.update(f.read(_LEIDA))
        return h.hexdigest()
    except OSError:
        return None


def buscar(ruta: str, version: int) -> dict | None:
    """Análisis guardado de ese archivo (como dict) o None si todavía no se analizó."""
    clave = clave_tema(ruta)
    if clave is None:
        return None
    try:
        con = _conectar()
        try:
            fila = con.execute("SELECT datos, version, ruta FROM analisis WHERE clave=?", (clave,)).fetchone()
            if fila is None or fila[1] != version:
                return None
            datos = json.loads(zlib.decompress(fila[0]).decode("utf-8"))
            if fila[2] != ruta:                                   # el archivo se movió o se renombró: se actualiza la ruta
                con.execute("UPDATE analisis SET ruta=?, nombre=? WHERE clave=?", (ruta, Path(ruta).stem, clave))
                con.commit()
            datos["ruta"] = ruta
            return datos
        finally:
            con.close()
    except (sqlite3.Error, ValueError, zlib.error, OSError):
        return None


def guardar(ruta: str, datos: dict, version: int) -> None:
    clave = clave_tema(ruta)
    if clave is None:
        return
    blob = zlib.compress(json.dumps(datos).encode("utf-8"), 6)
    try:
        con = _conectar()
        try:
            con.execute("INSERT OR REPLACE INTO analisis VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (clave, ruta, Path(ruta).stem, os.path.getsize(ruta), version, float(datos.get("bpm", 0.0)),
                         str(datos.get("camelot", "")), float(datos.get("duracion", 0.0)), float(datos.get("ganancia_db", 0.0)),
                         time.strftime("%Y-%m-%d %H:%M:%S"), blob))
            con.commit()
        finally:
            con.close()
    except (sqlite3.Error, OSError):
        pass


def cantidad() -> int:
    try:
        con = _conectar()
        try:
            return int(con.execute("SELECT COUNT(*) FROM analisis").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def importar_json_viejos(version: int) -> int:
    """Pasa a la base los análisis que NEXUS guardaba antes como archivos .json (datos/analisis/). Devuelve cuántos importó."""
    carpeta = carpeta_datos() / "analisis"
    if not carpeta.is_dir():
        return 0
    n = 0
    for p in carpeta.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("_v") == version and d.get("ruta") and os.path.exists(d["ruta"]) and buscar(d["ruta"], version) is None:
                guardar(d["ruta"], d, version)
                n += 1
        except (OSError, ValueError, TypeError):
            continue
    return n


# ------------------------------------------------------------------------------------------- fusiones ya calculadas
LIMITE_FUSIONES_MB = 2000                   # si las fusiones guardadas pasan de esto se borran las más viejas


def firma_fusion(plan: dict, ajustes: dict, version: int) -> str:
    """Identifica una fusión: mismo plan + mismos ajustes + misma versión del algoritmo = mismo resultado."""
    ajustes = {k: v for k, v in (ajustes or {}).items() if not str(k).startswith("_")}       # los datos internos (p. ej. los votos 👍/👎) no cambian la firma
    txt = json.dumps({"p": plan, "a": ajustes, "v": version}, sort_keys=True, default=str)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()


def existe_fusion(ca: str, cb: str, firma: str) -> bool:
    try:
        con = _conectar()
        try:
            return con.execute("SELECT 1 FROM fusiones WHERE clave_a=? AND clave_b=? AND firma=?", (ca, cb, firma)).fetchone() is not None
        finally:
            con.close()
    except sqlite3.Error:
        return False


def leer_fusion(ca: str, cb: str, firma: str) -> bytes | None:
    try:
        con = _conectar()
        try:
            fila = con.execute("SELECT datos FROM fusiones WHERE clave_a=? AND clave_b=? AND firma=?", (ca, cb, firma)).fetchone()
            return bytes(fila[0]) if fila else None
        finally:
            con.close()
    except sqlite3.Error:
        return None


def guardar_fusion(ca: str, cb: str, firma: str, blob: bytes, nombre_a: str = "", nombre_b: str = "") -> bool:
    """Guarda la fusión. Devuelve False si no se pudo (disco lleno, base bloqueada…)."""
    try:
        con = _conectar()
        try:
            con.execute("INSERT OR REPLACE INTO fusiones VALUES (?,?,?,?,?,?,?,?)",
                        (ca, cb, firma, nombre_a, nombre_b, len(blob), time.strftime("%Y-%m-%d %H:%M:%S"), blob))
            con.commit()
            _limpiar_fusiones(con)
            return True
        finally:
            con.close()
    except sqlite3.Error:
        return False


def borrar_fusion(ca: str, cb: str, firma: str) -> None:
    try:
        con = _conectar()
        try:
            con.execute("DELETE FROM fusiones WHERE clave_a=? AND clave_b=? AND firma=?", (ca, cb, firma))
            con.commit()
        finally:
            con.close()
    except sqlite3.Error:
        pass


def _limpiar_fusiones(con: sqlite3.Connection) -> None:
    """Mantiene las fusiones guardadas por debajo del límite de tamaño (borra las más viejas)."""
    total = con.execute("SELECT COALESCE(SUM(tamano),0) FROM fusiones").fetchone()[0]
    limite = LIMITE_FUSIONES_MB * 1024 * 1024
    if total <= limite:
        return
    for ca, cb, firma, tam in con.execute("SELECT clave_a, clave_b, firma, tamano FROM fusiones ORDER BY fecha ASC").fetchall():
        con.execute("DELETE FROM fusiones WHERE clave_a=? AND clave_b=? AND firma=?", (ca, cb, firma))
        total -= tam
        if total <= limite * 0.9:
            break
    con.commit()


def cantidad_fusiones() -> int:
    try:
        con = _conectar()
        try:
            return int(con.execute("SELECT COUNT(*) FROM fusiones").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error:
        return 0
