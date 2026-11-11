"""AutoMix — estiramiento de tempo que mantiene el tono (WSOLA)."""
import numpy as np


class WsolaFlujo:
    """Time-stretch en tiempo real por WSOLA (Waveform Similarity Overlap-Add): cambia la velocidad SIN cambiar el tono.
    Cada trama (N muestras, ventana de Hann, solape del 50 %) se toma del tema cerca de su posición nominal
    (que avanza HS*rate por trama) eligiendo, dentro de una ventana de búsqueda, el punto donde la onda más se parece
    a la continuación natural de la trama anterior: así las ondas empalman en fase (sin el sonido "de robot" del
    vocoder) y los golpes (bombo, caja) no se manchan.
    La ventana de búsqueda es asimétrica: se deja poco margen hacia el lado en que se acumularía el corrimiento de
    tiempo (según el tema vaya más rápido o más lento) y mucho hacia el otro, para que el audio quede siempre a
    pocos milisegundos de donde marca el cursor y el emparejado de los dos decks no se desfase."""
    N = 2048                 # largo de cada trama (46 ms a 44,1 kHz): más largo = voces más limpias
    HS = 1024                # avance de salida entre tramas (50 % de solape)
    DS = 256                 # margen corto de búsqueda (muestras): lado donde se acumularía el corrimiento
    DB = 512                 # margen largo de búsqueda (muestras): lado por el que se puede "volver"

    def __init__(self):
        self.W = (np.sin(np.pi * (np.arange(self.N) + 0.5) / self.N) ** 2).astype(np.float32)[:, None]
        self.reset(0.0)

    def reset(self, pos):
        self.p = float(pos)          # posición nominal (en el tema) de la próxima trama
        self.head = float(pos)       # posición (en el tema) de la próxima muestra a entregar
        self.tail = None             # mitad final de la trama anterior, ya con ventana (se suma a la próxima)
        self.prev_c = None           # dónde empezó de verdad la trama anterior
        self.cola = []               # tramas generadas pendientes de entregar: [array, desplazamiento, rate]
        self.fin = False

    @staticmethod
    def _tomar(a, ini, n):
        """a[ini:ini+n], con ceros fuera del audio."""
        L = len(a)
        if ini >= 0 and ini + n <= L:
            return a[ini:ini + n]
        out = np.zeros((n, 2), dtype=np.float32)
        d0, d1 = max(0, ini), min(L, ini + n)
        if d1 > d0:
            out[d0 - ini:d1 - ini] = a[d0:d1]
        return out

    def _generar(self, audio, rate):
        N, HS = self.N, self.HS
        if self.p >= len(audio):
            self.fin = True
            return False
        if self.prev_c is None:                        # arranque (o salto de posición): la primera mitad sale SIN ventana,
            c = int(round(self.p))                     # idéntica al tema, para empalmar sin clic con lo que sonaba antes
            fr = self._tomar(audio, c, N)
            out = fr[:HS].copy()
            self.tail = fr[HS:] * self.W[HS:]
        else:
            sg = float(np.clip((1.0 - rate) / 0.01, -1.0, 1.0))            # +1: más lento que el original, -1: más rápido
            da = int(round(self.DS + (self.DB - self.DS) * (1.0 + sg) / 2.0))     # cuánto se puede atrasar la trama
            nc = self.DS + self.DB + 1                                            # candidatos (siempre los mismos)
            lo = int(round(self.p)) - da
            ref = self._tomar(audio, self.prev_c + HS, HS)                  # continuación natural de la trama anterior
            seg = self._tomar(audio, lo, nc - 1 + HS)
            mr = ref[:, 0] + ref[:, 1]
            ms = seg[:, 0] + seg[:, 1]
            corr = np.correlate(ms, mr, "valid")                            # parecido de la onda con cada candidato
            e = np.concatenate(([0.0], np.cumsum(ms.astype(np.float64) ** 2)))
            en = e[HS:HS + nc] - e[:nc]                                     # energía de cada candidato (normaliza)
            c = lo + int(np.argmax(corr / np.sqrt(en + 1e-3)))
            fr = self._tomar(audio, c, N) * self.W
            out = self.tail + fr[:HS]
            self.tail = fr[HS:]
        self.prev_c = c
        self.p += HS * rate
        self.cola.append([out, 0, rate])
        return True

    def leer(self, audio, frames, rate):
        """Entrega `frames` muestras estiradas a `rate` (1,0 = original). Devuelve (audio, terminó)."""
        out = np.zeros((frames, 2), dtype=np.float32)
        lleno = 0
        while lleno < frames:
            if not self.cola:
                if self.fin or not self._generar(audio, rate):
                    return out, True
            blk = self.cola[0]
            arr, off, r = blk
            t = min(frames - lleno, len(arr) - off)
            out[lleno:lleno + t] = arr[off:off + t]
            lleno += t
            self.head += t * r
            blk[1] = off + t
            if blk[1] >= len(arr):
                self.cola.pop(0)
        return out, False
