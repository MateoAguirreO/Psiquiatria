# %%
# 1 - Instalar
!pip install demucs

# 2 - Subir archivo
from google.colab import files
uploaded = files.upload()

# 3 - Separar stems
import os
filename = list(uploaded.keys())[0]
!python -m demucs --two-stems=vocals "{filename}"

# 4 - Convertir a mono
nombre_base = os.path.splitext(filename)[0]
!ffmpeg -y -i "separated/htdemucs/{nombre_base}/vocals.wav" -ac 1 vocals.wav
!ffmpeg -y -i "separated/htdemucs/{nombre_base}/no_vocals.wav" -ac 1 no_vocals.wav

print(f"vocals.wav: {os.path.getsize('vocals.wav') / 1e6:.2f} MB")
print(f"no_vocals.wav: {os.path.getsize('no_vocals.wav') / 1e6:.2f} MB")

# 5 - Descargar
files.download("vocals.wav")
files.download("no_vocals.wav")

# %%
from pathlib import Path

base = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/wav")
carpetas = ["0", "1"]

total = 0

for c in carpetas:
    ruta = base / c
    if not ruta.exists():
        print(f"La carpeta {ruta} no existe")
        continue

    n = sum(1 for f in ruta.rglob("*.wav") if f.is_file())
    total += n
    print(f"Carpeta {c}: {n} archivos denoised")

print(f"Total entre 0 y 1: {total}")

# %%
from pathlib import Path
import re

base = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav")
carpetas = ["0", "1"]

ids = set()

for c in carpetas:
    ruta = base / c
    if not ruta.exists():
        print(f"No existe: {ruta}")
        continue

    for f in ruta.rglob("*_denoised.wav"):
        m = re.match(r"^(\d{3})", f.name)   # toma solo 001, 002, ...
        if m:
            ids.add(m.group(1))

print("IDs encontrados:", sorted(ids))
print("Total de IDs únicos:", len(ids))

# %%
from pathlib import Path

base = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/MUESTRAS_SAMANÁ/Muestras")

wav_por_carpeta = {}

# Recorre cada carpeta tipo 001_..., 002_..., etc.
for carpeta in sorted([d for d in base.iterdir() if d.is_dir()]):
    wavs = [
        f for f in carpeta.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".wav"
        and not f.name.startswith("._")
        and not f.name.startswith("-")
        and not f.name.startswith(".")
    ]
    wav_por_carpeta[carpeta.name] = wavs

# Mostrar resultados
total = 0
for nombre_carpeta, lista_wavs in wav_por_carpeta.items():
    print(f"{nombre_carpeta}: {len(lista_wavs)} wav")
    for f in sorted(lista_wavs):
        print(f"  {f.name}")
    total += len(lista_wavs)

print(f"\nTotal WAV en todas las carpetas: {total}")


