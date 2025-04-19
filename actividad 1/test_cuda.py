import spacy
import torch
import os
import cupy
from thinc.api import get_current_ops, prefer_gpu

# Verificar versiones
print(f"spaCy version: {spacy.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Verificar CuPy
print(f"\nCuPy version: {cupy.__version__}")
try:
    print(f"CuPy CUDA available: {cupy.cuda.is_available()}")
    if cupy.cuda.is_available():
        print(f"CUDA version (CuPy): {cupy.cuda.runtime.runtimeGetVersion()}")
        print(f"GPU device count: {cupy.cuda.runtime.getDeviceCount()}")
        for i in range(cupy.cuda.runtime.getDeviceCount()):
            print(f"  Device {i}: {cupy.cuda.runtime.getDeviceProperties(i)['name'].decode('utf-8')}")
except Exception as e:
    print(f"Error al verificar CuPy CUDA: {e}")

# Configurar spaCy para usar GPU 
print("\nConfigurando spaCy para usar GPU...")
if cupy.cuda.is_available():
    try:
        # Configurar GPU para spaCy usando thinc
        spacy.prefer_gpu()
        prefer_gpu()
        print(f"GPU activada: {spacy.prefer_gpu()}")
        print(f"Operaciones en: {get_current_ops()}")
    except Exception as e:
        print(f"Error al configurar spaCy para GPU: {e}")
else:
    print("CUDA no está disponible. Usando CPU.")

# Cargar un modelo pequeño para probar
print("\nCargando modelo...")
nlp = spacy.load("en_core_web_sm")

# Procesar texto
print("\nProcesando texto de prueba...")
text = "This is a test to check if spaCy is working properly with CUDA."
doc = nlp(text)

print("\nTokens:")
for token in doc:
    print(f"  {token.text} - {token.pos_}")

print("\n¡Prueba completada!") 