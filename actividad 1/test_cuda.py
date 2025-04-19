import spacy
import cupy
from spacy.util import use_gpu, get_gpu_allocator

# Verificar versiones
print(f"spaCy version: {spacy.__version__}")
print(f"CuPy version: {cupy.__version__}")
print(f"CUDA available: {cupy.cuda.is_available()}")

# Intentar usar GPU
use_gpu()
print(f"GPU enabled: {spacy.prefer_gpu()}")
print(f"GPU allocator: {get_gpu_allocator()}")

# Cargar un modelo pequeño para probar
print("Cargando modelo...")
nlp = spacy.load("en_core_web_sm")

# Procesar texto
print("\nProcesando texto de prueba...")
text = "This is a test to check if spaCy is working properly with CUDA."
doc = nlp(text)

print("\nTokens:")
for token in doc:
    print(f"  {token.text} - {token.pos_}")

print("\n¡Prueba completada!") 