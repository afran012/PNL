import json
import os
import time

# Cargar el archivo del notebook
notebook_path = "caracteristicasOdioCuda.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Crear una copia de seguridad del notebook original
backup_path = "caracteristicasOdioCuda_backup.ipynb"
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Backup creado en: {backup_path}")

# Crear el archivo helper de CUDA si no existe
if not os.path.exists("cuda_helper.py"):
    with open("cuda_helper.py", 'w', encoding='utf-8') as f:
        f.write("""import spacy
import cupy
import time
from thinc.api import prefer_gpu, require_gpu, get_current_ops

def setup_cuda():
    \"\"\"Configura spaCy para usar CUDA si está disponible\"\"\"
    cuda_info = {
        "spacy_version": spacy.__version__,
        "cuda_available": False,
        "gpu_active": False,
        "device_name": None
    }
    
    try:
        # Verificar si CuPy está disponible
        cuda_info["cuda_available"] = cupy.cuda.is_available()
        
        if cuda_info["cuda_available"]:
            # Obtener información del dispositivo
            device_id = 0
            cuda_info["device_name"] = cupy.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
            
            # Activar GPU para spaCy
            require_gpu()
            cuda_info["gpu_active"] = spacy.prefer_gpu()
            
            print(f"CUDA está disponible en: {cuda_info['device_name']}")
            print(f"GPU activada para spaCy: {cuda_info['gpu_active']}")
            print(f"Operaciones en: {get_current_ops()}")
        else:
            print("CUDA no está disponible. Se usará CPU.")
    except Exception as e:
        print(f"Error al configurar CUDA: {e}")
        
    return cuda_info

def time_operation(func, *args, **kwargs):
    \"\"\"Mide el tiempo de ejecución de una función\"\"\"
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

def process_with_nlp(nlp, texts, batch_size=1000):
    \"\"\"Procesa textos con spaCy, opcionalmente en lotes para GPU\"\"\"
    start_time = time.time()
    
    # Procesar en lotes para GPU es más eficiente
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return docs, processing_time
""")
    print("Archivo cuda_helper.py creado")

# Modificaciones al notebook
# 1. Agregar importación de cuda_helper y time después de la celda de importaciones
import_cell_index = None
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code" and "import pathlib" in cell.get("source", ""):
        import_cell_index = i
        break

if import_cell_index is not None:
    # Añadir time e importar cuda_helper
    notebook["cells"][import_cell_index]["source"] += "\nimport time\nimport cuda_helper"
    
    # Añadir celda para configurar CUDA después de importaciones
    cuda_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Configurar CUDA para spaCy (aceleración GPU)\n",
            "print(\"\\n=== Configurando CUDA para spaCy ===\")\n",
            "start_time = time.time()\n",
            "cuda_info = cuda_helper.setup_cuda()\n",
            "setup_time = time.time() - start_time\n",
            "print(f\"Configuración completada en {setup_time:.2f} segundos\")"
        ],
        "outputs": []
    }
    notebook["cells"].insert(import_cell_index + 1, cuda_setup_cell)
    
    # Modificar celda de carga del modelo para medir tiempo
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "nlp = es_core_news_md.load()" in cell.get("source", ""):
            notebook["cells"][i]["source"] = "# Medir tiempo de carga del modelo\nstart_time = time.time()\nnlp = es_core_news_md.load()\nload_time = time.time() - start_time\nprint(f\"Modelo cargado en {load_time:.2f} segundos\")"
            break
    
    # Modificar la celda que procesa los documentos con el enfoque por lotes
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "doc = []" in cell.get("source", "") and "for i in range(0, lines_number):" in cell.get("source", ""):
            notebook["cells"][i]["source"] = """# Procesamiento en lotes optimizado para GPU
print("\\n=== Procesando textos con spaCy ===")
start_time = time.time()

# Preparar datos
texts = data["CONTENIDO A ANALIZAR"][:lines_number].tolist()
values = data["INTENSIDAD"][:lines_number].tolist()

# Procesar en lotes (más eficiente para GPU)
doc, processing_time = cuda_helper.process_with_nlp(nlp, texts, batch_size=1000)
value = values

print(f"Procesamiento completado en {processing_time:.2f} segundos")
print(f"Dispositivo utilizado: {'GPU' if cuda_info['gpu_active'] else 'CPU'}")
if cuda_info['gpu_active']:
    print(f"Modelo de GPU: {cuda_info['device_name']}")

# Ejemplo de cómo recorrer un comentario palabra por palabra    
for token in doc[1]:
    print(token)"""
            break

# Modificar el nombre del notebook para indicar que usa CUDA
output_path = "caracteristicasOdioCudaGPU.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook con CUDA creado en: {output_path}")
print("Ahora puedes ejecutar el notebook modificado para usar aceleración GPU") 