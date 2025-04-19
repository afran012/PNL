import spacy
import cupy
import time
from thinc.api import prefer_gpu, require_gpu, get_current_ops

def setup_cuda():
    """Configura spaCy para usar CUDA si está disponible"""
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
    """Mide el tiempo de ejecución de una función"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

def process_with_nlp(nlp, texts, batch_size=1000):
    """Procesa textos con spaCy, opcionalmente en lotes para GPU"""
    start_time = time.time()
    
    # Procesar en lotes para GPU es más eficiente
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return docs, processing_time 