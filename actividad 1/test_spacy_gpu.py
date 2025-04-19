import spacy
import time
import cupy
from thinc.api import prefer_gpu, require_cpu, require_gpu, get_current_ops

# Cargar el modelo en inglés
print("Cargando modelo...")
nlp = spacy.load('en_core_web_sm')

# Texto largo para procesar varias veces
texto_largo = """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence.

The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it is confronted with.

Up to the 1980s, most natural language processing systems were based on complex sets of hand-written rules. Starting in the late 1980s, however, there was a revolution in natural language processing with the introduction of machine learning algorithms for language processing. This was due to both the steady increase in computational power (see Moore's law) and the gradual lessening of the dominance of Chomskyan theories of linguistics (e.g. transformational grammar), whose theoretical underpinnings discouraged the sort of corpus linguistics that underlies the machine-learning approach to language processing.
"""

# Asegurarse de que estamos usando CPU inicialmente
require_cpu()
print(f"spaCy prefiere GPU: {spacy.prefer_gpu()}")
print(f"CUDA disponible (CuPy): {cupy.cuda.is_available()}")
print(f"Operaciones actuales: {get_current_ops()}")

# Prueba de rendimiento en CPU
print("\n--- PRUEBA EN CPU ---")
start_time = time.time()
for i in range(20):  # Procesar el texto 20 veces
    doc = nlp(texto_largo)
    # Contar entidades nombradas y tokens
    ents_count = len(doc.ents)
    tokens_count = len(doc)
    
cpu_time = time.time() - start_time
print(f"Tiempo en CPU: {cpu_time:.2f} segundos")

# Cambiar a GPU si está disponible
if cupy.cuda.is_available():
    print("\n--- PRUEBA EN GPU ---")
    try:
        # Intentar usar GPU
        require_gpu()
        print(f"spaCy prefiere GPU: {spacy.prefer_gpu()}")
        print(f"Operaciones actuales: {get_current_ops()}")
        
        # Prueba de rendimiento en GPU
        start_time = time.time()
        for i in range(20):  # Procesar el texto 20 veces
            doc = nlp(texto_largo)
            # Contar entidades nombradas y tokens
            ents_count = len(doc.ents)
            tokens_count = len(doc)
            
        gpu_time = time.time() - start_time
        print(f"Tiempo en GPU: {gpu_time:.2f} segundos")
        
        # Comparar rendimiento
        if gpu_time < cpu_time:
            print(f"\n¡La GPU es {cpu_time/gpu_time:.2f}x más rápida que la CPU!")
        else:
            print(f"\nLa CPU es {gpu_time/cpu_time:.2f}x más rápida que la GPU en este caso.")
            print("Nota: Para modelos pequeños como en_core_web_sm, la CPU puede ser más rápida debido a la sobrecarga de transferencia de datos a la GPU.")
    except Exception as e:
        print(f"Error al usar GPU: {e}")
else:
    print("\nNo se detectó CUDA disponible. No se puede probar en GPU.") 