import pathlib
import spacy
import pandas as pd
from spacy import displacy
import csv
import es_core_news_md
import time
from collections import Counter
import matplotlib.pyplot as plt
import cuda_helper

# Configurar CUDA
print("\n=== Configurando CUDA para spaCy ===")
cuda_info = cuda_helper.setup_cuda()

# Cargar el modelo
print("\n=== Cargando modelo de lenguaje ===")
start_time = time.time()
nlp = es_core_news_md.load()
load_time = time.time() - start_time
print(f"Modelo cargado en {load_time:.2f} segundos")

# Cargar datos
print("\n=== Cargando datos ===")
filename = "./02Dataset_sin_procesar.csv"
# Puedes ajustar el número de líneas o quitar el límite para todo el conjunto
lines_number = 20  
data = pd.read_csv(filename, delimiter=';', encoding='latin-1', nrows=lines_number)
print(f"Datos cargados: {len(data)} registros")

# Procesar textos con CUDA (más eficiente en lotes)
print("\n=== Procesando textos con spaCy ===")
texts = data["CONTENIDO A ANALIZAR"].tolist()
docs, processing_time = cuda_helper.process_with_nlp(nlp, texts, batch_size=1000)
print(f"Procesamiento completado en {processing_time:.2f} segundos")

# Separar por grupos (odio y no odio)
print("\n=== Separando comentarios por grupos ===")
comentarios_odio = [docs[i] for i in range(len(docs)) if data["INTENSIDAD"].iloc[i] > 0]
comentarios_no_odio = [docs[i] for i in range(len(docs)) if data["INTENSIDAD"].iloc[i] == 0]
print(f"Comentarios de odio: {len(comentarios_odio)}")
print(f"Comentarios sin odio: {len(comentarios_no_odio)}")

# Análisis - Pregunta 2: Total de palabras
print("\n=== ANÁLISIS 1: Total de palabras ===")
total_palabras = sum(len([token for token in doc if not token.is_punct and not token.is_space]) 
                    for doc in docs)
print(f"El corpus contiene un total de {total_palabras} palabras.")

# Análisis - Pregunta 3: Promedio de palabras por comentario
print("\n=== ANÁLISIS 2: Promedio de palabras por comentario ===")
palabras_por_comentario = [len([token for token in doc if not token.is_punct and not token.is_space]) 
                          for doc in docs]
promedio_palabras = sum(palabras_por_comentario) / len(palabras_por_comentario)
print(f"El número promedio de palabras por comentario es: {promedio_palabras:.2f}")

# Análisis - Pregunta 4: Promedio de palabras por grupo
print("\n=== ANÁLISIS 3: Promedio de palabras por grupo ===")
palabras_odio = [len([token for token in doc if not token.is_punct and not token.is_space]) 
                for doc in comentarios_odio]
palabras_no_odio = [len([token for token in doc if not token.is_punct and not token.is_space]) 
                   for doc in comentarios_no_odio]
promedio_palabras_odio = sum(palabras_odio) / len(palabras_odio) if palabras_odio else 0
promedio_palabras_no_odio = sum(palabras_no_odio) / len(palabras_no_odio) if palabras_no_odio else 0
print(f"Promedio de palabras en comentarios de odio: {promedio_palabras_odio:.2f}")
print(f"Promedio de palabras en comentarios sin odio: {promedio_palabras_no_odio:.2f}")

# Análisis - Pregunta 5: Promedio de oraciones por grupo
print("\n=== ANÁLISIS 4: Promedio de oraciones por grupo ===")
oraciones_odio = [len(list(doc.sents)) for doc in comentarios_odio]
oraciones_no_odio = [len(list(doc.sents)) for doc in comentarios_no_odio]
promedio_oraciones_odio = sum(oraciones_odio) / len(oraciones_odio) if oraciones_odio else 0
promedio_oraciones_no_odio = sum(oraciones_no_odio) / len(oraciones_no_odio) if oraciones_no_odio else 0
print(f"Promedio de oraciones en comentarios de odio: {promedio_oraciones_odio:.2f}")
print(f"Promedio de oraciones en comentarios sin odio: {promedio_oraciones_no_odio:.2f}")

# Análisis - Pregunta 6: Porcentaje de comentarios con entidades NER
print("\n=== ANÁLISIS 5: Porcentaje de comentarios con entidades NER ===")
comentarios_con_ner_odio = sum(1 for doc in comentarios_odio if len(doc.ents) > 0)
comentarios_con_ner_no_odio = sum(1 for doc in comentarios_no_odio if len(doc.ents) > 0)
porcentaje_ner_odio = (comentarios_con_ner_odio / len(comentarios_odio)) * 100 if comentarios_odio else 0
porcentaje_ner_no_odio = (comentarios_con_ner_no_odio / len(comentarios_no_odio)) * 100 if comentarios_no_odio else 0
print(f"Porcentaje de comentarios de odio con entidades NER: {porcentaje_ner_odio:.2f}%")
print(f"Porcentaje de comentarios sin odio con entidades NER: {porcentaje_ner_no_odio:.2f}%")

# Análisis - Pregunta 7: Porcentaje de comentarios con entidades PER (PERSON)
print("\n=== ANÁLISIS 6: Porcentaje de comentarios con entidades de personas ===")
comentarios_con_person_odio = sum(1 for doc in comentarios_odio 
                                if any(ent.label_ == "PER" for ent in doc.ents))
comentarios_con_person_no_odio = sum(1 for doc in comentarios_no_odio 
                                   if any(ent.label_ == "PER" for ent in doc.ents))
porcentaje_person_odio = (comentarios_con_person_odio / len(comentarios_odio)) * 100 if comentarios_odio else 0
porcentaje_person_no_odio = (comentarios_con_person_no_odio / len(comentarios_no_odio)) * 100 if comentarios_no_odio else 0
print(f"Porcentaje de comentarios de odio con entidades PERSON: {porcentaje_person_odio:.2f}%")
print(f"Porcentaje de comentarios sin odio con entidades PERSON: {porcentaje_person_no_odio:.2f}%")

# Análisis - Pregunta 9: Tipos de entidades en cada grupo
print("\n=== ANÁLISIS 7: Tipos de entidades por grupo ===")
tipos_entidades_odio = {}
for doc in comentarios_odio:
    for ent in doc.ents:
        tipos_entidades_odio[ent.label_] = tipos_entidades_odio.get(ent.label_, 0) + 1

tipos_entidades_no_odio = {}
for doc in comentarios_no_odio:
    for ent in doc.ents:
        tipos_entidades_no_odio[ent.label_] = tipos_entidades_no_odio.get(ent.label_, 0) + 1

todos_tipos = set(tipos_entidades_odio.keys()) | set(tipos_entidades_no_odio.keys())

print(f"{'Tipo de entidad':<15} | {'Comentarios de odio':<20} | {'Comentarios sin odio':<20}")
print("-" * 60)
for tipo in sorted(todos_tipos):
    count_odio = tipos_entidades_odio.get(tipo, 0)
    count_no_odio = tipos_entidades_no_odio.get(tipo, 0)
    print(f"{tipo:<15} | {count_odio:<20} | {count_no_odio:<20}")

# Análisis - Pregunta 10: Lemas más frecuentes
print("\n=== ANÁLISIS 8: Lemas más frecuentes (Top 20) ===")
lemas_odio = [token.lemma_.lower() for doc in comentarios_odio 
             for token in doc 
             if not token.is_stop and not token.is_punct and not token.is_space]

lemas_no_odio = [token.lemma_.lower() for doc in comentarios_no_odio 
                for token in doc 
                if not token.is_stop and not token.is_punct and not token.is_space]

contador_odio = Counter(lemas_odio)
contador_no_odio = Counter(lemas_no_odio)

print(f"{'Posición':<8} | {'Lema (odio)':<15} | {'Frecuencia':<10} | {'Lema (no odio)':<15} | {'Frecuencia':<10}")
print("-" * 70)

# Mostrar solo top 20 para no saturar la consola
top_odio = contador_odio.most_common(20)
top_no_odio = contador_no_odio.most_common(20)
max_items = min(20, len(top_odio), len(top_no_odio))

for i in range(max_items):
    lema_odio, freq_odio = top_odio[i]
    lema_no_odio, freq_no_odio = top_no_odio[i]
    print(f"{i+1:<8} | {lema_odio:<15} | {freq_odio:<10} | {lema_no_odio:<15} | {freq_no_odio:<10}")

print("\n=== Análisis completado con aceleración GPU ===")
print(f"Dispositivo utilizado: {'GPU' if cuda_info['gpu_active'] else 'CPU'}")
if cuda_info['gpu_active']:
    print(f"Modelo de GPU: {cuda_info['device_name']}")
print(f"Tiempo total de procesamiento de textos: {processing_time:.2f} segundos") 