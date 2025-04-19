import spacy

# Cargar el modelo en inglés
nlp = spacy.load('en_core_web_sm')

# Procesar un texto de ejemplo
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# Mostrar tokens y etiquetas
print("Tokens y etiquetas POS:")
for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.dep_}")

# Mostrar entidades nombradas
print("\nEntidades nombradas:")
for ent in doc.ents:
    print(f"{ent.text}\t{ent.label_}")

# Mostrar información sobre el modelo
print(f"\nInformación del modelo: {nlp.meta['name']}, versión: {nlp.meta['version']}") 