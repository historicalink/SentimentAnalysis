
#Diccionario de categorias
CATEGORIES_DICT = {
    0: 'NEGATIVO',
    1: 'NEUTRO',
    2: 'POSITIVO'
#    3: 'IRONÍA'
}

#Aplica regularizacion R2
WEIGHT_DECAY = 0.001

#Define la seed que se usará durante el proyecto
SEED = 42

#Ubicacion de los datos de entrenamiento
TRAINING_DATA = '../TrainingData/corpus_tagged.csv'

#Define el tamaño del batch de entrenamiento
BATCH_SIZE = 4

#Epocas de entrenamiento
EPOCHS = 60

#Tasa de aprendizaje
LR = 0.02

#Texto por defecto para prueba
text = """
    La igualdad supone comparación, y la comparación cosas ó personas que han de ser comparadas. Ya se sabe que todo sér es idéntico á sí mismo; de modo que, cuando se dice igual, evidentemente hay que referirse á otro. Igualdad supone pluralidad de personas ó cosas que no se aislan, sino que, por el contrario, se aproximan para compararlas ó ser comparadas.
"""