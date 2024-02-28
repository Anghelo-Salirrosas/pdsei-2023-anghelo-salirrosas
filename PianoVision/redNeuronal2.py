import os
import tensorflow as tf
import numpy as np

import random
import PIL
#import PIL.Image
from PIL import Image

import matplotlib.pyplot as plt

tf.keras.backend.clear_session() #Reseteo sencillo

#rutas fotos
ruta_actual = os.path.abspath(os.path.dirname(__file__))
datos_entrenamiento = os.path.join(ruta_actual, 'Fotos', 'Entrenamiento')
datos_validacion = os.path.join(ruta_actual, 'Fotos', 'Validacion')

#Parametros
iteraciones=10 #cuantas veces
altura, longitud=200, 200 #tamaño de imagen
imgEnvio=1 #imagenes por envio
pasos=500 #veces q se va a procesar la info en cada iteracion
pasos_validacion=500/1 #despues de cada iteracion, valida

filtrosconv1 = 32
filtrosconv2 = 64     #Numero de filtros que vamos a aplicar en cada convolucion
filtrosconv3 = 128
tam_filtro1 = (4,4)
tam_filtro2 = (3,3)
tam_filtro3 = (2,2)   #Tamaños de los filtros 1 y 2 y 3
tam_pool = (2,2)  #Tamaño del filtro en max pooling
clases = 7  #7 notas musicales
lr = 0.0005  #ajustes de la red neuronal para acercarse a una solucion optima nota: no subirle mas xD F

#direccionTF=tf.keras.utils.get_file(origin=datos_entrenamiento,untar=True)
# nota="1.MoraDO"

# rutaNota = "../Fotos/Entrenamiento/"+nota

# imagenesNotas=os.listdir(rutaNota)

# seleccionImagen=os.path.join(rutaNota,imagenesNotas[0])
# imagen=Image.open(seleccionImagen)
# imagen.show()

#ruta_dir="D:/UNT CICLO 8/PRUEBA/"

lista_clases = [
    '1.MoraDO',
    '2.AzulRE',
    '3.CelesteMI',
    '4.VerdeFA',
    '5.AmarilloSOL',
    '6.AnaranjadoLA',
    '7.RojoSI'
]

train_ds = tf.keras.utils.image_dataset_from_directory(
    datos_entrenamiento,
    #validation_split=0.1,
    #subset="training",
    #seed=123,
    image_size=(altura,longitud),
    batch_size=imgEnvio,
    labels="inferred",
    label_mode="int",
    class_names=lista_clases
)



# preprocesamiento_entre = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale= 1./255,   #Pasar los pixeles de 0 a 255 | 0 a 1
#     shear_range = 0.3, #Generar nuestras imagenes inclinadas para un  mejor entrenamiento
#     zoom_range = 0.3,  #Genera imagenes con zoom para un mejor entrenamiento
#     horizontal_flip=True #Invierte las imagenes para mejor entrenamiento
# )

capa_preprocesamiento=tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
        #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15)
    ]
)

train_ds2=train_ds.map(
    lambda x, y: (capa_preprocesamiento(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).repeat()


# train_ds = preprocesamiento_entre.flow_from_directory(
#     datos_entrenamiento,       #Va a tomar las fotos que ya almacenamos
#     target_size = (altura, longitud),
#     batch_size = imgEnvio,
#     class_mode = 'categorical',  #Clasificacion categorica = por clases
#     #labels="inferred",
#     #label_mode="int",
#     classes=lista_clases
# )

# # Convertir train_ds a numpy arrays
# train_images = []
# train_labels = []

# for images, labels in train_ds:
#     train_images.append(images.numpy())
#     train_labels.append(labels.numpy())

# train_images = np.vstack(train_images)
# train_labels = np.hstack(train_labels)

# # Crear un generador de datos de entrenamiento con las transformaciones aplicadas a los numpy arrays
# generador_entrenamiento = preprocesamiento_entre.flow(
#     train_images,
#     train_labels,
#     batch_size=imgEnvio,
#     shuffle=False  # Desactivar el barajado para que coincida con los datos originales
# )

# # Reemplazar train_ds con el generador de datos preprocesados
# train_ds = generador_entrenamiento

validation_ds = tf.keras.utils.image_dataset_from_directory(
    datos_validacion,
    #validation_split=0.1,
    #subset="training",
    #seed=123,
    image_size=(altura,longitud),
    batch_size=imgEnvio,
    labels="inferred",
    label_mode="int",
    class_names=lista_clases
)

capa_preprocesamientoVal=tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ]
)

validation_ds2=validation_ds.map(
    lambda x, y: (capa_preprocesamientoVal(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).repeat()

#clasesDentro=train_ds.class_names
clasesDentro=lista_clases
#print(clasesDentro)

#Visualizar los datos
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(9):
#     num_images = min(len(images), 9)  # Asegurarse de no superar el número de imágenes disponibles
#     print(len(images))
#     for i in range(num_images):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(clasesDentro[labels[i]])
#         plt.axis("off")

# plt.show()

# for i, (imagen,labels) in enumerate(train_ds.take(9)):
#     plt.subplot(3,3,i+1)
#     plt.imshow(imagen)
# plt.show()

#FUNCIONA
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     batch = next(iter(train_ds))  # Tomar un lote de imágenes aleatorio
#     images, labels = batch
#     idx = random.randint(0, len(images) - 1)  # Elegir una imagen aleatoria dentro del lote
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[idx].numpy().astype("uint8"))
#     plt.title(clasesDentro[labels[idx]])
#     plt.axis("off")

# plt.show()

# print(len(train_ds))

# #Se define la normalizacion de 0 a 255 a 0 a 1:
# capaNormalizacion = tf.keras.layers.Rescaling(1./255)

# datosNormalizados = train_ds.map(lambda x, y: (capaNormalizacion(x),y))
# image_batch, labels_batch = next(iter(datosNormalizados))
# first_image = image_batch[0]

# print(image_batch)
# print(len(image_batch))
# print(first_image)

# # Pixeles en relacion entre 0 a 1.
# print(np.min(first_image), np.max(first_image))


#USAR CACHE
# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

#CreacionModelo
#neuronas=128*clases
neuronas=128*7
modeloNotas = tf.keras.Sequential(
    [
        #tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura,longitud,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size=tam_pool),
        tf.keras.layers.Conv2D(filtrosconv2,tam_filtro2,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=tam_pool),
        tf.keras.layers.Conv2D(filtrosconv3,tam_filtro3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=tam_pool),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(neuronas,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(clases, activation='softmax')
        
    ]
)

modeloNotas.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

#Entrenar el modelo


modeloNotas.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=iteraciones,
    steps_per_epoch=pasos,
    validation_steps=pasos_validacion
)

modeloNotas.save('ModeloNotas.h5')
modeloNotas.save_weights('PesosNotas.h5')