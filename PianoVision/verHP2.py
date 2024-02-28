import pandas as pd
import tensorflow as tf

# Cargar el modelo desde el archivo .h5
modelo = tf.keras.models.load_model('PesosNotas.h5')

# Mostrar la arquitectura del modelo con sus hiperparámetros
modelo.summary()

# Otra opción es acceder a la configuración del modelo
configuracion_modelo = modelo.get_config()

# Crear un DataFrame para almacenar la información
columnas = ['Capa', 'Tipo', 'Nombre', 'Filtros/Unidades', 'Tamaño del kernel', 'Activación']
informacion_capas = []

# Iterar sobre las capas y almacenar la información en el DataFrame
for i, capa_config in enumerate(configuracion_modelo['layers']):
    tipo_capa = capa_config['class_name']
    
    # Verificar si la clave 'name' está presente en la configuración de la capa
    if 'name' in capa_config:
        nombre_capa = capa_config['name']
    else:
        nombre_capa = "Sin nombre"
    
    # Obtener hiperparámetros según el tipo de capa
    if tipo_capa == 'Conv2D':
        filtros = capa_config['config']['filters']
        tamano_kernel = capa_config['config']['kernel_size']
        activacion = capa_config['config']['activation']
        informacion_capas.append([i + 1, tipo_capa, nombre_capa, f'{filtros}', str(tamano_kernel), activacion])
    elif tipo_capa == 'Dense':
        unidades = capa_config['config']['units']
        activacion = capa_config['config']['activation']
        informacion_capas.append([i + 1, tipo_capa, nombre_capa, f'{unidades}', 'N/A', activacion])

# Crear un DataFrame con la información
df = pd.DataFrame(informacion_capas, columns=columnas)

# Mostrar el DataFrame
print(df)

# También puedes guardar el DataFrame en un archivo CSV si lo necesitas
df.to_csv('informacion_modelo.csv', index=False)
