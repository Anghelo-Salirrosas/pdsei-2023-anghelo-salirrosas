import tensorflow as tf
import matplotlib as plt

# Cargar el modelo desde el archivo .h5
modelo = tf.keras.models.load_model('ModeloNotas.h5')

# Mostrar la arquitectura del modelo con sus hiperparámetros
modelo.summary()

# Otra opción es acceder a la configuración del modelo
configuracion_modelo = modelo.get_config()

# Iterar sobre las capas y mostrar los hiperparámetros
# for capa_config in configuracion_modelo['layers']:
#     nombre_capa = capa_config['name']
#     tipo_capa = capa_config['class_name']
    
#     # Mostrar hiperparámetros según el tipo de capa
#     if tipo_capa == 'Conv2D':
#         filtros = capa_config['config']['filters']
#         tamano_kernel = capa_config['config']['kernel_size']
#         activacion = capa_config['config']['activation']
#         print(f'{nombre_capa} ({tipo_capa}): Filtros={filtros}, Tamaño del kernel={tamano_kernel}, Activación={activacion}')
#     elif tipo_capa == 'Dense':
#         unidades = capa_config['config']['units']
#         activacion = capa_config['config']['activation']
#         print(f'{nombre_capa} ({tipo_capa}): Unidades={unidades}, Activación={activacion}')
#     # Agrega más bloques según las capas que tengas en tu modelo

for capa_config in configuracion_modelo['layers']:
    # Verificar si la clave 'name' está presente en la configuración de la capa
    if 'name' in capa_config:
        nombre_capa = capa_config['name']
    else:
        nombre_capa = "Sin nombre"
    
    tipo_capa = capa_config['class_name']
    
    # Mostrar hiperparámetros según el tipo de capa
    if tipo_capa == 'Conv2D':
        filtros = capa_config['config']['filters']
        tamano_kernel = capa_config['config']['kernel_size']
        activacion = capa_config['config']['activation']
        print(f'{nombre_capa} ({tipo_capa}): Filtros={filtros}, Tamaño del kernel={tamano_kernel}, Activación={activacion}')
    elif tipo_capa == 'Dense':
        unidades = capa_config['config']['units']
        activacion = capa_config['config']['activation']
        print(f'{nombre_capa} ({tipo_capa}): Unidades={unidades}, Activación={activacion}')
    # Agrega más bloques según las capas que tengas en tu modelo


# Si quieres obtener los pesos específicos de cada capa:
pesos_capas = modelo.get_weights()
for i, capa in enumerate(modelo.layers):
    print(f'Capa {i + 1} - Pesos: {pesos_capas[i].shape}')

# Obtener pesos de cada capa
pesos_capas = []
for capa in modelo.layers:
    pesos_capa = capa.get_weights()
    pesos_capas.append(pesos_capa)

# Mostrar los pesos de cada capa
for i, pesos_capa in enumerate(pesos_capas):
    print(f'Capa {i + 1} - Pesos: {pesos_capa}')




historia_entrenamiento = modelo.history.history

# Plot de la pérdida
plt.plot(historia_entrenamiento['loss'], label='Entrenamiento')
plt.plot(historia_entrenamiento['val_loss'], label='Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Plot de la precisión
plt.plot(historia_entrenamiento['accuracy'], label='Entrenamiento')
plt.plot(historia_entrenamiento['val_accuracy'], label='Validación')
plt.title('Precisión durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()
