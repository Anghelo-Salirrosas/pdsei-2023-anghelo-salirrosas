# Importamos las librerias
from ultralytics import YOLO
import cv2
import pygame
import os

# Inicializar pygame
pygame.init()

# Leer nuestro modelo
model = YOLO("modeloPulidoI.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Configuración de sonidos
carpeta_notas = "NotasMusicales"
notas = {
    "DO": "Do.mp3",
    "azul": "Re.mp3",
    "MI": "Mi.mp3",
    "verde": "Fa.mp3",
    "SOL": "Sol.mp3",
    "Naranja": "La.mp3",
    "Rojo": "Si.mp3"
}

# Obtener la ruta completa de la carpeta de notas
ruta_carpeta_notas = os.path.join(os.path.dirname(os.path.realpath(__file__)), carpeta_notas)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.90)

    # Obtener la primera nota detectada (puedes ajustar esto según tus necesidades)
    if len(resultados):
        primera_nota = resultados[0].labels[0]
        print(f"Nota detectada: {primera_nota}")

        # Reproducir el sonido correspondiente
        if primera_nota in notas:
            ruta_sonido = os.path.join(ruta_carpeta_notas, notas[primera_nota])
            pygame.mixer.music.load(ruta_sonido)
            pygame.mixer.music.play()

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    print(anotaciones)
    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
