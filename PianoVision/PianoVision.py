# Importamos las librerias
from ultralytics import YOLO
import cv2
import os

# Leer nuestro modelo
model = YOLO("modeloPulidoI.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.90)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("PIANO VISION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
