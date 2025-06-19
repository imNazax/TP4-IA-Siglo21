import cv2
import numpy as np

def detectar_circunferencias_mejorado(img_path):
    """
    Versión para detectar circunferencias usando transformada de Hough.
    Limita el rango de radios y ajusta la sensibilidad para evitar falsos positivos.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detectar circunferencias
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,             # Distancia mínima entre centros de círculos detectados
        param1=100,             # Umbral para Canny (borde externo)
        param2=40,              # Acumulador más exigente (mayor = menos falsos)
        minRadius=30,           # Radio mínimo de círculos a detectar
        maxRadius=70            # Radio máximo
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)    

    cv2.imshow('Circunferencias mejoradas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = 'C:/Users/Usuario/Desktop/Test/TP4/prueba2.png'
detectar_circunferencias_mejorado(img_path)
