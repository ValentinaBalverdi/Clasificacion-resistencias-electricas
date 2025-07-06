import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
# ---- Segmentacion de la Resistencia -------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def ordenar_puntos(puntos):
    """
    Recibe un array con 4 puntos (x,y) en orden arbitrario
    y devuelve [tl, tr, br, bl] (top-left, top-right…).
    """
    s = puntos.sum(axis=1)
    diff = np.diff(puntos, axis=1)
    tl = puntos[np.argmin(s)]
    br = puntos[np.argmax(s)]
    tr = puntos[np.argmin(diff)]
    bl = puntos[np.argmax(diff)]
    return np.array([tl, tr, br, bl])

def crear_mascara_azul(img):
    """Construye y limpia con morfología la máscara del rectángulo azul."""
    # Umbrales para azules en HSV
    azul_bajo = np.array([ 90,  80,  40]) 
    azul_alto = np.array([150, 255, 255])

    # Kernel para morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    # Convertimos la imagen a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Aplicamos mascara azul
    mask = cv2.inRange(img_hsv, azul_bajo, azul_alto)
    # Aplicamos morfología
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def extraer_rectangulo_azul(mask):
    """
    Encuentra el contorno más grande en la máscara,
    intenta aproximar a polígono de 4 vértices y los devuelve.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se encontraron contornos")   ############
    
    # Seleccionamos el contorno de mayor área 
    main_contour = max(contours, key=cv2.contourArea)

    # Encontramos el polígono convexo mínimo de 4 vertices que contiene todos los puntos del contorno
    hull = cv2.convexHull(main_contour)
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        raise RuntimeError(f"No se pudo aproximar a 4 vértices (encontrados: {len(approx)})")

    pts = approx.reshape(4, 2)
    
    return ordenar_puntos(pts)

def convertir_a_vista_superior(img, src_pts):
    """
    Dada la imagen original y 4 puntos ordenados,
    retorna la imagen con perspectiva corregida ('vista superior')
    utilizando Homografía.
    """
    # Calculamos dimensiones destino
    (tl, tr, br, bl) = src_pts
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    dst_w = int(max(widthA, widthB))
    dst_h = int(max(heightA, heightB))

    dst_pts = np.array([
        [0,       0],
        [dst_w-1, 0],
        [dst_w-1, dst_h-1],
        [0,       dst_h-1]
    ], dtype="float32")

    M = cv2.findHomography(src_pts, dst_pts)[0]
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

def procesar_imagen(ruta):
    """Pipeline de mascara, detectar rectángulo, corregir perspectiva"""
    img_bgr = cv2.imread(str(ruta))
    mask = crear_mascara_azul(img_bgr)
    src_pts = extraer_rectangulo_azul(mask)
    top_view = convertir_a_vista_superior(img_bgr, src_pts)
    return top_view

# -------------------------------------------------------------------------
# Directorio donde están las imágenes originales
carpeta_entrada = "Resistencias"
carpeta_salida = "Resistencias_out"

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

for i in range(1, 11):           
    for j in ['a', 'b', 'c', 'd']:
        nombre = f"R{i}_{j}.jpg"
        ruta = os.path.join(carpeta_entrada, nombre)
        print("Leyendo:", ruta)
        if not os.path.exists(ruta):
            print("¡No existe!, compruebe el directorio de trabajo", ruta)
            continue
        imagen_lista = procesar_imagen(ruta)

        # Guardamos en carpeta nueva con sufijo "_out"
        nombre_salida = f"R{i}_{j}_out.jpg"
        path_salida = os.path.join(carpeta_salida, nombre_salida)
        cv2.imwrite(path_salida, imagen_lista)

# -------------------------------------------------------------------------------------------------
# ------ Clasificacion de las resistencias --------------------------------------------------------
# -------------------------------------------------------------------------------------------------
color_a_valor = {
    "Negro":   0,
    "Marron":  1,
    "Rojo":    2,
    "Naranja": 3,
    "Amarillo":4,
    "Verde":   5,
    "Azul":    6,
    "Violeta": 7,
    "Gris":    8,
    "Blanco":  9
}
def calcular_resistencia(banda1: str, banda2: str, banda3: str) -> int:
    """
    Calcula el valor de una resistencia en ohmios a partir de las tres bandas:
      - banda1 y banda2 representan los dígitos significativos (0–9).
      - banda3 es el multiplicador (10^n).

    Parámetros:
        banda1 (str): color de la primera banda (dígito decenas).
        banda2 (str): color de la segunda banda (dígito unidades).
        banda3 (str): color de la tercera banda (multiplicador).

    Retorna:
        int: valor de la resistencia en ohmios.
    """
    # Obtener los dígitos correspondientes
    d1 = color_a_valor[banda1]
    d2 = color_a_valor[banda2]
    # El exponente para el multiplicador es el mismo valor numérico del color
    exponente = color_a_valor[banda3]

    # Construir el número base de dos dígitos y luego aplicar el multiplicador
    valor_base = d1 * 10 + d2
    resistencia_ohm = valor_base * (10 ** exponente)
    return resistencia_ohm

def mascara_azul_inversa(img_bgr):
    """
    Genera una máscara binaria inversa del área azul en la imagen.

    Esta función detecta las regiones azules en una imagen en formato BGR, 
    las elimina (invirtiendo la máscara) y aplica operaciones morfológicas 
    (clausura y apertura) para eliminar ruido y cerrar huecos.

    Parámetros:
        img_bgr (np.ndarray): Imagen de entrada en formato BGR.

    Retorna:
        np.ndarray: Máscara binaria (0-255) con fondo blanco y la región no azul segmentada en negro.
    """

    azul_bajo = np.array([ 90,  80,  40]) 
    azul_alto = np.array([150, 255, 255])

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, azul_bajo, azul_alto)
    mask = cv2.bitwise_not(mask) # invertimos

    kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mascara_clausura = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clausura, iterations=2)
    mascara_apertura = cv2.morphologyEx(mascara_clausura, cv2.MORPH_OPEN, kernel_apertura, iterations=1)
    return mascara_apertura

def extraer_resistencia(mascara_apertura, img_bgr):
    """
    Extrae la región de interés correspondiente a la resistencia.

    Utiliza la máscara para encontrar el contorno principal, calcula un
    bounding box, aplica padding, recorta la imagen y la redimensiona 
    proporcionalmente a un ancho fijo.

    Parámetros:
        mascara_apertura (np.ndarray): Máscara binaria que contiene la resistencia segmentada.
        img_bgr (np.ndarray): Imagen original en formato BGR.

    Retorna:
        np.ndarray: Imagen de la resistencia recortada y redimensionada.
    """
    contornos, _ = cv2.findContours(mascara_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_max = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cont_max)
    padding = 25
    x1 = max(x + padding, 0)
    y1 = max(y + padding, 0)
    x2 = min(x + w - padding, img_bgr.shape[1])
    y2 = min(y + h - padding, img_bgr.shape[0])
    roi = img_bgr[y1:y2, x1:x2]

    # Redimensionar manteniendo proporción
    h_roi, w_roi = roi.shape[:2]
    new_h = int(h_roi * (200 / w_roi))
    resistor = cv2.resize(roi, (200, new_h))
    return resistor

def bordes_con_sobel(img_bgr):
    """
    Detecta bordes verticales utilizando el operador de Sobel sobre la imagen.

    Aplica CLAHE para mejorar el contraste, suaviza la imagen con un filtro Gaussiano,
    y luego aplica el gradiente en x (Sobel) para resaltar las transiciones verticales.

    Parámetros:
        img_bgr (np.ndarray): Imagen en formato BGR.

    Retorna:
        np.ndarray: Imagen en escala de grises con bordes verticales resaltados.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    gray_smooth = cv2.GaussianBlur(gray_eq, (5,5), 0)

    # O Sobel para resaltar bordes
    sobelx = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    
    return sobelx

def detectar_lineas_verticales(img_sobel, min_dist=10,umbral_inicial = 0.06,umbral_max = 0.5,paso_umbral = 0.005,num_bordes_esperado = 8):
    """
    Detecta bordes verticales utilizando el operador de Sobel sobre la imagen.

    Aplica CLAHE para mejorar el contraste, suaviza la imagen con un filtro Gaussiano,
    y luego aplica el gradiente en x (Sobel) para resaltar las transiciones verticales.

    Parámetros:
        img_bgr (np.ndarray): Imagen en formato BGR.

    Retorna:
        np.ndarray: Imagen en escala de grises con bordes verticales resaltados.
    """
    # Calculamos el promedio de intensidad por columna (perfil vertical)
    perfil = np.mean(img_sobel, axis=0)

    # Derivamos el perfil para detectar bordes verticales (gradientes)
    gradiente = np.abs(np.diff(perfil))

    # Variables para el bucle
    umbral = umbral_inicial
    franjas = []
    iteracion = 0
    max_iter = 30

    while iteracion < max_iter:
        # Umbral final como valor absoluto
        umbral_val = umbral * np.max(gradiente)
        
        # Detectamos picos sobre umbral
        picos = np.where(gradiente > umbral_val)[0]

        # Filtramos bordes muy cercanos
        franjas = []
        for pico in picos:
            if len(franjas) == 0 or (pico - franjas[-1] > min_dist):
                franjas.append(pico)

        # Cantidad de franjas detectadas
        num_franjas = len(franjas)

        # Comparamos con el número esperado 
        if num_franjas == num_bordes_esperado:
                break
        elif num_franjas > num_bordes_esperado:
            umbral += paso_umbral  # suba el umbral para detectar menos franjas
            if umbral > umbral_max:
                umbral = umbral_max
                break
        else:
            # número correcto o aceptable de franjas detectadas
            break

        iteracion += 1

    print(f"Umbral final ajustado: {umbral:.3f}")
    print(f"Franjas detectadas: {len(franjas)}")
    return franjas

def recortar_banda(img_bgr, franja):
    """
    Recorta una franja vertical (banda de color) de la imagen BGR.

    La franja se define por su rango horizontal. Convierte la imagen a RGB 
    para facilitar el análisis de color.

    Parámetros:
        img_bgr (np.ndarray): Imagen de entrada en formato BGR.
        franja (tuple): Tupla (x1, x2) con las columnas que definen la banda.

    Retorna:
        np.ndarray: Subimagen RGB correspondiente a la banda recortada.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x1, x2 = franja
    banda = img_rgb[:, x1:x2]
    return banda

   
def detectar_bandas(img_bgr):
    """
    Detecta las 3 bandas de color útiles para leer el valor de la resistencia.

    Usa detección de bordes verticales y calcula las distancias entre bandas internas
    para determinar si la franja dorada está a la izquierda o derecha. 
    Luego devuelve las 3 bandas activas: banda1, banda2 y multiplicador.

    Parámetros:
        img_bgr (np.ndarray): Imagen de la resistencia en vista superior.

    Retorna:
        tuple: Tres imágenes RGB correspondientes a banda1, banda2 y banda3.
    """
    imagen_sobel = bordes_con_sobel(img_bgr)
    franjas = detectar_lineas_verticales(imagen_sobel)
    franjas_pares = [(franjas[i], franjas[i + 1]) for i in range(0, len(franjas) - 1, 2)]

    # Calcular distancias entre bandas internas: 2-3, 4-5, 6-7
    distancia_12 = franjas[2] - franjas[1]
    distancia_34 = franjas[4] - franjas[3]
    distancia_56 = franjas[6] - franjas[5]

    # Determinar posición de la franja dorada (solo puede estar en bordes 0-1 o 7-8)
    if distancia_12 > distancia_56:
        posicion_dorada = 'izquierda'
        franja_dorada = (franjas[0], franjas[1])
        
        banda_3 = (franjas[2], franjas[3])
        banda_2   = (franjas[4], franjas[5])
        banda_1  = (franjas[6], franjas[7])
    else:
        posicion_dorada = 'derecha'
        franja_dorada = (franjas[6], franjas[7])
        
        banda_3 = (franjas[4], franjas[5])
        banda_2   = (franjas[2], franjas[3])  
        banda_1  = (franjas[0], franjas[1])

    print(f"Franja dorada detectada a la {posicion_dorada}")
    b1 = recortar_banda(img_bgr, banda_1)
    b2 = recortar_banda(img_bgr, banda_2)
    b3 = recortar_banda(img_bgr, banda_3)

    return b1, b2, b3

def puntos_promedio_color_rgb(banda_rgb, num_puntos=5, margen_vertical=10):
    """
    Calcula el color promedio de una banda vertical a partir de varios puntos.

    Toma varios píxeles distribuidos verticalmente en el centro de la banda
    y promedia sus valores RGB.

    Parámetros:
        banda_rgb (np.ndarray): Imagen de la banda en formato RGB.
        num_puntos (int): Número de puntos verticales a considerar.
        margen_vertical (int): Distancia entre los puntos.

    Retorna:
        np.ndarray: Vector RGB (int) con el color promedio estimado.
    """
    alto, ancho, _ = banda_rgb.shape
    x_centro = ancho // 2

    # Tomamos puntos verticales centrados, separados por margen_vertical
    y_inicio = alto // 2 - margen_vertical * (num_puntos // 2)
    puntos_y = [y_inicio + i * margen_vertical for i in range(num_puntos)]

    colores = []
    for y in puntos_y:
        # Control para que y esté dentro de la imagen
        y = max(0, min(y, alto-1))
        color = banda_rgb[y, x_centro]
        colores.append(color)

    color_promedio = np.mean(colores, axis=0)
    return color_promedio.astype(int)

def detectar_color_rgb(banda_rgb):
    """
    Clasifica el color promedio de una banda en una categoría (Rojo, Marrón, etc).

    Compara el color promedio con rangos definidos por canal para identificar 
    el color. Si no cae en ningún rango, devuelve 'desconocido'.

    Parámetros:
        banda_rgb (np.ndarray): Imagen RGB de la banda recortada.

    Retorna:
        str: Nombre del color detectado ('Rojo', 'Verde', etc.) o 'desconocido'.
    """
    color_promedio = puntos_promedio_color_rgb(banda_rgb)
    # Definimos los colores con sus rangos (mín, máximo) por canal RGB
    colores_rangos = {
        'Marron': {
            'min': np.array([62, 13, 2]),  
            'max': np.array([122, 73, 62])  
        },
        'Rojo': {
            'min': np.array([122, 21, 14]),
            'max': np.array([182, 81, 74])
        },
        'Naranja': {
            'min': np.array([149, 71, 16]),
            'max': np.array([209, 131, 76])
        },
        'Amarillo': {
            'min': np.array([150, 120, 20]),  
            'max': np.array([255, 200, 100])
        },
        'Violeta': {
            'min': np.array([50, 30, 70]),
            'max': np.array([130, 90, 150])
        },
        'Verde': {
            'min': np.array([30, 50, 20]),
            'max': np.array([80, 140, 80])
        },
        'Blanco': {
            'min': np.array([138, 115, 99]),   
            'max': np.array([198, 175, 159])  
        },
        'Negro': {
            'min': np.array([3, 0, 0]),       
            'max': np.array([60, 45, 40])     
        }
    }

    for color_nombre, rangos in colores_rangos.items():
        if np.all(color_promedio >= rangos['min']) and np.all(color_promedio <= rangos['max']):
            return color_nombre

    return 'desconocido'


def procesar_resistencia(ruta):
    """Pipeline de mascara, detectar rectángulo, corregir perspectiva"""
    img_bgr = cv2.imread(str(ruta))
    mask = mascara_azul_inversa(img_bgr)
    resistencia_recortada = extraer_resistencia(mask,img_bgr)
    img_banda1, img_banda2, img_banda3 = detectar_bandas(resistencia_recortada)
    c1 = detectar_color_rgb(img_banda1)
    c2 = detectar_color_rgb(img_banda2)
    c3 = detectar_color_rgb(img_banda3)

    valor = calcular_resistencia(c1, c2, c3)
    return c1, c2, c3, valor

# Bucle principal
for i in range(1, 11):
    nombre = f"R{i}_a_out.jpg"
    ruta = os.path.join(carpeta_salida, nombre)
    print("Leyendo:", ruta)
    if not os.path.exists(ruta):
        print("¡No existe!, compruebe el directorio de trabajo", ruta)
        continue
    try:
        c1, c2, c3, valor = procesar_resistencia(ruta)
        print(f"Resistencia {nombre}:")
        print(f"  Banda 1: {c1}")
        print(f"  Banda 2: {c2}")
        print(f"  Banda 3: {c3}")
        print(f"  Valor: {valor} Ω\n")
    except Exception as e:
        print(f"[ERROR] {nombre}: {e}")

