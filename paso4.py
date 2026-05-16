import funciones as f
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, binary_closing, binary_opening
from skimage.morphology import disk, remove_small_holes
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont

#cargar imagen
img_raw = f.cargar_imagen('img/pensamientos.jpg')
# img_raw = funciones.cargar_imagen(r"img/flores de lupino.png")
h, w, c = img_raw.shape

#1. reducir cantidad de colores
img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=16)

# Image.fromarray(img_cuantizada).save('output/paso2/filtrada_0.png')

# 2. eliminar ruido de 1px
img_cuantizada_filtrada, labels_filtrados = f.eliminar_1px(img_cuantizada, labels, colores_paleta)

#3. segmentar y reducir regiones pequeñas
def calcular_area_minima(h, w, porcentaje=0.1):
    calc = int((h * w) * (porcentaje / 100))
    return min(calc, 100) 

porcentaje = 0.1
area_minima_est = calcular_area_minima(h, w, porcentaje)

mapa_regiones, df_regiones = f.segmentar_regiones(
    img_cuantizada_filtrada,
    labels_filtrados,
    colores_paleta
)

#filtrar regiones pequeñas
mapa_regiones_limpio, df_regiones_limpio, stats_filtrado = f.filtrar_regiones_pequenas(
    mapa_regiones,
    df_regiones,
    area_minima_est
)

# Preparar imagen ya filtrada
mapa_colores = dict(zip(df_regiones_limpio['region_id'], df_regiones_limpio['color_id']))

img_regiones_filtrado = np.vectorize(mapa_colores.get)(mapa_regiones_limpio.reshape(-1))

img_regiones_filtrado = img_regiones_filtrado.reshape(h, w)

img_regiones_filtradas = colores_paleta[img_regiones_filtrado - 1]

KERNEL_SIZE = 9  # 3, 5, 7... (impar) - Mayor = más suavizado
APLICAR_MORFOLOGIA = True  # True = limpia mejor, False = más rápido

# Suavizar bordes
mapa_regiones_suavizado, df_regiones_suavizado, stats_suavizado = f.suavizar_bordes(
    mapa_regiones_limpio,
    df_regiones_limpio,
    kernel_size=KERNEL_SIZE,
    aplicar_morfologia=APLICAR_MORFOLOGIA
)

GROSOR_BORDE = 0.1  # Grosor de los bordes en píxeles (2-5 recomendado)
TAMANO_NUMERO = 7  # 'auto' o un número fijo (ej: 20, 30, 40)
FONDO_BLANCO = True  # True = fondo blanco, False = colores tenues de guía

img_para_pintar, img_solucion, paleta_numerada = f.generar_imagen_para_pintar(
    # mapa_regiones_suavizado,
    mapa_regiones_limpio,
    # df_regiones_suavizado,
    df_regiones_limpio,
    colores_paleta,
    grosor_borde=GROSOR_BORDE,
    tamano_numero=TAMANO_NUMERO,
    fondo_blanco=FONDO_BLANCO
)

archivos = f.guardar_archivos_finales(
    img_para_pintar,
    img_solucion,
    paleta_numerada,
    # df_regiones_suavizado,
    df_regiones_limpio,
    prefijo="mi_pintar_por_numeros"
)