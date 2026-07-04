import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import funciones as f
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, binary_closing, binary_opening
from skimage.morphology import disk, remove_small_holes
from scipy import ndimage
import matplotlib.pyplot as plt

os.makedirs('output/paso2', exist_ok=True)
os.makedirs('output_pintar_numeros', exist_ok=True)

#--------------------------------------------------------------
# Paso 0: cargar imagen
#--------------------------------------------------------------
img_raw = f.cargar_imagen('img/test3.jpeg')
# img_raw = f.cargar_imagen('img/pensamientos.jpg')
# img_raw = funciones.cargar_imagen(r"img/flores de lupino.png")
h, w, c = img_raw.shape
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 1: reducir cantidad de colores
#--------------------------------------------------------------

img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=24)

#guardar resultado paso 1
# Image.fromarray(img_cuantizada).save('output/paso2/filtrada_0.png')
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 2: eliminar ruido de 1px
#--------------------------------------------------------------
img_cuantizada_filtrada, labels_filtrados = f.eliminar_1px(img_cuantizada, labels, colores_paleta)
#guardar resultado paso 2
# Image.fromarray(img_cuantizada_filtrada).save('output/paso2/filtrada_1.png')
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 3: segmentar y reducir regiones pequeñas
#--------------------------------------------------------------
porcentaje = 0.1
area_minima_est = f.calcular_area_minima(h, w, porcentaje, area_minima=100)

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
# Guardar paso 3
mapa_colores = dict(zip(df_regiones_limpio['region_id'], df_regiones_limpio['color_id']))

img_regiones_filtrado = np.vectorize(mapa_colores.get)(mapa_regiones_limpio.reshape(-1))

img_regiones_filtrado = img_regiones_filtrado.reshape(h, w)

img_regiones_filtradas = colores_paleta[img_regiones_filtrado - 1]

# Image.fromarray(img_regiones_filtradas).save('output/pensamientos_paso3.png')
Image.fromarray(img_regiones_filtradas).save('output/test.png')


#--------------------------------------------------------------
# Paso 3.5: fusionar regiones demasiado delgadas para caber un número
#--------------------------------------------------------------
TAMANO_NUMERO = 10
mapa_regiones_limpio, df_regiones_limpio = f.fusionar_regiones_delgadas(
    mapa_regiones_limpio,
    df_regiones_limpio,
    tamano_fuente=TAMANO_NUMERO
)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 5: generar imagen para pintar por números
#--------------------------------------------------------------
img_para_pintar, img_solucion, paleta_numerada = f.generar_imagen_para_pintar(
    mapa_regiones_limpio,
    df_regiones_limpio,
    colores_paleta,
    grosor_borde=1,
    tamano_numero=TAMANO_NUMERO,
    fondo_blanco=True
)

f.guardar_archivos_finales(
    img_para_pintar,
    img_solucion,
    paleta_numerada,
    df_regiones_limpio,
    prefijo="mi_pintar_por_numeros"
)
#--------------------------------------------------------------

