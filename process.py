import funciones as f
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, binary_closing, binary_opening
from skimage.morphology import disk, remove_small_holes
from scipy import ndimage
import matplotlib.pyplot as plt

#--------------------------------------------------------------
# Paso 0: cargar imagen
#--------------------------------------------------------------
img_raw = f.cargar_imagen('img/pensamientos.jpg')
# img_raw = funciones.cargar_imagen(r"img/flores de lupino.png")
h, w, c = img_raw.shape
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 1: reducir cantidad de colores
#--------------------------------------------------------------

img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=16)

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
#--------------------------------------------------------------

resultado = f.colorear_zonas_delgadas(
    mapa_regiones,
    df_regiones_limpio,
    colores_paleta,
    grosor_max=10
)

Image.fromarray(resultado).save('output/paso3_2/regiones_delgadas_destacadas.png')


# Preparar imagen ya filtrada
# mapa_colores = dict(zip(df_regiones_limpio['region_id'], df_regiones_limpio['color_id']))

# img_regiones_filtrado = np.vectorize(mapa_colores.get)(mapa_regiones_limpio.reshape(-1))

# img_regiones_filtrado = img_regiones_filtrado.reshape(h, w)

# img_regiones_filtradas = colores_paleta[img_regiones_filtrado - 1]

# Image.fromarray(img_regiones_filtradas).save('output/paso3/regiones_filtrada_final.png')


#--------------------------------------------------------------
# Paso 4: generar imagen para pintar por números
#--------------------------------------------------------------
# Parámetros ajustables
GROSOR_BORDE = 1  # Grosor de los bordes en píxeles (2-5 recomendado)
TAMANO_NUMERO = 10  # 'auto' o un número fijo (ej: 20, 30, 40)
FONDO_BLANCO = True  # True = fondo blanco, False = colores tenues de guía

img_para_pintar, img_solucion, paleta_numerada = f.generar_imagen_para_pintar(
    mapa_regiones_limpio,
    df_regiones_limpio,
    colores_paleta,
    grosor_borde=GROSOR_BORDE,
    tamano_numero=TAMANO_NUMERO,
    fondo_blanco=FONDO_BLANCO
)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 5: guardar resultados
#--------------------------------------------------------------
# Guardar archivos
# archivos = f.guardar_archivos_finales(
#     img_para_pintar,
#     img_solucion,
#     paleta_numerada,
#     df_regiones_limpio,
#     prefijo="mi_pintar_por_numeros"
# )
#--------------------------------------------------------------
