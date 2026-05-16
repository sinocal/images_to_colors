import funciones as f
from PIL import Image
import numpy as np
import pandas as pd

img_raw = f.cargar_imagen('img/pensamientos.jpg')
# img_raw = funciones.cargar_imagen(r"img/flores de lupino.png")
h, w, c = img_raw.shape
img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=16)

Image.fromarray(img_cuantizada).save('output/paso2/filtrada_0.png')

img_cuantizada_filtrada, labels_filtrados = f.eliminar_1px(img_cuantizada, labels, colores_paleta)

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

Image.fromarray(img_regiones_filtradas).save('output/paso3/regiones_filtrada_final.png')