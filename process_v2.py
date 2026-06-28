import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import funciones as f
from PIL import Image
import numpy as np
import pandas as pd
from scipy import ndimage

os.makedirs('output/paso2', exist_ok=True)
os.makedirs('output_pintar_numeros', exist_ok=True)

#--------------------------------------------------------------
# Paso 0: cargar imagen
#--------------------------------------------------------------
img_raw = f.cargar_imagen('img/pensamientos.jpg')
h, w, c = img_raw.shape
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 1: reducir cantidad de colores
#--------------------------------------------------------------
img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=16)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 2: eliminar ruido de 1px
#--------------------------------------------------------------
img_cuantizada_filtrada, labels_filtrados = f.eliminar_1px(img_cuantizada, labels, colores_paleta)
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

mapa_regiones_limpio, df_regiones_limpio, _ = f.filtrar_regiones_pequenas(
    mapa_regiones,
    df_regiones,
    area_minima_est
)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 3.5: separar apéndices por cuello de botella
#--------------------------------------------------------------
TAMANO_NUMERO = 10

mapa_regiones_limpio, df_regiones_limpio = f.separar_apendices_por_cuello(
    mapa_regiones_limpio,
    df_regiones_limpio,
    tamano_fuente=TAMANO_NUMERO
)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 3.75: fusionar regiones standalone que siguen siendo demasiado delgadas
#--------------------------------------------------------------
mapa_regiones_limpio, df_regiones_limpio = f.fusionar_regiones_delgadas(
    mapa_regiones_limpio,
    df_regiones_limpio,
    tamano_fuente=TAMANO_NUMERO
)
#--------------------------------------------------------------


#--------------------------------------------------------------
# Paso 4: diagnóstico — zonas sin espacio para el número
#--------------------------------------------------------------
img_diagnostico = f.visualizar_zonas_sin_espacio(
    mapa_regiones_limpio,
    df_regiones_limpio,
    colores_paleta,
    tamano_fuente=TAMANO_NUMERO
)
Image.fromarray(img_diagnostico).save('output/diagnostico_v2.png')
print(f"Regiones tras separación: {len(df_regiones_limpio)}")
#--------------------------------------------------------------


# Paso 5 y 6: generación de imagen para pintar (pendiente)
# img_para_pintar, img_solucion, paleta_numerada = f.generar_imagen_para_pintar(
#     mapa_regiones_limpio, df_regiones_limpio, colores_paleta,
#     grosor_borde=1, tamano_numero=TAMANO_NUMERO, fondo_blanco=True
# )
# f.guardar_archivos_finales(
#     img_para_pintar, img_solucion, paleta_numerada,
#     df_regiones_limpio, prefijo="mi_pintar_por_numeros_v2"
# )
