import funciones as f
from PIL import Image
# import importlib
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
# from scipy.ndimage import generic_filter
# from collections import Counter
# import numpy as np
# from scipy import ndimage
# import pandas as pd
# importlib.reload(funciones)

img_raw = f.cargar_imagen('img/pensamientos.jpg')
# img_raw = funciones.cargar_imagen(r"img/flores de lupino.png")
img_cuantizada, paleta_img, labels, colores_paleta = f.cuantizar_imagen(img_raw, n_colores=16)

Image.fromarray(img_cuantizada).save('output/paso2/filtrada_0.png')

img_cuantizada_filtrada, labels_filtrados = f.eliminar_1px(img_cuantizada, labels, colores_paleta)