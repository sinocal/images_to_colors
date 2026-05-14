import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


## Cargar la imagen ##
ruta = 'img/pensamientos.jpg'
img_raw = Image.open(ruta)

if img_raw.mode != 'RGB':
    img_raw = img_raw.convert('RGB')

img = np.array(img_raw)


h, w, c = img.shape

pixels = img.reshape(-1, 3)
pixels_float = pixels.astype(float)


n_colores = 16
kmeans = KMeans(
    n_clusters=n_colores,
    init='k-means++',  # Mejor inicialización
    max_iter=300,
    n_init=10,  # Número de veces que se ejecuta con diferentes centroides
    random_state=42,
    verbose=0
)

labels = kmeans.fit_predict(pixels_float)
colores_paleta = kmeans.cluster_centers_

img_cuantizada = colores_paleta[labels]
img_cuantizada = img_cuantizada.reshape(h, w, c).astype(np.uint8)

colores_paleta = colores_paleta.astype(np.uint8)

paleta_vacia = np.zeros((160, 640, 3), dtype=np.uint8)
ancho_bloque = 40

for i, color in enumerate(colores_paleta):
    x0 = i * ancho_bloque
    x1 = x0 + ancho_bloque

    paleta_vacia[:, x0:x1] = color

## guardar imagen cuantizada y colores paleta ##
Image.fromarray(img_cuantizada).save('output/paso1/cuantizada.png')
Image.fromarray(paleta_vacia).save('output/paso1/colores_paleta.png')
