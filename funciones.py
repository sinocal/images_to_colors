import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def cargar_imagen(ruta):
    # ruta = 'img/pensamientos.jpg'
    img_raw = Image.open(ruta)

    if img_raw.mode != 'RGB':
        img_raw = img_raw.convert('RGB')
    
    return np.array(img_raw)

def cuantizar_imagen(img, n_colores=16):
    h, w, c = img.shape

    pixels = img.reshape(-1, 3)
    pixels_float = pixels.astype(float)


    # n_colores = 16
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

    paleta_img = np.zeros((160, 640, 3), dtype=np.uint8)
    ancho_bloque = 40

    for i, color in enumerate(colores_paleta):
        x0 = i * ancho_bloque
        x1 = x0 + ancho_bloque

        paleta_img[:, x0:x1] = color

    return img_cuantizada, paleta_img, labels.reshape(h, w), colores_paleta