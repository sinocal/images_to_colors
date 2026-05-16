import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
from scipy.ndimage import generic_filter
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_closing, binary_opening
from skimage.morphology import disk, remove_small_holes
import cv2
from PIL import Image, ImageDraw, ImageFont

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

# Filtro mayoría
def majority(window):
    return Counter(window).most_common(1)[0][0]

def nearest_color(img):
    # Aplicar vecindad 3x3
    pixeles_filtrados = generic_filter(
        img,
        function=majority,
        size=3,
        mode='nearest'
    )
    return pixeles_filtrados

def segmentar_regiones(imagen_cuantizada, labels, colores_paleta):
    """
    Identifica regiones conectadas (connected components) para cada color
    
    Args:
        imagen_cuantizada: imagen con colores reducidos
        labels: etiquetas de cluster de K-means (H, W)
        colores_paleta: array con los 20 colores
    
    Returns:
        mapa_regiones: array donde cada región tiene un ID único
        info_regiones: DataFrame con información de cada región
    """
    h, w = labels.shape
    # print(f"Procesando imagen de {h}x{w} píxeles...")
    
    # Crear mapa de regiones (inicialmente vacío)
    mapa_regiones = np.zeros((h, w), dtype=np.int32)
    
    # Lista para almacenar información de cada región
    regiones_info = []
    region_id_global = 1  # Contador global de regiones
    
    # Procesar cada color por separado
    # print(f"\nBuscando regiones conectadas por color...")
    
    for color_idx in range(len(colores_paleta)):
        # Crear máscara binaria para este color
        mascara_color = (labels == color_idx).astype(np.uint8)
        
        # Encontrar componentes conectados (8-connectivity)
        regiones_etiquetadas, num_regiones = ndimage.label(mascara_color)
        
        if num_regiones == 0:
            continue
        
        # print(f"  Color {color_idx + 1:2d} RGB{tuple(colores_paleta[color_idx])}: {num_regiones:3d} regiones")
        
        # Para cada región de este color
        for region_idx in range(1, num_regiones + 1):
            mascara_region = (regiones_etiquetadas == region_idx)
            area = np.sum(mascara_region)
            
            # Asignar ID global único a esta región
            mapa_regiones[mascara_region] = region_id_global
            
            # Guardar información
            regiones_info.append({
                'region_id': region_id_global,
                'color_id': color_idx + 1,  # 1-indexed para el usuario
                'color_rgb': tuple(colores_paleta[color_idx]),
                'area_pixels': area,
                'porcentaje': 100 * area / (h * w)
            })
            
            region_id_global += 1
    
    # Convertir a DataFrame
    df_regiones = pd.DataFrame(regiones_info)
    
    # print(f"\n{'='*60}")
    # print(f"Total de regiones encontradas: {len(df_regiones)}")
    # print(f"{'='*60}")
    
    return mapa_regiones, df_regiones


def eliminar_1px(img_cuantizada, labels, colores_paleta):
    mapa_regiones, df_regiones = segmentar_regiones(img_cuantizada, labels, colores_paleta)

    df_filtrar_pixeles = []
    n_iteraciones = 50
    historial_iteraciones = []
    i = 1

    df_filtrar_pixeles.append({
        'iteracion': 0,
        'regiones_con_area_1': df_regiones[df_regiones['area_pixels'] == 1].shape[0],
        'total_regiones': df_regiones.shape[0]
    })

    # for i in range(1, n_iteraciones):
    while i <= n_iteraciones:
        print(f"\nIteración {i}")
        #aplicar filtrado
        labels = nearest_color(labels)
        img_filtrada = colores_paleta[labels]

        mapa_regiones, df_regiones = segmentar_regiones(img_filtrada, labels, colores_paleta)    

        n_1px = df_regiones[df_regiones['area_pixels'] == 1].shape[0]
        
        df_filtrar_pixeles.append({
            'iteracion': i,
            'regiones_con_area_1': n_1px,
            'total_regiones': df_regiones.shape[0]
        })

        Image.fromarray(img_filtrada).save(f'output/paso2/filtrada_{i}.png')
        
        #gestion del loop
        historial_iteraciones.append(n_1px)
        if len(historial_iteraciones) > 3:
            historial_iteraciones.pop(0)
        
        if len(historial_iteraciones) == 3 and len(set(historial_iteraciones)) == 1:
            break
        i+=1

    # df_regiones[df_regiones['area_pixels'] == 1].shape[0]

    df_filtrar_pixeles = pd.DataFrame(df_filtrar_pixeles)
    # df_filtrar_pixeles.head()

    return img_filtrada, labels

def encontrar_vecino_mas_cercano(mapa_regiones, region_id, mascara_region, df_regiones):
    """
    Encuentra la región vecina más grande (preferiblemente del mismo color)
    
    Args:
        mapa_regiones: mapa de regiones actual
        region_id: ID de la región a fusionar
        mascara_region: máscara binaria de la región
        df_regiones: DataFrame con info de regiones
    
    Returns:
        ID de la región vecina más apropiada
    """
    # Dilatar la región para encontrar vecinos
    from scipy.ndimage import binary_dilation
    
    estructura = np.ones((3, 3))  # 8-connectivity
    mascara_dilatada = binary_dilation(mascara_region, structure=estructura)
    
    # Encontrar regiones vecinas (en el borde)
    borde = mascara_dilatada & ~mascara_region
    regiones_vecinas = np.unique(mapa_regiones[borde])
    regiones_vecinas = regiones_vecinas[regiones_vecinas != 0]  # Excluir background
    regiones_vecinas = regiones_vecinas[regiones_vecinas != region_id]  # Excluir a sí misma
    
    if len(regiones_vecinas) == 0:
        return None
    
    # Obtener info de la región actual
    color_actual = df_regiones[df_regiones['region_id'] == region_id]['color_id'].values[0]
    
    # Priorizar vecinos del mismo color, luego por tamaño
    vecinos_info = df_regiones[df_regiones['region_id'].isin(regiones_vecinas)].copy()
    
    # Score: mismo color = +1000000, luego por área
    vecinos_info['score'] = vecinos_info['area_pixels'].copy()
    vecinos_info.loc[vecinos_info['color_id'] == color_actual, 'score'] += 1000000
    
    mejor_vecino = vecinos_info.loc[vecinos_info['score'].idxmax(), 'region_id']
    
    return mejor_vecino

def filtrar_regiones_pequenas(mapa_regiones, df_regiones, area_minima):
    """
    Elimina regiones pequeñas fusionándolas con sus vecinos
    
    Args:
        mapa_regiones: mapa de regiones actual
        df_regiones: DataFrame con info de regiones
        area_minima: umbral de área mínima en píxeles
    
    Returns:
        mapa_regiones_limpio: mapa actualizado
        df_regiones_limpio: DataFrame actualizado
        estadisticas: dict con stats del proceso
    """
    print(f"\n{'='*70}")
    print(f"FILTRANDO REGIONES MENORES A {area_minima} PÍXELES")
    print(f"{'='*70}\n")
    
    # Identificar regiones pequeñas
    regiones_pequenas = df_regiones[df_regiones['area_pixels'] < area_minima].copy()
    regiones_grandes = df_regiones[df_regiones['area_pixels'] >= area_minima].copy()
    
    print(f"Regiones pequeñas a eliminar: {len(regiones_pequenas)}")
    print(f"Regiones que se mantienen: {len(regiones_grandes)}")
    
    if len(regiones_pequenas) == 0:
        print("\n¡No hay regiones pequeñas que filtrar!")
        return mapa_regiones, df_regiones, {'eliminadas': 0, 'fusionadas': 0}
    
    # Ordenar por tamaño (procesar las más pequeñas primero)
    regiones_pequenas = regiones_pequenas.sort_values('area_pixels')
    
    # Crear copia del mapa para modificar
    mapa_limpio = mapa_regiones.copy()
    
    # Tracking de fusiones
    fusiones = {}  # region_pequena -> region_destino
    regiones_eliminadas = 0
    
    print("\nProcesando fusiones...")
    for idx, row in regiones_pequenas.iterrows():
        region_id = row['region_id']
        
        # Crear máscara de esta región
        mascara_region = (mapa_regiones == region_id)
        
        if not mascara_region.any():
            continue
        
        # Encontrar vecino más apropiado
        vecino_id = encontrar_vecino_mas_cercano(mapa_limpio, region_id, mascara_region, df_regiones)
        
        if vecino_id is None:
            print(f"  ⚠ Región {region_id} sin vecinos, se mantiene")
            continue
        
        # Fusionar: reasignar píxeles de region_id a vecino_id
        mapa_limpio[mapa_limpio == region_id] = vecino_id
        fusiones[region_id] = vecino_id
        regiones_eliminadas += 1
        
        if regiones_eliminadas % 50 == 0:
            print(f"  Procesadas {regiones_eliminadas} fusiones...")
    
    print(f"\n✓ Fusiones completadas: {regiones_eliminadas}")
    
    # Reconstruir DataFrame de regiones
    print("\nRecalculando áreas de regiones...")
    nuevas_regiones = []
    
    for region_id in np.unique(mapa_limpio):
        if region_id == 0:
            continue
        
        mascara = (mapa_limpio == region_id)
        area = np.sum(mascara)
        
        # Buscar color original de esta región
        info_original = df_regiones[df_regiones['region_id'] == region_id]
        
        if len(info_original) > 0:
            color_id = info_original.iloc[0]['color_id']
            color_rgb = info_original.iloc[0]['color_rgb']
        else:
            # Esta región fue destino de fusión, mantener su info
            continue
        
        nuevas_regiones.append({
            'region_id': region_id,
            'color_id': color_id,
            'color_rgb': color_rgb,
            'area_pixels': area,
            'porcentaje': 100 * area / mapa_limpio.size
        })
    
    df_limpio = pd.DataFrame(nuevas_regiones)
    
    # Estadísticas
    stats = {
        'eliminadas': regiones_eliminadas,
        'fusionadas': len(fusiones),
        'antes': len(df_regiones),
        'despues': len(df_limpio),
        'reduccion_porcentaje': 100 * (len(df_regiones) - len(df_limpio)) / len(df_regiones)
    }
    
    print(f"\n{'='*70}")
    print(f"RESUMEN DE FILTRADO")
    print(f"{'='*70}")
    print(f"Regiones antes: {stats['antes']}")
    print(f"Regiones después: {stats['despues']}")
    print(f"Reducción: {stats['reduccion_porcentaje']:.1f}%")
    print(f"{'='*70}\n")
    
    return mapa_limpio, df_limpio, stats

def suavizar_bordes(mapa_regiones, df_regiones, kernel_size=3, aplicar_morfologia=True):
    """
    Suaviza los bordes de las regiones usando filtros
    
    Args:
        mapa_regiones: mapa de regiones actual
        df_regiones: DataFrame con info de regiones
        kernel_size: tamaño del kernel para median filter (3, 5, 7, etc.)
        aplicar_morfologia: si aplicar operaciones morfológicas adicionales
    
    Returns:
        mapa_suavizado: mapa con bordes suavizados
        df_actualizado: DataFrame actualizado
        stats: estadísticas del proceso
    """
    print(f"\n{'='*70}")
    print(f"SUAVIZANDO BORDES (kernel size: {kernel_size})")
    print(f"{'='*70}\n")
    
    h, w = mapa_regiones.shape
    
    # Paso 1: Aplicar median filter
    print("Aplicando filtro de mediana...")
    mapa_suavizado = median_filter(mapa_regiones, size=kernel_size)
    
    # Paso 2: Operaciones morfológicas por región (opcional pero recomendado)
    if aplicar_morfologia:
        print("Aplicando operaciones morfológicas para limpiar artefactos...")
        
        mapa_morfologico = np.zeros_like(mapa_suavizado)
        
        # Procesar cada región individualmente
        regiones_unicas = np.unique(mapa_suavizado)
        regiones_unicas = regiones_unicas[regiones_unicas != 0]
        
        for i, region_id in enumerate(regiones_unicas):
            if (i + 1) % 20 == 0:
                print(f"  Procesadas {i + 1}/{len(regiones_unicas)} regiones...")
            
            # Máscara binaria de esta región
            mascara = (mapa_suavizado == region_id)
            
            # Cerrar pequeños huecos (closing)
            mascara_cerrada = binary_closing(mascara, structure=disk(2))
            
            # Rellenar huecos internos
            mascara_rellena = remove_small_holes(mascara_cerrada, area_threshold=50)
            
            # Asignar a mapa final
            mapa_morfologico[mascara_rellena] = region_id
        
        mapa_suavizado = mapa_morfologico
        print(f"  ✓ {len(regiones_unicas)} regiones procesadas")
    
    # Paso 3: Re-etiquetar regiones (por si alguna se fragmentó)
    print("\nRe-etiquetando regiones...")
    mapa_final = np.zeros_like(mapa_suavizado)
    nuevas_regiones = []
    nuevo_id = 1
    
    # Diccionario para mapear color_id original
    color_map = df_regiones.set_index('region_id')['color_id'].to_dict()
    rgb_map = df_regiones.set_index('region_id')['color_rgb'].to_dict()
    
    regiones_procesadas = set()
    
    for region_id_original in np.unique(mapa_suavizado):
        if region_id_original == 0 or region_id_original in regiones_procesadas:
            continue
        
        # Encontrar componentes conectados de esta región
        mascara = (mapa_suavizado == region_id_original)
        componentes, num_componentes = ndimage.label(mascara)
        
        # Obtener color original
        color_id = color_map.get(region_id_original, 1)
        color_rgb = rgb_map.get(region_id_original, (0, 0, 0))
        
        # Re-etiquetar cada componente
        for comp_idx in range(1, num_componentes + 1):
            mascara_comp = (componentes == comp_idx)
            area = np.sum(mascara_comp)
            
            if area > 10:  # Ignorar fragmentos muy pequeños
                mapa_final[mascara_comp] = nuevo_id
                
                nuevas_regiones.append({
                    'region_id': nuevo_id,
                    'color_id': color_id,
                    'color_rgb': color_rgb,
                    'area_pixels': area,
                    'porcentaje': 100 * area / (h * w)
                })
                
                nuevo_id += 1
        
        regiones_procesadas.add(region_id_original)
    
    df_actualizado = pd.DataFrame(nuevas_regiones)
    
    # Estadísticas
    stats = {
        'regiones_antes': len(df_regiones),
        'regiones_despues': len(df_actualizado),
        'fragmentacion': len(df_actualizado) - len(df_regiones),
        'kernel_size': kernel_size,
        'morfologia': aplicar_morfologia
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTADO DEL SUAVIZADO")
    print(f"{'='*70}")
    print(f"Regiones antes: {stats['regiones_antes']}")
    print(f"Regiones después: {stats['regiones_despues']}")
    
    if stats['fragmentacion'] > 0:
        print(f"⚠ Fragmentación detectada: +{stats['fragmentacion']} regiones")
        print(f"  (Algunas regiones se dividieron durante el suavizado)")
    elif stats['fragmentacion'] < 0:
        print(f"✓ Fusión detectada: {abs(stats['fragmentacion'])} regiones menos")
    else:
        print(f"✓ Número de regiones se mantuvo estable")
    
    print(f"{'='*70}\n")
    
    return mapa_final, df_actualizado, stats


def visualizar_suavizado(imagen_cuantizada, mapa_antes, mapa_despues, df_antes, df_despues):
    """Visualiza el efecto del suavizado en los bordes"""
    
    from skimage.segmentation import find_boundaries
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ANTES - Bordes
    bordes_antes = find_boundaries(mapa_antes, mode='thick')
    img_bordes_antes = imagen_cuantizada.copy()
    img_bordes_antes[bordes_antes] = [255, 0, 0]  # Rojo
    
    axes[0, 0].imshow(img_bordes_antes)
    axes[0, 0].set_title('ANTES: Bordes', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    # ANTES - Zoom de una sección
    h, w = mapa_antes.shape
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    
    axes[0, 1].imshow(img_bordes_antes[y1:y2, x1:x2])
    axes[0, 1].set_title('ANTES: Zoom (detalle)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    
    # ANTES - Solo bordes en negro
    solo_bordes_antes = np.ones_like(imagen_cuantizada) * 255
    solo_bordes_antes[bordes_antes] = [0, 0, 0]
    
    axes[0, 2].imshow(solo_bordes_antes)
    axes[0, 2].set_title('ANTES: Contornos aislados', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    
    # DESPUÉS - Bordes
    bordes_despues = find_boundaries(mapa_despues, mode='thick')
    img_bordes_despues = imagen_cuantizada.copy()
    img_bordes_despues[bordes_despues] = [0, 255, 0]  # Verde
    
    axes[1, 0].imshow(img_bordes_despues)
    axes[1, 0].set_title('DESPUÉS: Bordes Suavizados', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    
    # DESPUÉS - Zoom
    axes[1, 1].imshow(img_bordes_despues[y1:y2, x1:x2])
    axes[1, 1].set_title('DESPUÉS: Zoom (detalle)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # DESPUÉS - Solo bordes en negro
    solo_bordes_despues = np.ones_like(imagen_cuantizada) * 255
    solo_bordes_despues[bordes_despues] = [0, 0, 0]
    
    axes[1, 2].imshow(solo_bordes_despues)
    axes[1, 2].set_title('DESPUÉS: Contornos suavizados', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Comparación de complejidad de bordes
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    complejidad_antes = np.sum(bordes_antes)
    complejidad_despues = np.sum(bordes_despues)
    
    ax.bar(['Antes', 'Después'], 
           [complejidad_antes, complejidad_despues],
           color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Píxeles de borde', fontsize=12)
    ax.set_title('Complejidad de Bordes (menos = más suave)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for i, v in enumerate([complejidad_antes, complejidad_despues]):
        ax.text(i, v + complejidad_antes*0.02, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
    
    reduccion = 100 * (complejidad_antes - complejidad_despues) / complejidad_antes
    ax.text(0.5, complejidad_antes * 0.5, f'Reducción: {reduccion:.1f}%', 
            ha='center', fontsize=13, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def generar_imagen_para_pintar(mapa_regiones, df_regiones, colores_paleta, 
                                grosor_borde=1, tamano_numero='auto',
                                fondo_blanco=True):
    """
    Genera la imagen final tipo "pintar por números"
    
    Args:
        mapa_regiones: mapa de regiones final
        df_regiones: DataFrame con info de regiones
        colores_paleta: array con los 20 colores
        grosor_borde: grosor de los bordes en píxeles
        tamano_numero: tamaño de fuente ('auto' o int)
        fondo_blanco: si True, fondo blanco; si False, mantiene colores tenues
    
    Returns:
        img_para_pintar: imagen con bordes y números
        img_solucion: imagen con colores (referencia)
        paleta_numerada: imagen de la paleta con números
    """
    print(f"\n{'='*70}")
    print(f"GENERANDO IMAGEN FINAL PARA PINTAR")
    print(f"{'='*70}\n")
    
    h, w = mapa_regiones.shape
    
    # 1. Crear imagen solución (con colores)
    print("Generando imagen solución...")
    img_solucion = np.zeros((h, w, 3), dtype=np.uint8)
    
    for _, row in df_regiones.iterrows():
        region_id = row['region_id']
        color_rgb = row['color_rgb']
        mascara = (mapa_regiones == region_id)
        img_solucion[mascara] = color_rgb
    
    # 2. Crear imagen base para pintar
    print("Generando imagen base para pintar...")
    if fondo_blanco:
        img_base = np.ones((h, w, 3), dtype=np.uint8) * 255  # Blanco
    else:
        # Colores muy tenues (30% de saturación)
        img_base = (img_solucion * 0.3 + 255 * 0.7).astype(np.uint8)
    
    # 3. Detectar y dibujar contornos
    print("Detectando contornos...")
    from skimage.segmentation import find_boundaries
    
    # Encontrar todos los bordes
    bordes = find_boundaries(mapa_regiones, mode='thick')
    
    # Dilatar bordes para hacerlos más gruesos
    # if grosor_borde > 1:
    #     from scipy.ndimage import binary_dilation
    #     estructura = np.ones((grosor_borde, grosor_borde))
    #     bordes = binary_dilation(bordes, structure=estructura)
    
    # Dibujar bordes en negro
    img_base[bordes] = [0, 0, 0]
    
    # 4. Calcular centroides y colocar números
    print("Calculando centroides y colocando números...")
    
    # Convertir a PIL para dibujar texto
    img_pil = Image.fromarray(img_base)
    draw = ImageDraw.Draw(img_pil)
    
    # Determinar tamaño de fuente automáticamente si es necesario
    if tamano_numero == 'auto':
        # Basado en el área promedio de las regiones
        area_promedio = df_regiones['area_pixels'].median()
        tamano_fuente = max(12, min(60, int(np.sqrt(area_promedio) / 3)))
    else:
        tamano_fuente = tamano_numero
    
    print(f"Tamaño de fuente: {tamano_fuente}")
    
    # Intentar cargar fuente, usar default si falla
    try:
        fuente = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", tamano_fuente)
    except:
        try:
            fuente = ImageFont.truetype("arial.ttf", tamano_fuente)
        except:
            fuente = ImageFont.load_default()
            print("⚠ Usando fuente por defecto (puede verse pequeña)")
    
    # Mapear region_id a número secuencial por color
    print("Asignando números a regiones...")
    df_regiones_sorted = df_regiones.sort_values(['color_id', 'area_pixels'], ascending=[True, False])
    df_regiones_sorted['numero'] = df_regiones_sorted.groupby('color_id').cumcount() + 1
    
    # Crear diccionario region_id -> (color_id, numero_en_color)
    numeros_map = {}
    for _, row in df_regiones_sorted.iterrows():
        numeros_map[row['region_id']] = (row['color_id'], row['numero'])
    
    # Colocar números en centroides
    regiones_numeradas = 0
    
    for region_id in np.unique(mapa_regiones):
        if region_id == 0:
            continue
        
        mascara = (mapa_regiones == region_id)
        
        # Calcular centroide
        coords = np.argwhere(mascara)
        if len(coords) == 0:
            continue
        
        centroide_y = int(coords[:, 0].mean())
        centroide_x = int(coords[:, 1].mean())
        
        # Obtener número para esta región
        color_id, numero = numeros_map.get(region_id, (0, 0))
        
        # Texto a mostrar: solo el color_id (más simple)
        texto = str(color_id)
        
        # Obtener tamaño del texto
        bbox = draw.textbbox((0, 0), texto, font=fuente)
        texto_w = bbox[2] - bbox[0]
        texto_h = bbox[3] - bbox[1]
        
        # Posición centrada
        pos_x = centroide_x - texto_w // 2
        pos_y = centroide_y - texto_h // 2
        
        # Dibujar fondo blanco para el número (mejor legibilidad)
        # padding = 3
        # draw.rectangle(
        #     [pos_x - padding, pos_y - padding, 
        #      pos_x + texto_w + padding, pos_y + texto_h + padding],
        #     fill=(255, 255, 255)
        # )
        
        # Dibujar número en negro
        draw.text((pos_x, pos_y), texto, fill=(0, 0, 0), font=fuente)
        
        regiones_numeradas += 1
    
    print(f"✓ {regiones_numeradas} regiones numeradas")
    
    # Convertir de vuelta a numpy
    img_para_pintar = np.array(img_pil)
    
    # 5. Generar paleta numerada
    print("Generando paleta de colores...")
    paleta_numerada = generar_paleta_numerada(colores_paleta, df_regiones)
    
    print(f"\n{'='*70}")
    print("✓ IMAGEN PARA PINTAR COMPLETADA")
    print(f"{'='*70}\n")
    
    return img_para_pintar, img_solucion, paleta_numerada

def generar_paleta_numerada(colores_paleta, df_regiones):
    """
    Genera una imagen de paleta con cada color y su número
    """
    n_colores = len(colores_paleta)
    
    # Dimensiones
    ancho_cuadro = 150
    alto_cuadro = 80
    cols = 4
    rows = (n_colores + cols - 1) // cols
    
    ancho_total = ancho_cuadro * cols + 40
    alto_total = alto_cuadro * rows + 80
    
    # Crear imagen
    img = Image.new('RGB', (ancho_total, alto_total), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Título
    try:
        fuente_titulo = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        fuente_numero = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        fuente_rgb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        fuente_titulo = ImageFont.load_default()
        fuente_numero = ImageFont.load_default()
        fuente_rgb = ImageFont.load_default()
    
    draw.text((20, 20), "PALETA DE COLORES", fill=(0, 0, 0), font=fuente_titulo)
    
    # Dibujar cada color
    y_offset = 70
    
    for i, color in enumerate(colores_paleta):
        row = i // cols
        col = i % cols
        
        x = 20 + col * ancho_cuadro
        y = y_offset + row * alto_cuadro
        
        # Cuadro de color
        draw.rectangle([x, y, x + 60, y + 60], fill=tuple(color), outline=(0, 0, 0), width=2)
        
        # Número
        draw.text((x + 70, y + 5), f"#{i+1}", fill=(0, 0, 0), font=fuente_numero)
        
        # Valores RGB
        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
        draw.text((x + 70, y + 45), rgb_text, fill=(100, 100, 100), font=fuente_rgb)
    
    return np.array(img)

def visualizar_resultado_final(img_original, img_para_pintar, img_solucion, paleta_numerada):
    """Visualiza el resultado final completo"""
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
    
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_original)
    ax1.set_title('1. Imagen Original', fontsize=15, fontweight='bold')
    ax1.axis('off')
    
    # Para pintar
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_para_pintar)
    ax2.set_title('2. PARA PINTAR (con números)', fontsize=15, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # Solución
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(img_solucion)
    ax3.set_title('3. Solución (referencia de colores)', fontsize=15, fontweight='bold')
    ax3.axis('off')
    
    # Zoom de para pintar
    h, w = img_para_pintar.shape[:2]
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img_para_pintar[y1:y2, x1:x2])
    ax4.set_title('4. Zoom - Detalle de números', fontsize=15, fontweight='bold')
    ax4.axis('off')
    
    # Paleta
    ax5 = fig.add_subplot(gs[2, :])
    ax5.imshow(paleta_numerada)
    ax5.set_title('5. Paleta de Colores Numerada', fontsize=15, fontweight='bold')
    ax5.axis('off')
    
    plt.tight_layout()
    plt.show()

def guardar_archivos_finales(img_para_pintar, img_solucion, paleta_numerada, 
                             df_regiones, prefijo="pintar_por_numeros"):
    """Guarda todos los archivos del proyecto"""
    
    print(f"\n{'='*70}")
    print("GUARDANDO ARCHIVOS FINALES")
    print(f"{'='*70}\n")
    
    # Crear carpeta de salida
    import os
    carpeta = "output_pintar_numeros"
    os.makedirs(carpeta, exist_ok=True)
    
    # Guardar imágenes
    archivos_guardados = []
    
    # 1. Imagen para pintar
    ruta = f"{carpeta}/{prefijo}_PARA_PINTAR.png"
    Image.fromarray(img_para_pintar).save(ruta, dpi=(300, 300))
    archivos_guardados.append(ruta)
    print(f"✓ {ruta}")
    
    # 2. Solución
    ruta = f"{carpeta}/{prefijo}_solucion.png"
    Image.fromarray(img_solucion).save(ruta, dpi=(300, 300))
    archivos_guardados.append(ruta)
    print(f"✓ {ruta}")
    
    # 3. Paleta
    ruta = f"{carpeta}/{prefijo}_paleta.png"
    Image.fromarray(paleta_numerada).save(ruta, dpi=(300, 300))
    archivos_guardados.append(ruta)
    print(f"✓ {ruta}")
    
    # 4. CSV con información de regiones
    ruta = f"{carpeta}/{prefijo}_regiones_info.csv"
    df_regiones.to_csv(ruta, index=False)
    archivos_guardados.append(ruta)
    print(f"✓ {ruta}")
    
    # 5. Versión de alta resolución para imprimir (opcional)
    ruta = f"{carpeta}/{prefijo}_PARA_PINTAR_alta_res.png"
    Image.fromarray(img_para_pintar).save(ruta, dpi=(600, 600), optimize=False)
    archivos_guardados.append(ruta)
    print(f"✓ {ruta}")
    
    print(f"\n{'='*70}")
    print(f"✓ TODOS LOS ARCHIVOS GUARDADOS EN: ./{carpeta}/")
    print(f"{'='*70}\n")
    
    return archivos_guardados