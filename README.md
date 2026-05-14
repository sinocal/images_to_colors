# Images to Colors - Paint-by-Numbers Generator

Proyecto Python que convierte imágenes a formato paint-by-numbers (pintar por números).

## Estructura del Proyecto

```
images_to_colors/
├── img/                     # Imágenes fuente (PNG, JPG)
├── output/                  # Resultados intermedios
├── funciones.py            # Biblioteca de funciones del pipeline
├── ejecutar.py              # Ejecución completa del proceso
├── procesar_fotos.ipynb    # Notebook para ejecución paso a paso
├── requirements.txt
└── README.md
```

## Archivos Principales

### `funciones.py`
Biblioteca con todas las funciones organizadas por pasos:
- **Paso 1**: Cuantización de colores (K-means)
- **Paso 2**: Segmentación de regiones (connected components)
- **Paso 3**: Filtrado de regiones pequeñas
- **Paso 4**: Generación de salida paint-by-numbers

### `ejecutar.py`
Script que importa `funciones.py` y ejecuta el pipeline completo de forma automatizada.

### `procesar_fotos.ipynb`
Notebook interactivo para ejecutar el proceso paso a paso, permitiendo revisar resultados visuales entre cada etapa.

## Uso

### Opción 1: Ejecución paso a paso (recomendado para desarrollo)
1. Colocar imagen en `img/nombre_imagen.png`
2. Abrir `procesar_fotos.ipynb` en Jupyter
3. Ejecutar celdas una por una, revisando resultados
4. Confirmar OK antes de continuar al siguiente paso

### Opción 2: Ejecución completa
1. Colocar imagen en `img/`
2. Ejecutar: `python ejecutar.py`

## Metodología de Desarrollo

Desarrollo iterativo paso a paso:
- Se construye cada funcionalidad y se verifica
- El usuario da OK antes de continuar al siguiente paso
- Los pasos pueden ajustarse según resultados obtenidos

## Requisitos

```
numpy
matplotlib
scikit-learn
Pillow
scipy
pandas
scikit-image
```

## Instalación

```bash
pip install -r requirements.txt
```