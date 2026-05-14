# Development Context

## Environment
- Python 3 with conda/miniconda environment `sandbox`
- Jupyter Notebook for development

## Key Files
- `procesar_fotos.ipynb` - Main pipeline notebook (run cells 1-5 sequentially)
- `explorar_img.ipynb` - Image exploration and testing
- `k_means_colores.ipynb` - K-means color quantization standalone

## Workflow
1. Put image in `img/` directory
2. Edit `RUTA_IMAGEN` variable in notebook
3. Adjust `N_COLORES` (default: 16)
4. Run cells in order: functions → execution cells

## Important Patterns
- K-means with `random_state=42` for reproducibility
- Connected components via `scipy.ndimage.label`
- Output folder `output/` for incremental filtering results

## Data
- Source images: `img/*.png`, `img/*.jpg`
- Output: `output/*.png`, `output/*.csv`
- `output_pintar_numeros/` contains final numbered output