# Evaluación de IA – CNN con CIFAR-10

**Código:** 020201359C  
**Semilla provisional:** 59 (dos últimos dígitos numéricos; confirmar con el docente por la letra final C).

## Objetivo y dataset
Clasificar imágenes RGB de 32 × 32 píxeles en las 10 clases de CIFAR-10 usando PyTorch. Se usa una partición estratificada de los 50 000 ejemplos de entrenamiento: 40 000 para entrenar y 10 000 para validar; los 10 000 ejemplos originales de prueba se mantienen independientes.

## Pregunta 1: datos y transformaciones
`src/dataset.py` fija la semilla en NumPy, Python y PyTorch, muestra 16 fotografías etiquetadas y aplica `ToTensor`, normalización RGB y aumento de datos (`RandomCrop` y `RandomHorizontalFlip`) **solo al entrenamiento**. La validación y la prueba no reciben aumentos aleatorios.

## Pregunta 2: CNN frente a MLP
Un MLP tradicional aplana cada imagen RGB de 32 × 32 en 3072 características, destruyendo la organización espacial explícita. Una capa densa con 1024 neuronas requeriría 3072 × 1024 + 1024 = 3 146 752 parámetros, solo en esa capa. Las convoluciones explotan **conectividad local** y **compartición de pesos**: un filtro 3 × 3 aprende patrones reutilizables en distintas posiciones. La convolución es aproximadamente equivariante a traslaciones; el pooling y el aumento de datos aportan cierta robustez espacial, no invariancia perfecta.

### Arquitectura y shapes (batch = 8)
| Etapa | Operación | Tensor de salida |
|---|---|---|
| Entrada | Imagen RGB | (8, 3, 32, 32) |
| Bloque 1 | Conv2d(3,32,3,pad=1), ReLU, MaxPool2d(2) | (8, 32, 16, 16) |
| Bloque 2 | Conv2d(32,64,3,pad=1), ReLU, MaxPool2d(2) | (8, 64, 8, 8) |
| Bloque 3 | Conv2d(64,128,3,pad=1), ReLU, MaxPool2d(2) | (8, 128, 4, 4) |
| Aplanado | Flatten | (8, 2048) |
| Densa | Linear(2048,256), ReLU, Dropout(0.3) | (8, 256) |
| Salida | Linear(256,10) | (8, 10) |

## Pregunta 3: prueba de tensores 4D
`python src/model.py` imprime los shapes de cada bloque y comprueba con `assert` que el resultado sea `(8,10)`.

## Pregunta 4: entrenamiento y análisis
Se utiliza `CrossEntropyLoss`, optimizador Adam, 10 épocas por defecto y registro de loss/accuracy de entrenamiento y validación. Se guarda el mejor modelo según pérdida de validación; la evaluación final se hace **una sola vez** sobre test. El script produce `outputs/loss_accuracy.png`, `outputs/confusion_matrix.png` y `outputs/metrics.json`.

**Análisis crítico pendiente de ejecutar:** después del entrenamiento, reemplazar este apartado con los valores reales de accuracy y loss de entrenamiento, validación y prueba. Una brecha persistente de entrenamiento mucho mejor que validación sugiere *overfitting*; resultados pobres en ambos conjuntos sugieren *underfitting*. No se deben inventar métricas.

## Ejecución
```bash
pip install -r requirements.txt
python src/dataset.py
python src/model.py
python src/train.py --epochs 10
```
Ejecutar desde la raíz del repositorio. También puede abrir `notebooks/experiment.ipynb` en Google Colab; active GPU en el entorno de ejecución para acelerar el entrenamiento. Los gráficos se generan al completar las épocas y deben subirse al repositorio.

## Entrega
Repositorio público en GitHub, con scripts, notebook ejecutado, README actualizado y gráficos reales en `outputs/`.

## Google Colab (instrucciones directas)
1. Descarga el ZIP y abre `notebooks/experiment.ipynb` en Google Colab (Archivo → Subir cuaderno).
2. Activa GPU T4 en Entorno de ejecución → Cambiar tipo de entorno de ejecución.
3. Ejecuta la primera celda de código y selecciona **el ZIP completo** cuando aparezca el selector.
4. Ejecuta las siguientes celdas en orden, espera al entrenamiento y descarga `resultados_CNN_semilla59.zip`.
5. Extrae sus imágenes y súbelas a `outputs/` de GitHub; actualiza el README con las métricas reales y sustituye el notebook por la versión ejecutada.

**Nota:** La semilla 59 usa los últimos dos dígitos numéricos de `020201359C`, no los últimos dos caracteres (`9C`). Confirma este criterio con el docente.
