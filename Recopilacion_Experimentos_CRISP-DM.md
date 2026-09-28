# Recopilación exhaustiva de experimentos — Proyecto Psiquiatría (Detección de Depresión y Ansiedad desde Voz)

**Propósito de este documento:** recopilación cruda y exhaustiva de TODOS los experimentos, datasets, pipelines, hiperparámetros y resultados cuantitativos encontrados en el repositorio del proyecto, generada para servir de insumo a la redacción de un paper académico bajo metodología **CRISP-DM**. Este documento NO es el paper — es material de referencia sin interpretación editorial añadida más allá de la organización y las observaciones metodológicas que los propios artefactos del proyecto (código, nombres de carpetas, docstrings, logs) documentan explícitamente.

**Método de recopilación:** 4 agentes de investigación independientes exploraron en paralelo, con acceso de solo lectura, la totalidad de notebooks (`.ipynb`), scripts (`.py`), archivos de resultados (`.csv`/`.json`), logs de ejecución y estructura de datasets del repositorio. Ningún archivo del proyecto fue modificado. Las cifras reportadas son las que aparecen literalmente en los archivos fuente (sin redondeos adicionales salvo que el propio archivo ya viniera redondeado).

**Fecha de recopilación:** 2026-08-06
**Directorio raíz del proyecto:** `/home/ci2dt2-ai/Proyectos/Psiquiatria`

---

## Cómo está organizado este documento

1. **Esta síntesis** (roadmap por fases CRISP-DM, hallazgos transversales, tablas comparativas globales)
2. **Anexo A** — Pipeline raíz + dataset público DAIC-WOZ (`report_1_daic_root.md`)
3. **Anexo B** — Track completo de Ansiedad (`report_2_ansiedad.md`)
4. **Anexo C** — Track completo de Depresión (`report_3_depresion.md`)
5. **Anexo D** — Dataset propio "Clasificación Final" + minería completa de logs de ejecución, incluyendo tabla de 581 resultados extraídos (`report_4_dataset_logs.md`)

Cada anexo conserva las cifras exactas, tablas por configuración y rutas de archivo fuente tal como fueron extraídas. Esta síntesis resume y cruza la información entre anexos, pero **para cualquier cifra que se vaya a citar en el paper, verificar contra el anexo correspondiente**.

---

## 1. Visión general del proyecto (Business Understanding, inferido)

El proyecto aborda la **detección automática de ansiedad y depresión a partir de biomarcadores de voz**, mediante dos condiciones clínicas tratadas en paralelo con pipelines metodológicamente equivalentes:

- **Ansiedad** (`Ansiedad/`)
- **Depresión** (`Depresion/`)

Se usan **dos fuentes de datos**:
1. Un **dataset propio en español**, recolectado por los autores (`Clasificación Final/`), con grabaciones de entrevistas/cuestionarios por paciente, clasificadas en clase 0 (control) / clase 1 (caso).
2. El corpus público **DAIC-WOZ** (`DATA-DAIC/`, en inglés, protocolo de entrevista clínica semi-estructurada AVEC/DAIC), usado como (a) benchmark independiente vía fine-tuning de `wav2vec2-base`, y (b) fuente para pruebas de **generalización cross-dataset / cross-idioma** (entrenar en un corpus, evaluar en el otro).

No se encontró evidencia de un objetivo de negocio explícito documentado (no hay README ni documento de planteamiento), por lo que la fase de Business Understanding del CRISP-DM deberá formularse en el paper a partir del contexto del dominio (cribado/apoyo diagnóstico temprano de ansiedad/depresión mediante voz, no invasivo, escalable) — no hay texto fuente que lo declare literalmente.

No se encontró evidencia de una fase de **Deployment**: no hay API, interfaz, script de inferencia empaquetado ni documentación de puesta en producción. El proyecto llega hasta la fase de Evaluación (comparación de modelos/configuraciones). Esto debe declararse explícitamente como alcance/limitación en el paper.

---

## 2. Data Understanding — los dos datasets

### 2.1 Dataset propio ("Clasificación Final")

- **Tamaño oficial (nivel paciente), confirmado de forma cruzada por 3 fuentes independientes** (conteo de archivos únicos, `manifest.json` de embeddings, y logs de ejecución de los scripts de DL): **79 pacientes por condición**.
  - **Ansiedad**: 60 controles (clase 0) / 19 casos (clase 1) → ratio ≈ 3.2:1
  - **Depresión**: 58 controles (clase 0) / 21 casos (clase 1) → ratio ≈ 2.76:1
- Cadena de preprocesamiento confirmada: **grabación cruda (m4a audio-only o mp4 video)** → extracción/recorte de audio (`wav`, recorte de silencios) → denoising con **Demucs** (`--two-stems=vocals`) → (a) generación de **4 tipos de espectrograma** (CQT, Gammatone, log-Mel, PCEN) en **4 duraciones de ventana** (2s/3s/4s/10s) → ~116,200 imágenes espectrales por condición; (b) extracción de **embeddings** con **6 modelos self-supervised de habla** (`xlsr-300m`, `xlsr-53`, `whisper-large-encoder`, `wav2vec2-large-robust`, `wavlm-large`, `hubert-large`) sobre ventanas de 5s con 50% overlap → 31,512 registros/condición en `manifest.json`.
- Existen también **transcripciones de texto** (80 archivos `.txt`, entrevista semiestructurada de 7-8 preguntas temáticas, muy similar en estructura al protocolo DAIC-WOZ) en `Clasificación Final/DatosPreprocesados/Muestras_Text(corregidos)/` — **no se encontró evidencia de que estas transcripciones se hayan usado en ningún modelo** (todos los pipelines de modelado usan audio/espectrogramas/embeddings, no texto). Esto es relevante como *dato disponible pero no explotado* — posible línea futura a mencionar en el paper.
- Detalle completo, conteos por formato/clase/ventana y estructura de `manifest.json`: **Anexo D, secciones 1.1–1.5**.

### 2.2 Dataset público DAIC-WOZ

- **189 pacientes** en `DATA-DAIC/` (IDs 300–492), cada uno con audio, transcripción con timestamps, y features faciales/prosódicos oficiales del corpus (CLNF, COVAREP, FORMANT) — estos últimos **no fueron usados** por el proyecto (solo se usó el audio).
- Splits propios del proyecto (`DAIC-RESULTS/splits/`): train=84 (24 dep/60 no-dep), valid=29 (9/20), test=29 (9/20) — **distintos** de los splits oficiales AVEC2017 (train=106/dev=34/test=47) que también están presentes en el corpus.
- Embeddings extraídos con los mismos 6 modelos SSL sobre los 189 pacientes (`embeddings_daic/manifest.json`, 218,712 registros = 6×36,452 segmentos, class_0=133/class_1=56 a nivel paciente) — usados como fuente de test/train en los experimentos de generalización cross-dataset.
- Detalle completo: **Anexo A, secciones 12–17**; conteos adicionales en **Anexo D, sección 1.6**.

---

## 3. Data Preparation — resumen de pipelines de transformación

| Pipeline | Script(s) | Qué produce | Parámetros clave |
|---|---|---|---|
| Limpieza de audio | `Codigo_limpieza_de_audio.ipynb`, `batch_copy_preprocess_audio.py` | wav recortado + denoised | Demucs `htdemucs --two-stems=vocals`, trim silencio 30dB |
| Generación de espectrogramas | `gen_espectrogramas_v2.py` | PNG log-mel/CQT/gammatone/PCEN × 2s/3s/4s/10s | SR=16000, N_FFT=1024, HOP=512, N_MELS=128, overlap 0.5 (0.0 en 10s) |
| Extracción de embeddings | `embedding_pipeline_5s.py`, `embedding_pipeline.py`, `daic_embedding_pipeline*.py` | Embeddings por segmento de 5s (6 modelos SSL) | pooling `attention` aprendido, `extract_all_layers=True`, fp16 |
| Extracción de features acústicas artesanales | `pipeline_feat_ansiedad.py`, `pipeline_feat_depresion.py` | ~133-179 features/segmento (MFCC, LPCC, F0, formantes, jitter/shimmer/HNR, espectrales) | segmentos 3s/50% overlap |
| Aumento de datos sintéticos | `fase3_vae_train_generate.ipynb` (Ansiedad) | Espectrogramas sintéticos clase minoritaria vía VAE convolucional | dim_z=128, β=1.0, balanceo a ratio 1:1 |
| Splits | `fase3_split_holdout.ipynb` (Ansiedad, 63/16 pacientes), splits internos por fold en CNN2D_fase1/AST_fase2 (RepeatedStratifiedKFold) | Conjuntos train/val/test estratificados, a nivel paciente | `random_state=42` en la mayoría |

---

## 4. Modeling — mapa de los tracks experimentales

El proyecto siguió, para cada condición (Ansiedad/Depresión), **tres grandes tracks de modelado en paralelo**, más un cuarto track de validación cruzada entre corpus:

### Track A — Visión por computadora sobre espectrogramas 2D
`Fase0` (transfer learning: CNN2D propia vs EfficientNetB0 vs ResNet50 vs VGG19) → `CNN2D_fase1` (grid 4 espectrogramas × 4 duraciones, CV 5×5=25 folds) → `CNN2D_resize_exp` (ablación resize) → `AST_fase2` / `AST_fase2_unfreeze0` (Audio Spectrogram Transformer fine-tuned, mismo grid) → `fase3`/`fase3_viejo` (selección de modelo vía mini-CV + aumento VAE + evaluación final).

**Estado por condición**: el pipeline completo (Fase0→Fase1→Fase2→Fase3) sólo se completó íntegramente para **Ansiedad**. Para **Depresión** sólo se ejecutó `CNN2D_fase1` (el grid base); no hay evidencia en el repositorio de Fase0/AST_fase2/Fase3-VAE para Depresión — asimetría real de desarrollo entre condiciones, a declarar explícitamente en el paper (posiblemente por restricciones de tiempo/cómputo, no verificable con certeza desde los artefactos).

### Track B — ML clásico sobre features acústicas artesanales
EDA (`EDA_Ansiedad_Voz15.ipynb`, `EDA_Depresion_Voz15.ipynb`) → limpieza (filtro IQR + ANOVA + winsorización) → `LazyClassifier` (~30 modelos) con `GroupKFold(5)`+SMOTE agrupado por paciente → ajuste de hiperparámetros del mejor modelo (NearestCentroid en Ansiedad, BernoulliNB en Depresión).

### Track C — ML clásico + Deep Learning sobre embeddings SSL (Wav2Vec-family)
El track más extenso y con mejores resultados en ambas condiciones. Incluye:
- ML clásico (LazyClassifier + GridSearchCV) sobre embeddings pooled por paciente y por segmento (`GroupKFold`), en notebooks `ML-Wav2Vec*.ipynb`.
- DL (`dl_script1_feature_reduction.py`=MLP, `dl_script2_architectures*.py`=CNN1D/BiLSTM/BiGRU/CNN+BiGRU), con `RepeatedStratifiedKFold(5,10)`=50 evaluaciones, replicado sobre dos generaciones de embeddings ("viejos" vs "nuevos" — mismo método, distinto pipeline de extracción).
- **Sólo en Depresión**: 6 rondas sucesivas de optimización fina de BiLSTM (`dl_script2b`→`dl_script2c`→`dl_script2d`→`dl_script2e`→`dl_script2e_fixed`→`dl_script_bilstm_final`), documentando explícitamente en el código la detección y corrección progresiva de **fugas de datos (data leakage)** — ver sección 6.

### Track D — Validación cruzada entre corpus (DAIC-WOZ ↔ dataset propio)
Sólo en Depresión (`ML_Wav2Vec_DAIC_Holdout_1_.ipynb`, `ML_Wav2Vec_DAIC_merged.ipynb`, `cross_dataset_common.py`, `dl_cross_dataset_all_archs.py`, `dl_script_cross_dataset.py`). Entrena en un corpus, evalúa en el otro (y viceversa), y también un pool combinado con `StratifiedGroupKFold`.

### Track E — DAIC-WOZ standalone
Fine-tuning directo de `facebook/wav2vec2-base` (extractor congelado, cabezal+encoder entrenable) sobre segmentos de transcripción del propio DAIC-WOZ, sin pasar por embeddings pre-extraídos (`DAIC-RESULTS/daic-c2-rmse-roc.yaml`). Único experimento de fine-tuning end-to-end de un modelo de habla encontrado en el proyecto (todo lo demás usa embeddings congelados + clasificador separado).

---

## 5. Evaluation — tabla comparativa global (mejor configuración por track y condición)

### Ansiedad

| Track | Mejor configuración | F1-macro | AUC | Accuracy |
|---|---|---|---|---|
| A — Fase0 (transfer learning) | CNN2D propia, 4s | 0.543 | — | 0.666 |
| A — CNN2D_fase1 | log-mel, 2s | 0.517 | 0.575 | 0.593 |
| A — AST_fase2 | gammatone, 2s | 0.570 | 0.615 | 0.679 |
| A — fase3 (resultado final oficial, CV 5-fold) | AST_gammatone_3s | 0.564 | 0.564 | 0.618 |
| A — fase3_viejo (holdout único, descartado por poco robusto) | AST_gammatone_4s | 0.654 | 0.688 | 0.688 |
| B — ML clásico (features artesanales) | NearestCentroid (manhattan, shrink=2.0) | 0.400 (clase1) | 0.418* | 0.577 |
| C — ML clásico sobre embeddings (dl_script1, MLP) | wav2vec2-large-robust + Mixup | **0.723** | 0.668 | 0.773 |
| C — DL secuencial (dl_script2) | xlsr-300m + CNN1D + raw | 0.712–0.715 | 0.614–0.621 | 0.772–0.775 |

*AUC de NearestCentroid es un artefacto de la aproximación por distancia a centroides — no comparable directamente con AUC de modelos probabilísticos (ver Anexo B, nota 6).

**Mejor resultado global — Ansiedad: embeddings `wav2vec2-large-robust` + MLP + Mixup (Track C), F1-macro ≈ 0.72, Accuracy ≈ 0.77** — muy por encima de cualquier enfoque de imagen 2D (tope ≈ 0.57–0.65 F1-macro) o de ML clásico sobre features artesanales (F1 clase1 ≈ 0.40).

### Depresión

| Track | Mejor configuración | F1-macro | AUC | Accuracy |
|---|---|---|---|---|
| A — CNN2D_fase1 (único, no hay Fase0/AST/Fase3 para esta condición) | log-mel, 4s | 0.544 | 0.576 | 0.651 |
| B — ML clásico (features artesanales) | BernoulliNB tuneado, GroupKFold | 0.543 | 0.634 | 0.610 |
| C — ML clásico sobre embeddings, GroupKFold por segmento (más riguroso) | ExtraTrees, wav2vec2-large-robust | 0.522 | 0.580 | 0.675 |
| C — ML clásico sobre embeddings, sin agrupar por paciente (N=79, cautela — optimista) | ExtraTrees tuneado, wav2vec2-large-robust | 0.668 | 0.679 | 0.762 |
| C — DL BiLSTM, protocolo final validado ("para paper") | pooling_meanmax, wav2vec2-large-robust | 0.518–0.582 | **0.766** | 0.737 |
| D — Cross-dataset (DAIC↔propio), mejor caso | CNN_BiGRU, xlsr-300m, PCA | 0.451 | 0.570 | 0.466 |
| E — DAIC-WOZ standalone (wav2vec2-base fine-tuned) | — | 0.544 | — (RMSE=0.608) | 0.630 |
| **INVÁLIDO — bug F1-macro inflado (no citar)** | MLP, wav2vec2-large-robust, raw | ~~0.779~~ | ~~0.713~~ | ~~0.824~~ |

**Mejor resultado global fiable — Depresión: protocolo final de BiLSTM sobre embeddings `wav2vec2-large-robust`, AUC ≈ 0.77, F1-macro ≈ 0.52–0.58**, tras corregir las fugas de validación documentadas en el propio proyecto (ver sección 6). Antes de esa corrección, el proyecto había reportado F1-macro ≈ 0.78 con la misma configuración — cifra que debe considerarse **inválida**.

### DAIC-WOZ ↔ dataset propio, generalización cross-idioma

En **ninguna** combinación evaluada (Ansiedad ni Depresión, ambas direcciones, MLP o arquitecturas de secuencia) se superó AUC ≈ 0.60. Rango observado: 0.42–0.60. **Conclusión transversal: la generalización entre el corpus en inglés (DAIC-WOZ, entrevistas clínicas formales) y el dataset propio en español (grabaciones con protocolo distinto) es pobre**, lo cual es relevante como hallazgo de dominio (domain shift) para la discusión del paper.

---

## 6. Hallazgos metodológicos transversales (para la sección de Evaluación / Lecciones Aprendidas del CRISP-DM)

Estos son los hallazgos que los propios autores del proyecto documentaron (en código, nombres de carpetas o comentarios) como parte de su proceso iterativo, y que son especialmente valiosos para una sección de "lecciones aprendidas" o "limitaciones metodológicas" en el paper:

1. **Bug de F1-macro inflado (Depresión, track C, arquitecturas de secuencia)** — el hallazgo más importante del repositorio. Una generación completa de resultados (`Wav2Vec/results_DL/Resultados ANTIGUOS CON BUG F1_MACRO ALTO/`) quedó invalidada por dos fugas de datos combinadas: (a) normalización (StandardScaler/PCA) ajustada incluyendo posiciones de *padding* en secuencias de longitud variable, y (b) umbral de clasificación optimizado directamente sobre el propio conjunto de validación. El equipo corrigió esto en **6 rondas sucesivas documentadas** (`dl_script2b` → `dl_script_bilstm_final`), cuantificando el efecto: al eliminar el último resquicio de fuga (selección de checkpoint por AUC/F1 de validación en vez de loss), el AUC cayó consistentemente **~0.09 puntos** en las configuraciones re-evaluadas. Ver Anexo C, secciones 5.4 y 6.
2. **Selección de modelo inestable con dataset pequeño (Ansiedad, track A, fase3)** — el ranking de los 4 candidatos finales (CNN2D_logmel_2s, AST_gammatone_2s/3s/4s) se invirtió **tres veces** entre: (a) la mini-CV de selección (6 folds, ganador: AST_gammatone_2s), (b) la evaluación en holdout único de 16 pacientes (ganador: AST_gammatone_4s), y (c) la CV rigurosa final de 5 folds con VAE re-entrenado por fold (ganador: AST_gammatone_3s; AST_gammatone_4s pasa a ser el peor). Documentado explícitamente como motivación para reescribir el pipeline en `Final_fase3.ipynb`, que corrige 4 bugs y aísla el VAE por fold "para prevenir fuga entre conjuntos de test cruzados". Ver Anexo B, secciones 9–11.
3. **Discrepancia de fuente de datos en un notebook de EDA (Depresión)** — `EDA_Depresion_Voz15.ipynb` tiene una celda con output cacheado que corresponde al dataset de **Ansiedad** (mismo notebook reutilizado entre proyectos, sin re-ejecutar esa celda específica sobre Depresión). Las celdas posteriores sí son correctas. Ver Anexo C, sección 3.
4. **Notebooks/celdas sin ejecutar** — `Ansiedad1DServidor.ipynb` (0 outputs guardados en 65 celdas) y `Depresion1DServidor.ipynb` (0 outputs en 28 celdas) definen arquitecturas exploratorias sobre forma de onda cruda (CNN1D, RNN/GRU, BiLSTM, CRNN, Transformer) que **nunca produjeron resultados** — deben documentarse como trabajo exploratorio no concluido, no como resultados negativos.
5. **Ejecución interrumpida** — `ML_Wav2Vec_DAIC_merged.ipynb` (Depresión) se detuvo con `NameError` a mitad de proceso; sólo 2 de 6 modelos de embedding completados, sin resultados concluyentes sobre el pool combinado DAIC+propio.
6. **Decisión de diseño validada empíricamente**: ablación de *resize* de espectrogramas (Ansiedad, `CNN2D_resize_exp`) mostró diferencias &lt;0.05 en F1-macro/AUC frente a no aplicar resize → se descartó el resize en el resto del pipeline.
7. **Ablación de fine-tuning parcial vs. backbone congelado** (Ansiedad, AST): congelar todo el backbone (`unfreeze0`) fue sistemáticamente **peor** que descongelar las últimas 3 capas, contrario a la hipótesis inicial de que el descongelamiento parcial causaba overfitting.
8. **Granularidad de evaluación no siempre a nivel paciente** — en `Fase0` (Ansiedad) la validación es a nivel de imagen/segmento, no de paciente, lo que puede ser optimista por fuga de segmentos del mismo paciente entre folds; a partir de `CNN2D_fase1` en adelante sí se agrega correctamente por paciente (voto/promedio de segmentos). Aplicar el mismo cuidado al citar cifras de ML clásico "sin agrupar por paciente" (Anexo C, sección 4.1) vs. "con GroupKFold" (sección 4.2) — la diferencia es de ~0.15 puntos de F1-macro.
9. **Transcripciones de texto no explotadas** — existen 80 transcripciones completas del dataset propio (protocolo similar a DAIC-WOZ) que no se usaron en ningún modelo textual/multimodal — posible limitación/línea futura a mencionar.
10. **Higiene de nombres de archivo/carpeta** reveló procesos de trabajo: carpeta `Resultados ANTIGUOS CON BUG F1_MACRO ALTO`, sufijos `_viejo`/`_nuevo` en múltiples lugares, un archivo con ruta de Windows filtrada en su nombre — indicios de iteración manual y control de versiones informal (sin Git: el repositorio no tenía commits al momento de esta recopilación).

---

## 7. Notas de trazabilidad para quien redacte el paper

- **Los notebooks de la raíz del proyecto no pertenecen todos a la misma condición.** Verificado en código: `CNN2D_fase1_cv.ipynb` = Depresión; `CNN2D_resize_experiment.ipynb`, `fase3_split_holdout.ipynb`, `fase3_mini_cv_seleccion_modelo.ipynb`, `fase3_vae_train_generate.ipynb`, `fase3_train_final_evaluate.ipynb`, `Final_fase3.ipynb` = Ansiedad. No asumir por la ubicación del archivo.
- **"Viejo" vs "nuevo"** aparece en múltiples contextos con significados *distintos* según la carpeta — en general se refiere a una versión anterior vs. corregida/actualizada del pipeline de embeddings o del script de modelado. Verificar el significado específico en cada Anexo antes de citar.
- Las cifras de accuracy/F1/AUC en este documento están tomadas literalmente de archivos `_checkpoint.json`, `metrics_*.csv`, `summary.csv`, `*_cv_summary.json`, celdas de output de notebooks, y líneas `✅` de los logs de ejecución — no fueron recalculadas ni verificadas de forma independiente por los agentes de recopilación (salvo casos puntuales donde se cruzó una cifra entre dos fuentes distintas, indicados explícitamente en los Anexos).
- Ningún archivo del proyecto fue modificado durante esta recopilación.

---

# ANEXOS

Los siguientes 4 anexos contienen el detalle completo, con tablas por configuración, hiperparámetros exactos y cifras exactas de cada experimento.


---

## ANEXO A — Pipeline raíz del proyecto y dataset público DAIC-WOZ

# Reporte de recopilación — Notebooks raíz, scripts y DAIC-WOZ

Proyecto: `/home/ci2dt2-ai/Proyectos/Psiquiatria`.

**Nota metodológica importante:** cada notebook de la raíz fija explícitamente en su código (`CONDITION`, `SPECS_ROOT`, `RESULTS_ROOT`) a qué dataset/condición pertenece la ejecución capturada en sus outputs. Esto **no es uniforme entre notebooks**: `CNN2D_fase1_cv.ipynb` corresponde a **Depresión**, mientras que `CNN2D_resize_experiment.ipynb`, `fase3_split_holdout.ipynb`, `fase3_mini_cv_seleccion_modelo.ipynb`, `fase3_vae_train_generate.ipynb`, `fase3_train_final_evaluate.ipynb` y `Final_fase3.ipynb` corresponden a **Ansiedad**.

## 0. Resumen de archivos procesados

| Archivo | Condición detectada en código |
|---|---|
| `Codigo_limpieza_de_audio.ipynb` | Ansiedad y Depresión (inventario) + ruta externa `MUESTRAS_SAMANÁ` (ya no existe en disco) |
| `CNN2D_fase1_cv.ipynb` | **Depresión** |
| `CNN2D_resize_experiment.ipynb` | **Ansiedad** |
| `fase3_split_holdout.ipynb` | **Ansiedad** |
| `fase3_mini_cv_seleccion_modelo.ipynb` | **Ansiedad** |
| `fase3_vae_train_generate.ipynb` | **Ansiedad** |
| `fase3_train_final_evaluate.ipynb` | **Ansiedad** |
| `Final_fase3.ipynb` | **Ansiedad** |
| `gen_espectrogramas_v2.py` | `CONDITION="Depresión"` (parametrizable) |
| `embedding_pipeline_5s.py` | Genérico (CLI `--condition`) |
| `deepEmbedding.sh` | Ansiedad (orquesta scripts fuera de alcance) |
| `DAIC-RESULTS/*` | DAIC-WOZ (inglés, depresión) |
| `embeddings_daic/manifest.json` | DAIC-WOZ, condición "depression" |
| `DATA-DAIC/` | Corpus original DAIC-WOZ |
| `log_ejecucion_final.txt`, `log_general.txt` | 0 coincidencias de "DAIC"/"wav2vec2-base" |

---

## 1. `Codigo_limpieza_de_audio.ipynb`
**Ruta:** `/home/ci2dt2-ai/Proyectos/Psiquiatria/Codigo_limpieza_de_audio.ipynb`. Fase de preparación de audio (denoising), no modelado.

1. **Celda 0 (Colab):** pipeline `demucs --two-stems=vocals` + `ffmpeg -ac 1` para separar voz y convertir a mono → genera los `_denoised.wav`.
2. **Celda 1:** inventario `Clasificación Final/Ansiedad/wav/{0,1}` → **178 (clase 0) + 57 (clase 1) = 235 archivos denoised**.
3. **Celda 2:** inventario Depresión (regex sobre `*_denoised.wav`) → **78 IDs únicos** (001–080, con huecos en 014 y 066).
4. **Celda 3:** ruta externa `/home/ci2dt2-ai/Proyectos/Psiquiatria/MUESTRAS_SAMANÁ/Muestras` → **79 wavs** (esta carpeta **no existe actualmente**, confirmado con `ls`).

**Estado actual en disco** (para contraste): Ansiedad wav 0=180/1=57 (237 total); Depresión wav 0=174/1=63 (237 total) — el dataset creció ligeramente después de esta notebook.

---

## 2. `gen_espectrogramas_v2.py`
**Ruta:** `/home/ci2dt2-ai/Proyectos/Psiquiatria/gen_espectrogramas_v2.py`

- `SR=16000, N_FFT=1024, HOP_LENGTH=512, N_MELS=128, F_MIN=20, F_MAX=8000`.
- **Duraciones/overlap:** 2s(0.5), 3s(0.5), 4s(0.5), 10s(0.0).
- **4 filtros:** `log-mel` (melspectrogram power=2.0→dB), `cqt` (168 bins=7×24 bins/octava), `pcen` (gain=0.98,bias=2,power=0.5,time_constant=0.4,eps=1e-6), `gammatone` (vía `spafe`, fallback ERB con `scipy.signal.gammatone`).
- Trim silencio (`top_db=20`), descarta segmentos con RMS<1e-4. PNG colormap `inferno`, dpi=100, idempotente (`is_complete()` salta carpetas ya generadas).
- Salida: `Clasificación Final/{condición}/specs/{filtro}/{dur}s/{0|1}/*.png`.
- Ancho determinista de imagen: `int(dur*16000/512)+1` → 63px(2s), 94px(3s), 126px(4s), 313px(10s).

## 3. `embedding_pipeline_5s.py`
**Ruta:** `/home/ci2dt2-ai/Proyectos/Psiquiatria/embedding_pipeline_5s.py`

6 modelos SSL: `xlsr-300m` (facebook/wav2vec2-xls-r-300m, 24L/1024h, multilingüe), `xlsr-53` (wav2vec2-large-xlsr-53), `whisper-large-encoder` (solo encoder, 32L/1280h), `wav2vec2-large-robust`, `wavlm-large`, `hubert-large` (estos 3 últimos inglés puro). Segmentación 5s/overlap 0.5 (avance 2.5s), pooling `attention` (propio, `tanh(W·h)→softmax`), `extract_all_layers=True`, batch=4, fp16. Salida jerárquica `{condition}/{class_label}/{model}/{audio_id}/segment_n/embedding.json` + `manifest.json` acumulativo reanudable. Este script generó `embeddings_daic/manifest.json`.

## 4. `deepEmbedding.sh`
Orquestador que ejecuta en secuencia `Ansiedad/Scripts_Embedding_viejo/dl_script2_architectures_fixed.py` y luego `Ansiedad/Scripts_Embedding_nuevo/dl_script1_feature_reduction.py` + `dl_script2_architectures_fixed.py` (scripts fuera del alcance asignado, no leídos).

---

## 5. `CNN2D_fase1_cv.ipynb` — CNN-2D propia, Fase 1, CV (Depresión)
**Fase CRISP-DM:** Modelado+Evaluación baseline. `CONDITION="Depresión"`, `SPECS_ROOT="Clasificación Final/Depresión/specs"`, `RESULTS_ROOT="Depresion/CNN2D_fase1"`. GPU: RTX 5080.

**Dataset:** 79 pacientes en las 16 configs (58 sin / 21 con depresión, ratio 2.76:1). Segmentos (ambas clases): 10s=1,259; 2s=12,863; 3s=8,544; 4s=6,383.

**Arquitectura CNN2D:** 5 bloques conv (3→32×2, 32→64×2, 64→128×3, 128→256×3, 256→256×3; Conv3x3+BN+ReLU, MaxPool2d(2)+Dropout2d(0.2) por bloque) + AdaptiveAvgPool2d(1) + clasificador Linear(256,128)+Dropout(0.5)→Linear(128,64)+Dropout(0.4)→Linear(64,1)+Sigmoid. **3,725,601 parámetros entrenables.** Adam lr=1e-4, BCELoss ponderada por class_weight balanced, ReduceLROnPlateau(0.5,pat=3), batch=128, epochs=30(máx), patience=5. Umbral: barrido 0.05–0.95 step 0.02 maximizando F1-macro en val interna 15%. Sin resize (ancho nativo). Normalización ImageNet + ColorJitter(0.1,0.1) en train.

**Validación:** `RepeatedStratifiedKFold(5,5)=25 folds` × 16 configs (4 filtros × 4 duraciones), random_state=42. Agregación por paciente = media de probabilidades de sus segmentos.

**Resultados EXACTOS (media±std sobre 25 folds; Thr/Acc/Prec/Rec/F1/F1-Macro/AUC):**

| Filtro-Dur | Thr | Accuracy | Precision | Recall | F1 | F1-Macro | AUC |
|---|---|---|---|---|---|---|---|
| log-mel 10s | 0.58±0.11 | 0.633±0.125 | 0.147±0.185 | 0.246±0.303 | 0.180±0.220 | 0.457±0.100 | 0.517±0.143 |
| log-mel 2s | 0.35±0.17 | 0.628±0.130 | 0.297±0.276 | 0.394±0.323 | 0.307±0.240 | 0.512±0.138 | 0.587±0.148 |
| log-mel 3s | 0.40±0.19 | 0.623±0.119 | 0.372±0.287 | 0.364±0.241 | 0.315±0.165 | 0.518±0.102 | 0.572±0.167 |
| log-mel 4s | 0.44±0.22 | 0.651±0.129 | 0.389±0.245 | 0.384±0.251 | 0.346±0.183 | **0.544±0.121** | 0.576±0.163 |
| cqt 10s | 0.55±0.09 | 0.638±0.135 | 0.184±0.245 | 0.216±0.264 | 0.173±0.196 | 0.458±0.102 | 0.575±0.125 |
| cqt 2s | 0.40±0.23 | 0.586±0.142 | 0.274±0.184 | 0.388±0.281 | 0.293±0.179 | 0.480±0.110 | 0.531±0.134 |
| cqt 3s | 0.41±0.20 | 0.606±0.135 | 0.382±0.260 | 0.390±0.201 | 0.335±0.121 | 0.517±0.097 | 0.516±0.119 |
| cqt 4s | 0.43±0.22 | 0.591±0.127 | 0.246±0.155 | 0.352±0.282 | 0.267±0.162 | 0.469±0.092 | 0.540±0.148 |
| pcen 10s | 0.53±0.03 | 0.698±0.093 | 0.034±0.092 | 0.080±0.232 | 0.047±0.130 | 0.426±0.027 | 0.523±0.136 |
| pcen 2s | 0.54±0.13 | 0.636±0.095 | 0.260±0.193 | 0.280±0.240 | 0.250±0.183 | 0.499±0.098 | 0.586±0.140 |
| pcen 3s | 0.55±0.13 | 0.633±0.125 | 0.255±0.202 | 0.328±0.306 | 0.269±0.218 | 0.502±0.124 | 0.550±0.160 |
| pcen 4s | 0.52±0.17 | 0.576±0.150 | 0.196±0.162 | 0.328±0.324 | 0.223±0.181 | 0.439±0.101 | 0.539±0.128 |
| gammatone 10s | 0.56±0.07 | 0.653±0.149 | 0.080±0.149 | 0.122±0.220 | 0.089±0.158 | 0.421±0.089 | 0.503±0.111 |
| gammatone 2s | 0.37±0.17 | 0.625±0.139 | 0.314±0.262 | 0.418±0.322 | 0.327±0.236 | 0.519±0.147 | **0.596±0.173** |
| gammatone 3s | 0.42±0.19 | 0.595±0.140 | 0.288±0.277 | 0.342±0.274 | 0.270±0.197 | 0.482±0.122 | 0.573±0.152 |
| gammatone 4s | 0.38±0.16 | 0.578±0.133 | 0.221±0.189 | 0.330±0.276 | 0.254±0.205 | 0.468±0.124 | 0.575±0.169 |

Mejor F1-Macro: **log-mel 4s (0.544)**. Mejor AUC: **gammatone 2s (0.596)**. Todos en rango 0.42–0.54 F1-Macro, cercano al azar en varios casos (recall muy bajo en 10s).

**Observaciones:** la función de gráficos (celda 8) tiene hard-codeado "Sin/Con Ansiedad" pese a ser el run de Depresión; el output final de celda 9 es residual de una ejecución previa apuntando a `Ansiedad/CNN2D_fase1/pcen/4s` con código comentado — evidencia de que el mismo notebook también corrió sobre Ansiedad en algún momento.

---

## 6. `CNN2D_resize_experiment.ipynb` — Resize vs No-Resize (Ansiedad)
`CONDITION="Ansiedad"`, `RESULTS_ROOT="Ansiedad/CNN2D_resize_exp"`. Objetivo: decidir si `T.Resize(H,128)` mejora vs ancho nativo. Mismos hiperparámetros/arquitectura/CV que sección 5 (25 folds). 4 "mejores configs" de una Fase1-Ansiedad previa: log-mel 2s, cqt 4s, pcen 2s, gammatone 3s. **79 pacientes (60 sin/19 con ansiedad).**

| Config | Métrica clave | No-Resize | Resize | Delta | Veredicto |
|---|---|---|---|---|---|
| log-mel 2s | F1-Macro / AUC | 0.517±0.121 / 0.575±0.158 | 0.511±0.126 / 0.564±0.126 | -0.006 / -0.011 | SIMILAR |
| cqt 4s | F1-Macro / AUC | 0.511±0.096 / 0.554±0.166 | 0.477±0.115 / 0.550±0.128 | -0.033 / -0.004 | SIMILAR |
| pcen 2s | F1-Macro / AUC | 0.442±0.067 / 0.471±0.146 | 0.462±0.086 / 0.501±0.143 | +0.020 / +0.031 | SIMILAR |
| gammatone 3s | F1-Macro / AUC | 0.489±0.107 / 0.584±0.127 | 0.492±0.116 / 0.571±0.143 | +0.004 / -0.013 | SIMILAR |

**Decisión de diseño documentada:** ninguna diferencia supera el umbral 0.05 → **se descarta el resize**, se mantiene ancho nativo en toda la Fase 1/3.

---

## 7. `fase3_split_holdout.ipynb` — Split holdout 80/20 (Ansiedad)
`StratifiedShuffleSplit(test_size=0.20, random_state=42)` a nivel paciente, inmutable (JSON por filtro×duración). **Train: 63 pac (48/15, ratio 3.20:1). Test: 16 pac (12/4).** Verificación de no-fuga: OK en las 16 combinaciones. Consistencia entre duraciones del mismo filtro: OK.

| Duración | Train C0/C1 segs | Sintéticos necesarios (ratio 1:1) | Test C0/C1 segs |
|---|---|---|---|
| 2s | 7,665 / 2,783 | 4,882 | 1,927 / 491 |
| 3s | 5,090 / 1,847 | 3,243 | 1,281 / 327 |
| 4s | 3,802 / 1,381 | 2,421 | 957 / 243 |
| 10s | 751 / 272 | 479 | 189 / 47 |

---

## 8. `fase3_mini_cv_seleccion_modelo.ipynb` — Selección de modelo (Ansiedad)
`RepeatedStratifiedKFold(3,2)=6 folds` sobre solo los 63 pacientes de train (test nunca tocado). AST = `MIT/ast-finetuned-audioset-10-10-0.4593` (hidden=768, 12L/12H, patch=16, stride=10), solo 3 últimas capas + layernorm + clasificador nuevo descongelados; AdamW lr=2e-5, batch=32, epochs=20, patience=4. CNN2D idéntica a sección 5.

**Discrepancia detectada:** `CANDIDATOS` en código define 4 elementos (incl. `AST_gammatone_3s`), pero el output ejecutado y persistido solo procesa 3 (falta `AST_gammatone_3s`) — probable edición post-ejecución sin re-correr.

| Candidato | F1-Macro | AUC | Accuracy | Recall |
|---|---|---|---|---|
| CNN2D_logmel_2s | 0.432±0.089 | 0.458±0.090 | 0.508±0.162 | 0.533±0.320 |
| AST_gammatone_2s | 0.518±0.071 | 0.592±0.051 | 0.587±0.131 | 0.533±0.221 |
| AST_gammatone_4s | **0.529±0.067** | **0.596±0.055** | **0.611±0.101** | 0.500±0.252 |

**Ganador: `AST_gammatone_4s`** → guardado en `modelo_ganador.json`.

**Nota de reproducibilidad:** un archivo alterno (`Ansiedad/fase3_viejo/mini_cv/todos_candidatos.json`, de una corrida anterior) da resultados distintos con los 4 candidatos completos (CNN2D_logmel_2s f1mac=0.4290, AST_gammatone_2s=0.5257, AST_gammatone_3s=0.5204, AST_gammatone_4s=0.4980) — en esa corrida el ganador habría sido `AST_gammatone_2s`, no `AST_gammatone_4s`. Evidencia de iteración del pipeline entre `fase3_viejo/` y `fase3/`.

---

## 9. `fase3_vae_train_generate.ipynb` — VAE de augmentation (Ansiedad)
VAE convolucional (Encoder Conv 32→64→128→256 stride2 + FC μ/logσ²; Decoder simétrico ConvTranspose; `dim_z=128, β=1.0`), entrenado sobre clase 1 (con ansiedad) de train, 200 épocas máx, batch=64, lr=1e-3 (Adam), patience=20, seed=42. Procesa los 4 candidatos.

| Candidato | Imgs train C1 | Tamaño interno | Parámetros | Mejor loss | Sintéticos generados | Ratio final |
|---|---|---|---|---|---|---|
| CNN2D_logmel_2s | 2,783 | 128×64 | 4,533,505 | 4,665.48 | 4,882 | 1.00:1 |
| AST_gammatone_2s | 2,783 | 128×64 | 4,533,505 | 4,883.02 | 4,882 | 1.00:1 |
| AST_gammatone_3s | 1,847 | 128×96 | 6,110,465 | 7,317.08 | 3,243 | 1.00:1 |
| AST_gammatone_4s | 1,381 | 128×128 | 7,687,425 | 9,834.68 | 2,421 | 1.00:1 |

---

## 10. `fase3_train_final_evaluate.ipynb` — Entrenamiento final + test holdout (Ansiedad)
Entrena y evalúa los 4 candidatos con train real+sintético (fix documentado: class_weight a nivel de segmento). Val interna 10 pac (8/2), train real 53 pac (40/13) en todos los candidatos.

**Overfitting detectado:** los 3 AST llegan a loss de entrenamiento ≈0.0000 (ej. AST_gammatone_2s: 0.3717→0.0000 en 20 épocas), fuerte señal de sobreajuste tras fine-tuning parcial sobre train pequeño con >1/3 de clase 1 sintética. CNN2D no muestra este patrón (loss final 0.0543).

**Resultados EXACTOS en test holdout (16 pacientes nunca antes tocados):**

| Candidato | Accuracy | F1-Macro | AUC | Threshold | ΔF1-Mac vs mini-CV | ΔAUC vs mini-CV |
|---|---|---|---|---|---|---|
| CNN2D_logmel_2s | 0.7500 | 0.4286 | 0.7917 | 0.73 | -0.000 | +0.350 |
| AST_gammatone_2s | 0.5625 | 0.5466 | 0.7708 | 0.15 | +0.021 | +0.152 |
| AST_gammatone_3s | 0.5000 | 0.4921 | 0.7292 | 0.13 | -0.028 | +0.119 |
| AST_gammatone_4s | **0.6875** | **0.6537** | 0.6875 | 0.15 | **+0.156** | +0.106 |

Mejor resultado: `AST_gammatone_4s` (consistente con ser el ganador de mini-CV).

---

## 11. `Final_fase3.ipynb` — Pipeline consolidado y corregido, 5-fold CV externo (Ansiedad)
Reescritura completa que corrige 4 bugs documentados explícitamente en el código: (1) manejo de `return_stem` en inferencia; (2) `target_w` calculado antes del dataset AST; **(3) el más importante — VAE re-entrenado y aislado por fold** (`fold_XX/1_sintetico/`) "para prevenir fuga entre conjuntos de test cruzados" (los notebooks 3-4 entrenaban el VAE una sola vez sobre el split fijo); (4) cálculo correcto de segmentos reales para `n_synth`.

**Validación:** `StratifiedKFold(5, shuffle=True, random_state=42)` externo sobre todos los pacientes disponibles (no reutiliza el split holdout fijo) + `StratifiedShuffleSplit(0.20)` interno para threshold. VAE re-entrenado desde cero en cada fold.

**Resultados EXACTOS — 5-fold CV externo:**

| Candidato | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | **Media F1-Macro** | **Media AUC** |
|---|---|---|---|---|---|---|---|
| CNN2D_logmel_2s | F1=0.429/AUC=0.333 | F1=0.667/AUC=0.604 | F1=0.385/AUC=0.604 | F1=0.600/AUC=0.708 | F1=0.550/AUC=0.500 | 0.526±0.105 | 0.550±0.127 |
| AST_gammatone_2s | 0.417/0.396 | 0.619/0.771 | 0.375/0.562 | 0.287/0.583 | 0.330/0.444 | 0.406±0.115 | 0.551±0.130 |
| AST_gammatone_3s | 0.492/0.458 | 0.667/0.750 | 0.654/0.625 | 0.543/0.542 | 0.464/0.444 | **0.564±0.083** | 0.564±0.113 |
| AST_gammatone_4s | 0.310/0.292 | 0.417/0.458 | 0.590/0.667 | 0.200/0.604 | 0.167/0.667 | 0.337±0.154 | 0.537±0.145 |

Verificado cruzando con `Ansiedad/fase3/resultado_final_cv/AST_gammatone_3s_cv_summary.json`: `f1_macro=0.5639±0.0826, auc=0.5639±0.1134` — coincide exactamente.

**Hallazgo clave para el paper:** con este esquema más riguroso, `AST_gammatone_4s` pasa de "mejor" en el holdout único (F1-Macro=0.654, sección 10) a **"peor" de los 4** (F1-Macro=0.337±0.154), mientras `AST_gammatone_3s` se vuelve el mejor (0.564±0.083). Sugiere que el resultado del notebook 4 sobre un único holdout de 16 pacientes fue optimista/inestable (alta varianza con n pequeño).

---

## 12. DAIC-WOZ — Configuración (`daic-c2-rmse-roc.yaml`)
Confirmado y ampliado:
```
seed: 103
processor_name_or_path: facebook/wav2vec2-base
pooling_mode: mean
freeze_feature_extractor: True
use_transcript_segmentation: True
segment_group_size: 5
segment_keep_remainder: True
segment_min_remainder_duration: 0.0
label_column: class_2
num_train_epochs: 10
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
learning_rate: 1e-5
fp16: False
save_steps/eval_steps/logging_steps: 10
save_total_limit: 2
test_corpora: null
```
Extractor CNN de bajo nivel congelado, solo se entrena encoder transformer + cabezal. Pooling `mean` (no attention, a diferencia de `embedding_pipeline_5s.py`). Segmentación por grupos de 5 turnos de transcripción (no ventana fija de tiempo). Semilla no estándar (103).

## 13. DAIC-WOZ — Resultados (`clsf_report.csv`, `conf_matrix.csv`)
Verificado exacto (sin redondear):
- dep: precision=0.48148148148148145, recall=0.2708333333333333, f1=0.3466666666666667, support=336
- ndep: precision=0.6680216802168022, recall=0.8341793570219966, f1=0.7419112114371708, support=591
- **accuracy=0.6299892125134844**
- macro avg: precision=0.5747515808491418, recall=0.552506345177665, **f1=0.5442889390519188**
- weighted avg: precision=0.6004084042997927, recall=0.6299892125134844, f1=0.5986510528148521
- **MSE=0.37001078748651567, RMSE=0.6082851202244846** (idéntico en las 5 filas — es un valor global "propagado", no por clase; probablemente ligado a un objetivo de regresión auxiliar dado el nombre del yaml, no confirmable solo con estos archivos).

Matriz de confusión: dep/dep=91, dep/ndep=245 (total fila=336 ✓); ndep/dep=98, ndep/ndep=493 (total fila=591 ✓). Total=927, coincide exacto con `test_dataset.num_examples=927`.

**Granularidad:** estas 927 muestras son **segmentos de transcripción** (no pacientes), provenientes de los 29 pacientes de test.

## 14. DAIC-WOZ — Splits (`DAIC-RESULTS/splits/*.csv`)
A nivel paciente (`name, path, class_2, class_4, score`): **train=84 (24 dep/60 ndep), valid=29 (9/20), test=29 (9/20)**. Total=142 de los 189 pacientes disponibles en `DATA-DAIC/` (47 no incluidos). `class_4` = severidad (non/mild/moderate/severe), `score` = PHQ numérico.

## 15. DAIC-WOZ — `DAIC-RESULTS/features/`
- `segmented_g5_keep_min0.000_class_2/` (formato HF `datasets`, ya tokenizado con `input_values`): train_dataset=2,877 ejemplos, eval_dataset=974, test_dataset=927.
- `transcript_segments_g5_keep_min0.000_class_2/` (WAV crudo segmentado + `manifest.tsv`): train=2,877 filas, validation=974, test=927 — coincide exacto. Confirma ~34.3 segmentos/paciente en train, ~33.6 en valid, ~32.0 en test.

## 16. `embeddings_daic/manifest.json`
**218,712 registros = 6 modelos × 36,452 segmentos c/u** (idéntico entre modelos). **189 audio_id únicos** (coincide con carpetas `*_P` de DATA-DAIC): class_0=133, class_1=56 (ratio 2.38:1 a nivel de todo DAIC-WOZ). Segmentación 5s/50% overlap, ~192.9 segmentos/paciente. No hay evidencia en este alcance de que estos embeddings se hayan usado para entrenar un clasificador downstream sobre DAIC-WOZ.

## 17. `DATA-DAIC/` — Estructura
195 entradas: **189 carpetas `{id}_P`** (AUDIO.wav, CLNF_AUs/features/features3D/gaze/pose.txt, CLNF_hog.txt o .bin, COVAREP.csv, FORMANT.csv, TRANSCRIPT.csv) + 6 no-paciente (3 CSVs de splits oficiales AVEC2017: train=106, dev=34, test=47/46 filas; `documents/` con 8 PDFs de referencia; `util/` con 2 scripts MATLAB para HOG). **124 GB en disco.** Los splits oficiales AVEC2017 (106/34/47) son distintos de los splits propios del proyecto (84/29/29).

## 18. Logs raíz
`grep -i "DAIC\|wav2vec2-base"` sobre `log_ejecucion_final.txt` y `log_general.txt`: **0 coincidencias en ambos**.

## 19. Observaciones transversales clave
1. Condición por notebook confirmada en código (no asumida): Depresión solo en `CNN2D_fase1_cv.ipynb`; todo lo demás de Fase 3 es Ansiedad.
2. Dos generaciones de resultados en disco: `Ansiedad/fase3_viejo/` (outputs de notebooks 1-4) vs `Ansiedad/fase3/` (solo contiene `resultado_final_cv/` y `sinteticos/`, es decir, outputs de `Final_fase3.ipynb`) — sugiere que `Final_fase3.ipynb` es la versión depurada/final.
3. Sobreajuste severo en AST tras VAE-augmentation (loss→0) en el notebook 4.
4. El hallazgo metodológico más relevante: el esquema riguroso de 5-fold CV con VAE aislado por fold invierte el ranking de candidatos respecto al holdout único — importante para la sección de limitaciones/reproducibilidad del paper.
5. Decisión de diseño documentada: no usar resize en CNN2D (diferencias <0.05 en F1-Mac/AUC).
6. DAIC-WOZ es un experimento independiente (fine-tuning directo de wav2vec2-base, no CNN2D/AST sobre imágenes-espectrograma).
7. Discrepancia código-vs-output en `fase3_mini_cv_seleccion_modelo.ipynb` (4 candidatos definidos, solo 3 ejecutados).


---

## ANEXO B — Track completo: Ansiedad

# Reporte exhaustivo — Proyecto ANSIEDAD (detección desde voz)

Directorio raíz analizado: `/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad`
Dataset fuente: `/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad`

Todas las cifras se extrajeron directamente de archivos fuente (`.ipynb`, `.py`, `.csv`, `.json`) — sin modificar ningún archivo del proyecto.

## 0. Dataset base

- `wav/0` = 180 archivos, `wav/1` = 57 archivos (nivel archivo).
- A nivel de **paciente único** (`fase3_viejo/splits/*.json`, idéntico en las 16 combinaciones filtro×duración): **79 pacientes** = **60 clase 0** + **19 clase 1** (≈24.1% positivos). Split fijo (`random_state=42`, `test_size=0.2`): 63 train (48/15) + 16 test (12/4). Este split de 79 reaparece en `CNN2D_fase1`, `AST_fase2`, `CNN2D_resize_exp` (folds ~16 pacientes) y en Wav2Vec (`n_train_real=63`, `n_val=16`).
- `MachineLearning/features_ansiedad_15.csv`: 1753 filas (segmentos 3s/50% overlap) × 136 columnas. Versión limpia `features_ansiedad_15_clean.csv`: 1639×75.

## 1. `Ansiedad1DServidor.ipynb` — SIN EJECUTAR (0 outputs)

Verificado programáticamente: 65 celdas, **0 outputs guardados** — nunca se ejecutó (o se limpiaron outputs). No aporta cifras. Define 7 familias de arquitectura × 4 duraciones (28 bloques) sobre raw waveform/MFCC: CNN1D-VGG19-style, RNN-GRU, BiLSTM, CRNN, LSTM+MFCC(n_mfcc=40), Transformer-raw, Transformer-MFCC. Todas con SR=16000, `StratifiedKFold(5)` no repetido, batch=128, epochs=25, Adam(lr=1e-4). `main.py` (raíz) es casi idéntico al bloque BiLSTM-10s de este notebook (mismas clases) salvo SR=22050 vs 16000; probablemente generó los `confusion_matrix.png`/`roc_curve.png` de la raíz, pero sin log de cifras en texto disponible.

## 2. Scripts raíz

- **`main.py`**: BiLSTM sobre waveform crudo, SR=22050, 10s sin overlap, `StratifiedKFold(5)`, batch=64, epochs=25. CNN subsampler + BiLSTM(128)→BiLSTM(64) + MLP. Sin log de métricas en texto.
- **`embedding_pipeline.py`**: pipeline de embeddings SSL — registro de 6 modelos HF (`xlsr-300m`, `xlsr-53`, `whisper-large-encoder` [multilingües, prioridad 1-3], `wav2vec2-large-robust`, `wavlm-large`, `hubert-large` [inglés, baseline]). Segmentación 5s/50% overlap, pooling `attention` (aprendido) por defecto, fp16. Salida "v2" (1 JSON/segmento) en `embeddings_v2/` + `manifest.json`.

## 3. `MachineLearning/` — EDA + ML clásico

**Pipeline de features** (`pipeline_feat_ansiedad.py`): 133 features acústicos (MFCC+Δ+ΔΔ, LPCC, ZCR/ZCR-zPSD, F0/YIN, formantes F1-F3+jitter+shimmer+HNR vía Parselmouth, espectrales, RMS, temporales). Segmentos 3s/50% overlap.

**EDA_Ansiedad_Voz15.ipynb (sí ejecutado)**: 1753→1752 filas (1 NaN eliminado), 0 duplicados, label 1307/445 (segmento). ANOVA: features base (`_mean`) mayormente significativas (ej. `mfcc1_mean` F=11.13 p=0.001), deltas mayormente no significativos. Correlaciones máximas con target ≈0.15-0.16 (débiles). Limpieza avanzada → `features_ansiedad_15_clean.csv`: filtro IQR multivariado elimina 114 filas, elimina 61 columnas no significativas, winsoriza 66 columnas → **1639×75**. Importancia de features (RF/XGBoost/Permutation) **comentada, no ejecutada**.

**Modelado (GroupKFold(5)+SMOTE en train, 79 audios únicos, label 1221/418)**: LazyClassifier (~30 modelos) → top-5: **NearestCentroid 0.5053±0.0731** (f1_macro), BernoulliNB 0.5007±0.0893, ExtraTreeClassifier 0.4803±0.0557, GaussianNB 0.4798±0.0509, AdaBoost 0.4788±0.0228. Grid search NearestCentroid (`metric×shrink_threshold`, 15 combos): mejor = `{'metric':'manhattan','shrink_threshold':2.0}`. **Resultado final**: Accuracy 0.5770±0.1228, Precision 0.3198±0.1234, Recall 0.5430±0.1536, F1(clase1) 0.3996±0.1389, Especificidad 0.5859±0.1241, **AUC-ROC 0.4179±0.1753 (peor que azar** — AUC aproximado vía distancia a centroides, poco fiable).

## 4. `Fase0/` — Baseline transfer learning (log-mel, sí ejecutado)

79 audios → imágenes por duración: 10s=1813, 2s=18369, 3s=12202, 4s=9127. Validación: Repeated Stratified 5-Fold×5reps=25 folds **a nivel de imagen** (split por archivo original para evitar leakage). 4 arquitecturas × 4 duraciones:

| Modelo | Dur | Acc OOF | F1-macro OOF |
|---|---|---|---|
| CNN2D | 10s/2s/3s/4s | 0.572/0.679/0.668/0.666 | 0.511/0.541/0.541/**0.543** |
| ResNet50 | 10s/2s/3s/4s | **0.727**/0.701/0.709/0.722 | 0.430/0.463/0.439/0.460 |
| VGG19 | 10s/2s/3s/4s | 0.709/0.706/0.699/0.706 | 0.467/0.485/0.442/0.479 |
| EfficientNetB0 | 10s/2s/3s/4s | 0.681/0.681/0.699/0.714 | 0.482/0.440/0.449/0.489 |

CNN2D propia supera claramente a los 3 backbones transfer-learning en F1-macro; ResNet50/VGG19/EfficientNet muestran colapso hacia clase mayoritaria (precision/recall clase1 casi nulos pese a accuracy alta). **Anomalía**: `Fase0/CNN2D/log-mel/10s/_checkpoint.json` es un checkpoint suelto estilo CNN2D_fase1 (folds_done=5), aparentemente mal ubicado.

## 5. `CNN2D_fase1/` — Grid CNN2D × 4 espectrogramas × 4 duraciones (nivel paciente, 25 folds)

| Filtro | 2s F1mac | 3s F1mac | 4s F1mac | 10s F1mac |
|---|---|---|---|---|
| cqt | 0.5027 | 0.5025 | 0.5105 | 0.4684 |
| gammatone | 0.4767 | 0.4888 | 0.5040 | 0.4199 |
| log-mel | **0.5173** | 0.4936 | 0.4807 | 0.4341 |
| pcen | 0.4423 | 0.4373 | 0.4378 | 0.4361 |

Mejor: **log-mel 2s (F1mac 0.5173)**. 10s sistemáticamente el peor (colapso a clase mayoritaria). AUC máximo: gammatone 3s (0.5844).

## 6. `CNN2D_resize_exp/` — Ablación resize (4 combos, mejores de cada filtro)

| Combo | F1mac resize | F1mac sin-resize |
|---|---|---|
| cqt 4s | 0.4773 | 0.5105 |
| gammatone 3s | 0.4924 | 0.4888 |
| log-mel 2s | 0.5109 | 0.5173 |
| pcen 2s | 0.4618 | 0.4423 |

Sin ventaja consistente del resize — diferencias dentro del rango de std entre folds.

## 7. `AST_fase2/` — AST fine-tuned (N_UNFREEZE=3), 4×4 grid, 10 folds

Backbone `MIT/ast-finetuned-audioset-10-10-0.4593`, positional embeddings interpolados bilinealmente, últimas 3 capas+layernorm+head entrenables (85.7M total, 21.46M entrenables ≈25%). LR=2e-5, batch=32, epochs=20.

| Filtro | 2s F1mac/AUC | 3s F1mac/AUC | 4s F1mac/AUC | 10s F1mac/AUC |
|---|---|---|---|---|
| cqt | 0.489/0.549 | 0.461/0.485 | 0.474/0.533 | 0.393/0.467 |
| gammatone | **0.570/0.615** | 0.494/**0.636** | 0.531/0.590 | 0.490/0.544 |
| log-mel | 0.437/0.485 | 0.465/0.485 | 0.407/0.465 | 0.442/0.456 |
| pcen | 0.434/0.423 | 0.394/0.414 | 0.420/0.414 | 0.404/0.442 |

**Mejor combo de todo el grid de espectrogramas: gammatone 2s (F1mac 0.5702, AUC 0.6146)**. gammatone domina claramente sobre los otros 3 filtros.

## 8. `AST_fase2_unfreeze0/gammatone/` — Ablación backbone congelado

Motivación explícita (markdown del notebook): "N_UNFREEZE=3 generó overfitting severo". Prueba N_UNFREEZE=0 (solo cabeza), LR=1e-4, epochs=30.

| Dur | F1mac uf=0 | F1mac uf=3 |
|---|---|---|
| 2s | 0.483 | **0.570** |
| 3s | 0.469 | 0.494 |
| 4s | 0.510 | **0.531** |
| 10s | 0.462 | 0.490 |

**Contrario a la hipótesis: congelar todo NO mejora** — uf=3 (fine-tuning parcial) sigue siendo mejor en las 4 duraciones.

## 9. `fase3_viejo/` — Selección de modelo + VAE + evaluación (descartada)

**Mini-CV (6 folds)**: candidatos y ganador declarado:

| Candidato | f1mac_mean | auc_mean |
|---|---|---|
| **AST_gammatone_2s (ganador)** | **0.5257** | **0.6188** |
| AST_gammatone_3s | 0.5204 | 0.6104 |
| AST_gammatone_4s | 0.4980 | 0.5813 |
| CNN2D_logmel_2s | 0.4290 | 0.4417 |

**Splits**: 16 JSON (uno por filtro×duración), todos con el mismo split de 79 pacientes (63/16, ver sección 0).

**VAE + sintéticos**: `fase3_viejo/vae/<candidato>/` (checkpoint, reconstrucciones, preview), `fase3_viejo/sinteticos/{gammatone,log-mel}/{2s,3s,4s}/` con 2421-4882 PNGs sintéticos (split único, sin fold).

**Evaluación final (holdout único, 16 pacientes, evaluado UNA vez)**:

| Candidato | Acc test | F1mac test | AUC test |
|---|---|---|---|
| AST_gammatone_2s | 0.5625 | 0.5466 | 0.7708 |
| AST_gammatone_3s | 0.5000 | 0.4921 | 0.7292 |
| **AST_gammatone_4s** | **0.6875** | **0.6537** | 0.6875 |
| CNN2D_logmel_2s | 0.7500 | 0.4286 | 0.7917 |

CNN2D_logmel_2s colapsa completamente en test (predice solo clase 0, precision/recall/f1 clase1=0) pese a AUC más alto — threshold mal calibrado. AST_gammatone_4s da el mejor F1-macro de test, **superando al "ganador oficial" de la mini-CV** — señal de que la selección con 6 folds tuvo ruido considerable.

## 10. `fase3/` — Pipeline final (nuevo), CV completa post-VAE

Diferencia clave: 5-fold CV completo (no holdout único) + ~4× más sintéticos VAE, organizados por fold (20420 gammatone-2s vs 4882 en viejo).

| Candidato | Acc | F1-macro | AUC |
|---|---|---|---|
| CNN2D_logmel_2s | **0.6700±0.066** | 0.5260±0.105 | 0.5500±0.127 |
| AST_gammatone_2s | 0.4167±0.113 | 0.4058±0.115 | 0.5514±0.130 |
| **AST_gammatone_3s** | 0.6183±0.113 | **0.5639±0.083** | **0.5639±0.113** |
| AST_gammatone_4s | 0.3900±0.197 | 0.3366±0.154 | 0.5375±0.145 |

**Resultado final "oficial" del proyecto: AST_gammatone_3s (F1-macro 0.5639)**. AST_gammatone_2s (ganador de la mini-CV) es ahora el peor candidato — el ranking se invirtió por tercera vez entre mini-CV, holdout viejo y CV nuevo, reforzando la falta de robustez de la selección de modelo con este dataset tan pequeño.

## 11. `Wav2Vec/results_DL/` — MEJOR RESULTADO DEL PROYECTO

Protocolo: `RepeatedStratifiedKFold(5,10)`=50 evals, nivel paciente. Dos versiones de embeddings (`nuevos`=v2 con orden cronológico por `segment_id`, `viejos`=v1) × dos scripts (`dl_script1`=MLP sobre embedding pooled, 6 experimentos A-F; `dl_script2`=arquitecturas de secuencia CNN1D/BiLSTM/BiGRU/CNN_BiGRU, 4 experimentos).

**Mejor absoluto — `embeddings_nuevos/dl_script1`**: **wav2vec2-large-robust + D_raw_mixup: F1-macro = 0.7234±0.0851, Accuracy = 0.7728, AUC = 0.6681, F1-minoría = 0.6160** (dim=1024). Le sigue xlsr-53+D_raw_mixup (0.7198) y wav2vec2-large-robust+A_raw (0.7140). En `embeddings_viejos/dl_script1` el top es xlsr-53+D_raw_mixup (0.7210) / xlsr-300m+A_raw (0.7165).

**dl_script2 (secuencial)**: top consistente en ambas versiones: **xlsr-300m + CNN1D + raw (F1-macro 0.7115-0.7151, Acc ≈0.77)**. Arquitecturas simples (MLP, CNN1D) superan a recurrentes (BiLSTM/BiGRU) — coherente con dataset pequeño (79 pacientes). `hubert-large` es sistemáticamente el peor modelo SSL.

**nuevo vs viejo**: diffs de código muestran que la única diferencia real es la ruta/formato de embeddings de entrada (bug-fix de agrupación y orden cronológico de segmentos vía `segment_id`) — el impacto en métricas es mínimo (±0.01-0.02 F1-macro).

## 12. Resumen comparativo global

| Enfoque | Mejor config | F1-macro | AUC | Acc |
|---|---|---|---|---|
| Fase0 (transfer learning 2D) | CNN2D 4s | 0.543 | — | 0.666 |
| CNN2D_fase1 | log-mel 2s | 0.517 | 0.575 | 0.593 |
| AST_fase2 | gammatone 2s | 0.570 | 0.615 | 0.679 |
| fase3 (final oficial) | AST_gammatone_3s | 0.564 | 0.564 | 0.618 |
| fase3_viejo (descartado) | AST_gammatone_4s | 0.654 | 0.688 | 0.688 |
| ML clásico (features+NearestCentroid) | manhattan/shrink=2.0 | 0.400 (clase1) | 0.418 | 0.577 |
| **Wav2Vec + MLP (dl_script1)** | **wav2vec2-large-robust+mixup** | **0.723** | 0.668 | **0.773** |
| Wav2Vec + secuencial (dl_script2) | xlsr-300m+CNN1D+raw | 0.712-0.715 | 0.614-0.621 | 0.772-0.775 |

**El mejor modelo global del proyecto es el basado en embeddings Wav2Vec-family + MLP simple con Mixup (F1-macro≈0.72, Acc≈0.77)**, muy por encima de cualquier enfoque de espectrograma 2D (tope ≈0.57 F1-macro) o de ML clásico sobre features artesanales (F1 clase1≈0.40, AUC<0.5).

## 13. Notas metodológicas transversales (para el paper)

1. `Ansiedad1DServidor.ipynb` no tiene resultados — solo documentar como exploración de arquitecturas.
2. Fase0 evalúa "OOF" con n≈395 que no coincide ni con el total de imágenes (1813) ni con pacientes (79) — aclarar antes de citar como accuracy a nivel paciente. CNN2D_fase1/AST_fase2/fase3 sí evalúan correctamente a nivel paciente (segment-voting).
3. Desbalance de clases (19/79≈24%) es el desafío central — colapso a clase mayoritaria recurrente en varios modelos (ResNet50/VGG19/EfficientNet en Fase0, CNN2D_logmel_2s en fase3_viejo holdout).
4. La selección de modelo vía mini-CV (6 folds) resultó poco robusta: el ranking se invirtió en el holdout único y de nuevo en la CV de 5 folds.
5. `fase3` (nuevo, CV completa) es metodológicamente más robusto que `fase3_viejo` (holdout único de 16 con solo 4 positivos) y debe tratarse como el resultado oficial.
6. AUC-ROC de NearestCentroid <0.5 es un artefacto de aproximar el score con distancia a centroides — no reportar sin aclaración.
7. Los embeddings SSL (incluso modelos entrenados en inglés, como `wav2vec2-large-robust`) superan ampliamente a espectrogramas clásicos y a features artesanales para este dataset en español.
8. Existen 4 notebooks (`fase3_split_holdout.ipynb`, `fase3_mini_cv_seleccion_modelo.ipynb`, `fase3_vae_train_generate.ipynb`, `fase3_train_final_evaluate.ipynb`) en la raíz de `Psiquiatria/` (fuera de `Ansiedad/`), que resultaron ser el código fuente real del pipeline fase3 (documentado por el Agente 1 en `report_1_daic_root.md`).
9. Imágenes no leídas como texto (solo documentada su existencia): `confusion_matrix.png`/`roc_curve.png` (raíz y por combo en Fase0/CNN2D_fase1/AST_fase2/fase3), `boxplot_*`, `cm_global_bonita.png`/`roc_global_bonita.png`, `comparacion_resize_vs_noresize.png`, `comparacion_candidatos.png`, `resumen_global_comparacion.png`, `comparacion_cv_vs_test.png`, `vae_loss_curve.png`/`muestras_reales.png`/`vae_reconstrucciones.png`/`vae_muestras_sinteticas_preview.png`, `nearestcentroid_groupkfold_{metricas,auc}.png`.


---

## ANEXO C — Track completo: Depresión

# Reporte exhaustivo — Proyecto Depresión (voz, español)
Directorio: `/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion`

Este reporte cubre TODOS los notebooks, scripts y resultados cuantitativos de la carpeta `Depresion/` para la fase de detección de depresión a partir de voz. Está organizado por fases del pipeline: (0) preprocesamiento de audio, (1) CNN2D sobre espectrogramas, (2) exploración CNN1D/RNN sobre waveform (no ejecutada), (3) EDA + ML clásico sobre features acústicas, (4) ML clásico sobre embeddings Wav2Vec, (5) Deep Learning sobre embeddings Wav2Vec (feature reduction, arquitecturas, rondas BiLSTM), (6) validación cross-dataset DAIC↔propio, y (7) el hallazgo del bug de F1-macro inflado.

---

## 0. Preprocesamiento de audio (scripts)

### `batch_copy_preprocess_audio.py`
- **Objetivo**: copiar y preprocesar audios crudos desde `Clasificación Final/DatosPreprocesados/wav_audios_edit` hacia las carpetas de trabajo `Ansiedad/wav` y `Depresión/wav` (clases `0`/`1`).
- **Pipeline**: normaliza a float32 (soporta int16/int32/uint8), fuerza mono, recorta silencios (`trim_silence`, umbral 30 dB), y aplica **Demucs** (`htdemucs`, modo `--two-stems=vocals`) para separar/eliminar ruido de fondo, guardando `*_recortado.wav` y versión `_denoised`.
- No ejecuta ML; es utilitario de ETL de audio.

### `export_wav_stages.py`
- Utilidad de depuración/QA: para un único participante, exporta un `.wav` por cada etapa del pipeline (raw → solo turnos del participante vía transcript → recortado de silencios → normalizado en pico → denoised con Demucs → segmentos de 5s), para poder escuchar cada paso. No genera resultados de modelo.

### `Depresion/MachineLearning/pipeline_feat_depresion.py`
- Extrae **features acústicas artesanales** (~176 columnas) por segmento: MFCC 1-13 (+Δ, +ΔΔ), LPCC 1-13, F0/pitch (media, std, rango, pendiente), formantes F1-F3 (Parselmouth), jitter, shimmer, HNR, CPP, TEO, % voice breaks, RMS/intensidad, spectral tilt/centroid/rolloff/bandwidth/flatness, duración de habla/silencio/pausas, tasa de articulación. Fuente: `wav/0` (sin depresión) y `wav/1` (con depresión). Salida: `features_depresion.csv` (el proyecto usa la variante `features_depresion_15.csv`, 179 columnas incl. `audio_id`, `seg_idx`, `label`).
- Usa librosa + Parselmouth (opcional, con degradación si no está disponible).

---

## 1. CNN2D_fase1 — Grid de espectrogramas × duración (CNN2D)

**Objetivo**: fase 1 del proyecto — comparar 4 tipos de representación tiempo-frecuencia (CQT, gammatone, log-mel, PCEN) × 4 duraciones de segmento (2s, 3s, 4s, 10s) con una CNN2D, usando validación cruzada repetida (5 repeticiones × 5 folds = 25 evaluaciones por combinación).

- **Ruta**: `CNN2D_fase1/{cqt,gammatone,log-mel,pcen}/{2s,3s,4s,10s}/metrics_cnn2d_*.csv` y `_checkpoint.json`
- **Validación**: 5 repeticiones × StratifiedKFold(5) = 25 folds por config (confirmado: las 16 combinaciones tienen `folds_done: 25` en su `_checkpoint.json`). Cada fold reporta: threshold óptimo, acc, prec, rec, f1 (clase depresión), f1-macro, auc, matriz de confusión y curva ROC (fpr/tpr) por fold.
- Todas las 16 combinaciones están **completas** (25/25 folds).

### Tabla resumen (medias sobre 25 folds, ordenada por F1-macro medio)

| Espectrograma | Duración | Acc media | F1(dep) media | **F1-macro media** | AUC media | SD F1-macro |
|---|---|---|---|---|---|---|
| log-mel | 4s | 0.6513 | 0.3460 | **0.5435** | 0.5758 | 0.1211 |
| gammatone | 2s | 0.6255 | 0.3271 | **0.5185** | 0.5958 | 0.1473 |
| log-mel | 3s | 0.6230 | 0.3148 | **0.5176** | 0.5721 | 0.1021 |
| cqt | 3s | 0.6063 | 0.3351 | **0.5168** | 0.5163 | 0.0968 |
| log-mel | 2s | 0.6282 | 0.3066 | **0.5118** | 0.5873 | 0.1381 |
| pcen | 3s | 0.6328 | 0.2694 | **0.5024** | 0.5497 | 0.1241 |
| pcen | 2s | 0.6363 | 0.2504 | **0.4988** | 0.5857 | 0.0976 |
| gammatone | 3s | 0.5955 | 0.2696 | **0.4824** | 0.5729 | 0.1216 |
| cqt | 2s | 0.5855 | 0.2926 | **0.4805** | 0.5311 | 0.1100 |
| cqt | 4s | 0.5905 | 0.2669 | **0.4689** | 0.5397 | 0.0918 |
| gammatone | 4s | 0.5778 | 0.2540 | **0.4685** | 0.5751 | 0.1242 |
| cqt | 10s | 0.6382 | 0.1733 | **0.4581** | 0.5752 | 0.1019 |
| log-mel | 10s | 0.6328 | 0.1796 | **0.4571** | 0.5167 | 0.1003 |
| pcen | 4s | 0.5758 | 0.2229 | **0.4394** | 0.5394 | 0.1012 |
| pcen | 10s | 0.6978 | 0.0469 | **0.4258** | 0.5228 | 0.0267 |
| gammatone | 10s | 0.6532 | 0.0893 | **0.4214** | 0.5032 | 0.0893 |

**Observaciones**:
- Mejor combinación global: **log-mel a 4 segundos** (F1-macro 0.5435 ± 0.1211, accuracy 0.6513, AUC 0.5758).
- Patrón claro: las duraciones **10s degradan sistemáticamente el F1 de la clase depresión** (F1-dep cae a 0.05-0.18) aunque la accuracy global suba (por el desbalance de clases: el modelo tiende a predecir la clase mayoritaria "no depresión"); esto se ve en pcen-10s (acc=0.698 pero F1-dep=0.047, prácticamente colapsado a predecir siempre "no depresión").
- Los segmentos cortos (2-4s) dan mejor equilibrio F1-macro / recall de la clase minoritaria.
- Todas las configuraciones tienen AUC medio muy cercano al azar (0.50-0.60), señal de que el problema es difícil con este dataset/arquitectura.
- Fuente: `CNN2D_fase1/{spec}/{dur}/metrics_cnn2d_{spec}_{dur}.csv` y `_checkpoint.json` (25 registros por fold con `cm`, `roc_fpr`, `roc_tpr`).

---

## 2. `Depresion1DServidor.ipynb` — Arquitecturas exploratorias sobre waveform crudo (NO EJECUTADO)

**Hallazgo importante**: este notebook **no tiene ninguna salida** — las 28 celdas de código tienen `execution_count: None` y cero outputs. Es código de scaffold/exploración preparado para correr en servidor, pero **no hay evidencia de que se haya ejecutado y producido resultados**. No debe citarse como fuente de métricas.

Contiene 5 familias de arquitectura aplicadas directamente a la forma de onda cruda (`librosa.load`, SR=22050), cada una replicada para 4 duraciones de segmento (2s/3s/4s/10s), con 5-fold `StratifiedKFold`, oversampling de la clase minoritaria y `class_weight='balanced'`, optimizador Adam (lr=1e-4):

1. **CNN1D "VGG19-style"**: bloques Conv1D repetidos (64→128→256→512→512 filtros, kernel decreciente 9→3), GlobalAveragePooling, cabeza FC512→FC256→sigmoid.
2. **RNN/GRU**: CNN subsampler (reduce ~220 500 muestras → ~430 timesteps) + pila de 3 GRU (128→64→32) sin `recurrent_dropout` (para activar kernels CuDNN).
3. **BiLSTM**: misma estrategia de CNN-subsampler + 2 capas BiLSTM (128→64), explícitamente documentada como optimización de velocidad frente a una versión más profunda (3 capas BiLSTM 256+128+64) — la celda de entrenamiento está **comentada/deshabilitada** en el notebook.
4. **CRNN**: bloque combinado Conv+RNN (código presente, sin ejecutar).
5. **Transformer1D**: sobre MFCC (`N_MFCC=40`, `N_FFT=1024`, `HOP_LENGTH=512`), `D_MODEL=64`, `NUM_HEADS=4`, `FF_DIM=128`, `NUM_BLOCKS=2`, `DROPOUT=0.1`, `BATCH_SIZE=32`, `EPOCHS=40`.

Este track fue evidentemente abandonado/superado por el enfoque CNN2D sobre espectrogramas (sección 1) y el track de embeddings Wav2Vec (secciones 3-6).

- Fuente: `Depresion1DServidor.ipynb`

---

## 3. `MachineLearning/EDA_Depresion_Voz15.ipynb` — EDA + limpieza + baseline ML

**Objetivo**: análisis exploratorio y limpieza de `features_depresion_15.csv`, seguido de un modelado inicial (LazyClassifier + BernoulliNB ajustado) como baseline sobre features acústicas artesanales.

**⚠️ Hallazgo de calidad de datos**: la celda 3 (primera carga de datos) muestra en su output guardado `"Archivo cargado: features_ansiedad_15.csv"` con 1753 filas/136 columnas y distribución de clases 1307/445 — **estos números NO coinciden** con `features_depresion_15.csv` real (1752 filas de datos, 179 columnas, distribución 1269 clase 0 / 483 clase 1). Es decir, el **output cacheado de esa celda corresponde a una ejecución previa contra el dataset de Ansiedad** (mismo notebook reutilizado entre proyectos), no a Depresión. Las celdas posteriores (a partir de la celda de partición de datos, ~celda 34) sí cargan correctamente `features_depresion_15.csv` y el resto del pipeline (limpieza, LazyClassifier, BernoulliNB) es consistente con Depresión. Recomendación para el paper: **no citar las cifras de la celda 3 EDA inicial** (outliers, ANOVA, boxplots corridos sobre esas 131 features) como específicas de depresión sin re-verificar.

### Limpieza de datos (`features_depresion_15.csv` → `features_depresion_15_clean.csv`)
Pipeline aplicado (celda con salida real de Depresión, confirmado):
1. Carga `features_depresion_15.csv`: shape inicial **(1752, 179)**.
2. Filtra filas con `voice_breaks_ratio` no confiable (>0.999): **0 filas eliminadas** en esta corrida.
3. Filtro de outliers multivariados (IQR global, `lower/upper = Q1/Q3 ± 3.0×IQR`) sobre 40 columnas seleccionadas; se eliminan filas con `outlier_count_iqr >= 5`: **114 filas eliminadas** → shape (1638, 180).
4. Elimina columnas no significativas según ANOVA univariado previo (p≥0.05) → shape (1638, 106).
5. Winsorización (clip 1%-99%) de 97 columnas numéricas restantes.
6. Guarda `features_depresion_15_clean.csv`: **shape final (1638, 106)** (verificado en disco: 1639 líneas con header = 1638 filas, 106 columnas — coincide exactamente).

### ANOVA univariado (feature vs. `label`)
Ejemplos con mayor significancia (p<0.001): `mfcc10_mean` (F=21.36), `lpcc11_mean` (F=45.04), `lpcc8_mean` (F=38.47), `lpcc11_std` (F=25.71), `zcr_zpsd_mean` (F=42.79), `zcr_zpsd_std` (F=28.48), `zcr_zpsd_p25` (F=36.45), `zcr_zpsd_p75` (F=38.58), `rms_mean` (F=21.21), `rms_p25` (F=38.15), `mfcc6_std` (F=23.09), `mfcc5_mean` (F=17.59). Muchas variables no significativas, especialmente deltas/delta-deltas de MFCC.

### Modelado inicial — LazyClassifier + GroupKFold(5) + SMOTE (train-only)
Sobre `features_depresion_15_clean.csv` (1638 muestras, 79 audios únicos, distribución {0: 1185, 1: 453}), `GroupKFold(5)` por `audio_id`, SMOTE sólo en train de cada fold.

**Top 5 modelos (promedio de 5 folds):**

| Modelo | Accuracy | Balanced Acc. | F1 Score | F1-macro | ROC AUC |
|---|---|---|---|---|---|
| BernoulliNB | 0.6267 | 0.5744 | 0.6459 | 0.5494 | 0.6145 |
| GaussianNB | 0.5982 | 0.5835 | 0.6209 | 0.5330 | 0.5944 |
| NearestCentroid | 0.6136 | 0.5510 | 0.6319 | 0.5324 | 0.5969 |
| DecisionTreeClassifier | 0.6181 | 0.5069 | 0.6334 | 0.5169 | 0.5069 |
| Perceptron | 0.6150 | 0.5587 | 0.6210 | 0.5163 | 0.5536 |

**Evaluación final del top-5 con GroupKFold completo (F1-macro):** BernoulliNB media 0.5494±0.0640 (folds: 0.6584/0.5108/0.4990/0.5863/0.4925), GaussianNB 0.5330±0.0434, NearestCentroid 0.5324±0.0566, DecisionTreeClassifier 0.5169±0.0818, Perceptron 0.5163±0.0662.

### BernoulliNB con ajuste de hiperparámetros (GroupKFold + SMOTE)
Grid: `alpha ∈ {0.01,0.1,0.5,1.0,2.0}`, `fit_prior ∈ {True,False}`, `binarize ∈ {0.0,0.1,0.3,0.5}`. **Mejores hiperparámetros: `alpha=0.5, binarize=0.0, fit_prior=False`**.

**Métricas finales (GroupKFold 5 folds, promedio ± std):**
| Métrica | Media | SD |
|---|---|---|
| Accuracy | 0.6103 | 0.1222 |
| Precision | 0.3388 | 0.1433 |
| Recall (Sens.) | 0.5684 | 0.1499 |
| F1 Score (weighted) | 0.6306 | 0.1437 |
| **F1 Macro** | **0.5433** | **0.0782** |
| Especificidad | 0.6164 | 0.1303 |
| ROC AUC | 0.6336 | 0.1369 |

Por fold: Fold1 (Acc 0.545, F1-macro 0.535, AUC 0.598), Fold2 (0.700, 0.640, 0.697), Fold3 (0.762, 0.524, 0.794), Fold4 (0.588, 0.587, 0.654), Fold5 (0.456, 0.430, 0.425). Gráfico: `bernoullinb_final_report.png`.

Al final del notebook hay un bloque grande (grid search de BernoulliNB, NearestCentroid, XGBoost, LightGBM) **completamente comentado** — nunca se ejecutó.

- Fuente: `MachineLearning/EDA_Depresion_Voz15.ipynb`, `MachineLearning/features_depresion_15.csv`, `MachineLearning/features_depresion_15_clean.csv`

---

## 4. ML clásico sobre embeddings Wav2Vec (dataset propio)

### 4.1 `MachineLearning/ML-Wav2Vec.ipynb` — pipeline "antiguo" (embeddings por paciente, N=79)

**Dataset**: `Clasificación Final/Depresión/embeddings/` (pipeline antiguo, manifest agrupado por paciente), **79 pacientes** (1 vector por paciente). 6 modelos SSL: `xlsr-300m`, `xlsr-53`, `whisper-large-encoder`, `wav2vec2-large-robust`, `wavlm-large`, `hubert-large`.

**Validación**: `RepeatedStratifiedKFold(5,10)` = 50 evaluaciones, LazyClassifier por fold, 6 pipelines de preprocesamiento.

**Rankings globales (top 1 por pipeline):**

| Pipeline | Mejor combinación | F1-macro | Accuracy | AUC |
|---|---|---|---|---|
| Sin SMOTE (class_weight balanced) | wav2vec2-large-robust + ExtraTreeClassifier | 0.6332 | 0.7467 | 0.6503 |
| SMOTE | wav2vec2-large-robust + ExtraTreesClassifier | 0.6354 | 0.7609 | 0.6701 |
| StandardScaler+SMOTE | wav2vec2-large-robust + ExtraTreesClassifier | 0.6373 | 0.7633 | 0.6717 |
| BorderlineSMOTE | wav2vec2-large-robust + ExtraTreesClassifier | 0.6354 | 0.7584 | 0.6759 |
| StandardScaler+BorderlineSMOTE | wav2vec2-large-robust + ExtraTreesClassifier | **0.6543** | 0.7749 | 0.6777 |
| StandardScaler+PCA(0.95)+SMOTE | wav2vec2-large-robust + SVC | 0.5663 | 0.6863 | 0.5837 |

**GridSearchCV** (StandardScaler→BorderlineSMOTE→ExtraTreesClassifier, `wav2vec2-large-robust`, `RepeatedStratifiedKFold(5,3)`=15 folds, 8640 combinaciones = 129 600 fits): **mejor F1-macro de validación = 0.6862 (±0.1084)**, accuracy 0.7719 (±0.0883), hiperparámetros óptimos `class_weight=balanced, criterion=entropy, max_depth=None, max_features=sqrt, min_samples_leaf=5, min_samples_split=2, n_estimators=100`.

**Evaluación final con esos hiperparámetros** (`RepeatedStratifiedKFold(5,10)`=50 folds): **F1-macro = 0.6684 ± 0.1325**, Accuracy = 0.7622 ± 0.0923, ROC AUC = 0.6790 ± 0.1521 (celda ejecutada dos veces con resultados idénticos). Se guardaron `scores_evaluacion_final.csv`, `confusion_matrix.png`, `roc_curve.png` pero **no persisten actualmente en disco**.

**Nota metodológica**: N=79 (1 fila = 1 paciente) con 50 folds explica la alta desviación estándar (±0.13-0.15) — test por fold de solo ~16 pacientes.

- Fuente: `MachineLearning/ML-Wav2Vec.ipynb`

### 4.2 `MachineLearning/ML-Wav2Vec-GroupKFold.ipynb` — pipeline a nivel de segmento con GroupKFold (más riguroso)

**Dataset**: `Clasificación Final/Depresión/embeddings_v2/depresion` (por segmento: 5252 segmentos, 79 pacientes). `GroupKFold(5)` agrupado por `audio_id` — evita fuga de segmentos del mismo paciente entre train/val.

**Ranking global top-1 por pipeline:**

| Pipeline | Mejor combinación | F1-macro | Accuracy | AUC |
|---|---|---|---|---|
| Sin SMOTE | wav2vec2-large-robust + PassiveAggressiveClassifier | 0.5455 | 0.6277 | 0.5695 |
| SMOTE | xlsr-53 + SVC | 0.5454 | 0.6016 | 0.5832 |
| StandardScaler+SMOTE | wav2vec2-large-robust + SVC | 0.5463 | 0.6255 | 0.5813 |
| BorderlineSMOTE | xlsr-53 + SVC | 0.5448 | 0.5987 | 0.5840 |
| StandardScaler+BorderlineSMOTE | wav2vec2-large-robust + SVC | 0.5484 | 0.6249 | 0.5783 |
| StandardScaler+PCA(15)+SMOTE | wav2vec2-large-robust + SVC | **0.5539** | 0.6100 | 0.5944 |

**GridSearchCV** (ExtraTreesClassifier, `wav2vec2-large-robust`, `GroupKFold(5)`, 8640 combinaciones = 43 200 fits) fue **interrumpido manualmente (`KeyboardInterrupt`)**; el notebook fija manualmente hiperparámetros: `class_weight=balanced, criterion=entropy, max_depth=None, max_features=sqrt, min_samples_leaf=5, min_samples_split=2, n_estimators=100`.

**Evaluación final (`GroupKFold(5)`, wav2vec2-large-robust, confirmado en `scores_evaluacion_final_groupkfold.csv`):**

| Fold | F1-macro | Accuracy | ROC AUC |
|---|---|---|---|
| 1 | 0.4481 | 0.5970 | 0.4220 |
| 2 | 0.5714 | 0.7294 | 0.6294 |
| 3 | 0.5944 | 0.7170 | 0.6584 |
| 4 | 0.4806 | 0.6515 | 0.6101 |
| 5 | 0.5141 | 0.6815 | 0.5817 |
| **Media** | **0.5217** | **0.6753** | **0.5803** |

Guardados: `confusion_matrix_groupkfold.png` (35 KB), `roc_curve_groupkfold.png` (85 KB).

**Comparación 4.1 vs 4.2 (mismo modelo, ExtraTreesClassifier optimizado, wav2vec2-large-robust)**: sin agrupar por paciente (4.1, N=79) → F1-macro 0.6684±0.1325; agrupando por paciente (4.2, N=5252 segmentos) → F1-macro 0.5217±0.0546. La caída de ~0.15 puntos es consistente con que 4.1 es más optimista al no aislar completamente pacientes en repeticiones.

- Fuente: `MachineLearning/ML-Wav2Vec-GroupKFold.ipynb`, `MachineLearning/scores_evaluacion_final_groupkfold.csv`, `MachineLearning/confusion_matrix_groupkfold.png`, `MachineLearning/roc_curve_groupkfold.png`

### 4.3 `MachineLearning/ML_Wav2Vec_DAIC_Holdout_1_.ipynb` — Holdout: entrenar en DAIC-WOZ, testear en dataset propio

**Diseño**: TRAIN = DAIC-WOZ (`embeddings_daic/`), TEST = dataset propio (`embeddings_v2/depresion`). Scaler/SMOTE/BorderlineSMOTE ajustados sólo en TRAIN. LazyClassifier con 7 pipelines × 6 modelos de embeddings.

**Tamaños**: TRAIN (DAIC) = 36 452 segmentos (class_0=25 375, class_1=11 077, 189 pacientes); TEST (propio) = 5252 segmentos (class_0=3799, class_1=1453, 79 pacientes).

**Mejor configuración por modelo de embedding (`holdout_best_per_model.csv`):**

| Embedding | Pipeline | Clasificador | Accuracy | Balanced Acc. | F1 Score | F1-macro | ROC AUC |
|---|---|---|---|---|---|---|---|
| wav2vec2-large-robust | P6 (scaler+PCA15+Borderline) | PassiveAggressiveClassifier | 0.6588 | 0.5584 | 0.6531 | **0.5599** | **0.5967** |
| whisper-large-encoder | P6 | PassiveAggressiveClassifier | 0.4396 | 0.5483 | 0.4401 | 0.4396 | 0.5811 |
| xlsr-300m | P2 (scaler+SMOTE) | BernoulliNB | 0.7233 | 0.5000 | 0.6072 | 0.4197 | 0.5477 |
| xlsr-53 | P6 | PassiveAggressiveClassifier | 0.5128 | 0.5300 | 0.5377 | 0.4928 | 0.5447 |
| wavlm-large | P2 | KNeighborsClassifier | 0.5145 | 0.5271 | 0.5396 | 0.4927 | 0.5291 |
| hubert-large | P6 | NuSVC | 0.6632 | 0.5056 | 0.6245 | 0.4936 | 0.5203 |

Resultados completos: `holdout_all_results.csv` (1020 filas). El intento de reevaluar en detalle la configuración ganadora fue **interrumpido (`KeyboardInterrupt`)**, sin classification_report/confusión detallada persistida.

**Interpretación**: el mejor AUC cross-dataset (DAIC→propio) es 0.597 — apenas por encima del azar, evidenciando **generalización pobre** de inglés (DAIC-WOZ) a español a nivel de holdout puro.

- Fuente: `MachineLearning/ML_Wav2Vec_DAIC_Holdout_1_.ipynb`, `MachineLearning/holdout_outputs/holdout_all_results.csv`, `holdout_best_per_model.csv`, `plots/holdout_heatmap_auc.png`, `holdout_heatmap.png`

### 4.4 `MachineLearning/ML_Wav2Vec_DAIC_merged.ipynb` — CV agrupada sobre pool combinado (DAIC + propio)

**Diseño**: pool único DAIC+propio (IDs prefijados `ds0_`/`ds1_`), `StratifiedGroupKFold(5)` agrupado por paciente, LazyClassifier, mismos 7 pipelines que 4.3.

**Estado: ejecución incompleta.** Sólo se completaron 2 de 6 modelos (`xlsr-300m`, `xlsr-53`), quedó a mitad de `whisper-large-encoder` (pool: X=(41 704, 1024/1280), 268 pacientes, class_0=29 174, class_1=12 530) y luego falló con `NameError: name 'pd' is not defined`. `MachineLearning/merged_outputs/` está **vacío en disco**.

**Resultados parciales (top, ROC AUC apenas sobre azar 0.51-0.55):** xlsr-300m P0: NuSVC roc_auc=0.541, f1_macro=0.512; xlsr-53 P0: BernoulliNB roc_auc=0.527, f1_macro=0.508. **No concluyente.**

- Fuente: `MachineLearning/ML_Wav2Vec_DAIC_merged.ipynb`

---

## 5. Deep Learning sobre embeddings Wav2Vec (`Wav2Vec/results_DL/`)

Track más extenso: búsqueda de arquitectura (MLP → CNN1D/BiLSTM/BiGRU/CNN+BiGRU) en varias generaciones (`dl_script1`→`dl_script2`→rondas BiLSTM `2b→2c→2d→2e→final`), más cross-dataset y CV combinada.

### 5.1 `combined_cv/` — MLP, pool combinado DAIC+propio, StratifiedGroupKFold

Top 5 (por mean_f1_macro, 25 runs c/u):
| Embedding | Exp. | F1-macro | AUC | Accuracy |
|---|---|---|---|---|
| whisper-large-encoder | raw | **0.6622** ± 0.0577 | 0.6765 | 0.6889 |
| hubert-large | raw | 0.5884 ± 0.1003 | 0.6357 | 0.6063 |
| wavlm-large | raw | 0.5816 ± 0.0756 | 0.6465 | 0.5934 |
| wavlm-large | pca_smote | 0.5756 ± 0.1108 | 0.6173 | 0.5951 |
| whisper-large-encoder | pca_smote | 0.5641 ± 0.1083 | 0.6256 | 0.5842 |

- Fuente: `Wav2Vec/results_DL/combined_cv/summary.csv` (+`raw_folds.csv`, 450 filas)

### 5.2 `cross_dataset/` y `cross_dataset_all_archs/` — cross-dataset DAIC↔propio

`cross_dataset/summary.csv` (sólo MLP): mejor **daic→propio, xlsr-300m, pca_smote**: AUC 0.5314, F1-macro 0.4013, acc 0.4241. Todos AUC 0.43-0.53.

`cross_dataset_all_archs/summary.csv` (5 arquitecturas × 2 embeddings × 2 direcciones × raw/pca, 10 runs c/u):

| Dirección | Embedding | Arquitectura | Exp. | AUC | F1-macro | Acc |
|---|---|---|---|---|---|---|
| daic→propio | xlsr-300m | CNN_BiGRU | pca | 0.5695 | 0.4514 | 0.4658 |
| propio→daic | wav2vec2-large-robust | CNN_BiGRU | raw | 0.5510 | 0.3333 | 0.3667 |
| daic→propio | wav2vec2-large-robust | CNN_BiGRU | pca | 0.5509 | 0.4544 | 0.4570 |

`nuevos_embeddings/cross_dataset_all_archs/`: mejor **propio→daic, wav2vec2-large-robust, CNN_BiGRU, raw**: AUC 0.5726, F1-macro 0.4734, acc 0.4859.

**Conclusión clave**: en ninguna combinación se supera AUC≈0.57 cross-dataset. CNN_BiGRU es la arquitectura menos mala para transferencia, pero la generalización DAIC(inglés)↔propio(español) es pobre.

- Fuente: `Wav2Vec/results_DL/cross_dataset/summary.csv`, `raw_runs.csv`; `cross_dataset_all_archs/summary.csv`, `raw_runs.csv`; `nuevos_embeddings/cross_dataset_all_archs/summary.csv`, `raw_runs.csv`

### 5.3 `embeddings_nuevos/` y `embeddings_viejos/` — dl_script1 (MLP) y dl_script2 (arquitecturas secuencia)

**Importante**: "nuevos" vs "viejos" = sólo versión del pipeline de embeddings (`embeddings_v2/` segmentos sueltos vs `embeddings/` manifest antiguo agrupado), verificado por diff de código. **No** relacionado con el bug de F1-macro (sección 6).

Protocolo: `RepeatedStratifiedKFold(5,10)`=50 evaluaciones, N=79 pacientes.

**`dl_script1` (MLP, experimentos A-F):** A=raw, B=PCA(95%), C=SFS, D=raw+Mixup, E=PCA+SMOTE, F=PCA+ADASYN.

`embeddings_nuevos/dl_script1/summary.csv`:
| Embedding | Exp. | F1-macro | AUC | Accuracy | dim |
|---|---|---|---|---|---|
| wav2vec2-large-robust | A_raw | **0.7770** ± 0.1280 | 0.7400 | 0.8207 | 1024 |
| wav2vec2-large-robust | D_raw_mixup | 0.7721 ± 0.0727 | 0.7181 | 0.8192 | 1024 |
| wav2vec2-large-robust | C_sfs | 0.7573 ± 0.1008 | 0.6850 | 0.8045 | 48 |

`embeddings_viejos/dl_script1/`: equivalente, mejor = A_raw F1-macro 0.7575±0.1132, AUC 0.6943.

**`dl_script2` (CNN1D/BiLSTM/BiGRU/CNN+BiGRU):**

`embeddings_nuevos/dl_script2/summary.csv`:
| Embedding | Arquitectura | Exp. | F1-macro | AUC | Accuracy |
|---|---|---|---|---|---|
| wav2vec2-large-robust | BiLSTM | raw | **0.7638** ± 0.1123 | 0.7459 | 0.7957 |
| wav2vec2-large-robust | BiLSTM | raw_mixup | 0.7524 ± 0.1294 | 0.7276 | 0.7852 |

`embeddings_viejos/dl_script2/`: BiLSTM raw F1-macro 0.7549±0.1049, AUC 0.7396.

- Fuente: `Wav2Vec/results_DL/embeddings_nuevos/dl_script1/summary.csv` (+`raw_folds.csv`,1800 filas), `dl_script2/summary.csv` (+`raw_folds.csv`,4800 filas), `embeddings_viejos/dl_script1/summary.csv`, `dl_script2/summary.csv`

### 5.4 Rondas sucesivas de optimización de BiLSTM (`2b→2c→2d→2e→2e_fixed→bilstm_final`)

Solo `wav2vec2-large-robust`. Cada script documenta en su docstring qué cambió respecto a la ronda anterior:

**Ronda 1 (`dl_script2b_bilstm_optim.py`, 84 configs, 50 folds):** explora pooling (last/meanmax/attention), loss (bce_posw/focal), hidden (64/96/128/192), augmentación.
- FIX-A: mejor checkpoint por AUC (no F1). FIX-B: threshold ya no se busca en val (leakage) — se calcula en TRAIN, se reporta también con thr=0.5 fijo. FIX-C: longitud sintética de Mixup = max(len_i,len_j). FIX-D: semillas fijas.
- Mejor: meanmax+focal+warmup+tmask, hidden=128 → **AUC=0.784±0.105**, pero F1-macro(thr-train)=0.498 vs F1-macro(thr=0.5)=0.559.

**Ronda 2 (`dl_script2c_bilstm_round2.py`, sin carpeta de resultados en disco):**
- Problema: threshold-en-TRAIN da peor F1 que thr=0.5 fijo (el modelo sobreajusta en train). FIX-E: split FIT(80%)/CALIB(20%) dentro de cada fold — threshold calculado en CALIB (real, nunca visto en gradiente). FIX-F: ensembles multi-seed.
- Resultados (según docstring de Ronda 3): hidden=192→AUC 0.783 (mejora real); gamma=3.0→0.778; ensemble3(hidden=128)→AUC 0.781 pero F1-macro 0.620 vs 0.474 del ganador R1.

**Ronda 3 (`dl_script2d_bilstm_round3.py`, `summary_bilstm_r3.csv`, 8 configs):**
| Config | AUC | F1-macro | Acc |
|---|---|---|---|
| r3_hidden192_dropout05_ensemble3 | **0.7920**±0.1061 | 0.6013 | 0.6599 |
| r3_hidden192_gamma3_ensemble5 | 0.7916±0.1137 | **0.6115** | 0.6642 |
| r3_combined_final (PCA on) | 0.7119±0.1264 | 0.5374 | 0.6203 |

PCA descartado definitivamente (AUC 0.68-0.71).

**Ronda 4 (`dl_script2e_bilstm_round4.py`, `summary_bilstm_r4.csv`, 3 configs, hidden=192, ensemble5):**
| Config | AUC | F1-macro |
|---|---|---|
| r4_dropout05_wd5e4_ensemble5 | **0.7965**±0.1113 | 0.6133 |
| r4_dropout05_gamma3_ensemble5 | 0.7953±0.1191 | **0.6137** |
| r4_dropout055_gamma3_ensemble5 | 0.7962±0.1114 | 0.6033 |

El propio script anticipa: "si ninguna supera ~0.795-0.80, es señal de techo dado N=79... std_auc≈0.11-0.12 consistente en las 4 rondas".

**Ronda 4 corregida (`dl_script2e_bilstm_round4_fixed.py`, `summary_bilstm_r4_fixed.csv`):** FIX-F adicional: checkpoint elegido por **loss en CALIB**, no por AUC/F1.

| Config | AUC R4 | AUC R4-fixed | Δ | F1-macro fixed |
|---|---|---|---|---|
| r4_dropout05_wd5e4_ensemble5 | 0.7965 | **0.7086** | **-0.088** | 0.5633 |
| r4_dropout055_gamma3_ensemble5 | 0.7962 | 0.7074 | -0.089 | 0.5619 |
| r4_dropout05_gamma3_ensemble5 | 0.7953 | 0.6969 | -0.098 | 0.5657 |

**Hallazgo central**: al eliminar el último resquicio de fuga en selección de checkpoint, el AUC cae ~0.09 puntos de forma consistente en las 3 configs — confirma que buena parte del "techo 0.78-0.80" era optimismo por selección múltiple.

**Protocolo final "para paper" (`dl_script_bilstm_final.py`, `dl_bilstm_final/summary.csv`, 34 configs):** early stopping por val LOSS (no AUC/F1), threshold sólo en TRAIN, scaler/PCA sólo en timesteps reales.
- Mejor por AUC: `pooling_meanmax` (hidden=128, bce_posw, dropout=0.4): **AUC=0.7659±0.1295**, F1-macro(thr-train)=0.5175, F1-macro(thr=0.5)=0.5823, acc=0.737.
- Mejor por F1-macro: `pooling_attention`: AUC=0.6623, **F1-macro=0.5571**, acc=0.6796.

**Evolución del "techo" de AUC:**
| Ronda | Mejor AUC | Nota |
|---|---|---|
| 1 (2b) | 0.784 | threshold en TRAIN |
| 2 (2c) | ~0.781-0.784 | calibración honesta (CALIB) + ensembles |
| 3 (2d) | 0.792 | hidden=192+regularización+ensemble |
| 4 (2e) | 0.797 | combina 3 ejes ganadores |
| 4 fixed | **0.709** | fix selección checkpoint por loss en CALIB |
| Final | 0.766 | protocolo declarable en paper |

- Fuente: `Wav2Vec/results_DL/embeddings_nuevos/dl_script2b_bilstm_optim/summary_bilstm.csv`, `dl_script2d_bilstm_round3/summary_bilstm_r3.csv`, `dl_script2e_bilstm_round4/summary_bilstm_r4.csv`, `dl_script2e_bilstm_round4_fixed/summary_bilstm_r4_fixed.csv`, `dl_bilstm_final/summary.csv`; scripts `dl_script2b_bilstm_optim.py`, `dl_script2c_bilstm_round2.py`, `dl_script2d_bilstm_round3.py`, `dl_script2e_bilstm_round4.py`, `dl_script2e_bilstm_round4_fixed.py`, `dl_script_bilstm_final.py`

---

## 6. El bug de F1-macro inflado (`Resultados ANTIGUOS CON BUG F1_MACRO ALTO/`) — hallazgo central

**Qué contiene**: generación completa y **archivada/invalidada** de `dl_script1`(MLP) y `dl_script2`(CNN1D/BiLSTM/BiGRU/CNN+BiGRU), más 4 configs con gráficos detallados por fold en `dl_script3_plots/`.

`dl_script3_plots/summary_all_configs.csv`:
| Config | F1-macro | AUC | Accuracy |
|---|---|---|---|
| MLP · wav2vec2-large-robust · raw | **0.7792** ± 0.0891 | 0.7129 | 0.8240 |
| MLP · wav2vec2-large-robust · raw+Mixup | 0.7624 ± 0.1087 | 0.7080 | 0.8064 |
| CNN-1D · wavlm-large · raw | 0.7377 ± 0.1238 | 0.6749 | 0.7843 |
| CNN+BiGRU · wav2vec2-large-robust · raw | 0.7263 ± 0.1619 | 0.6890 | 0.7662 |

**Mecanismo técnico (confirmado por diff `dl_script2_architectures.py` vs `dl_script2_architectures_fixed.py`)**, en arquitecturas de secuencia `(n_pacientes, max_segmentos, dim)` con zero-padding dinámico:

1. **"Problema 2"** — StandardScaler/PCA ajustados sobre la secuencia completa **incluyendo el padding**: distorsiona media/varianza/covarianza porque el padding es proporcionalmente distinto por paciente. Fix: ajustar sólo sobre timesteps reales (`X_train[i, :lengths_train[i], :]`).
2. **"Problema 3"** — LSTM/GRU **sin `pack_padded_sequence`**: el hidden state final se calcula tras procesar TODO T (real+padding), contaminando `h_n` con una cantidad de pasos de padding que depende de la longitud real de cada paciente. Fix: `pack_padded_sequence`/`pad_packed_sequence`.
3. **Fuga transversal a `dl_script1`, `dl_script2`, `dl_script3_eval_plots`**: el **umbral de clasificación se buscaba directamente sobre las probabilidades del propio conjunto de validación** (confirmado en código: `find_best_threshold(val_labels, val_probs)`, y en docstring de `dl_script3_eval_plots.py`: *"Umbral óptimo buscado en val"*). Sólo se corrigió parcialmente en Ronda 1 de BiLSTM (FIX-B) y de forma robusta en Ronda 2 (FIX-E) y el protocolo final.

**Comparación cuantitativa** (dl_script2, 96 configs coincidentes, "ANTIGUOS CON BUG" vs `embeddings_nuevos`/`embeddings_viejos` post-fix):
- Diferencia media (bug−corregido) ≈ **-0.011** (nuevos) y **-0.008** (viejos) — no hay inflación sistemática uniforme en el agregado.
- Rango por config individual: **-0.20 a +0.10** de F1-macro — algunas configs cambian hasta 20 puntos porcentuales tras el fix.
- El bug da valores más altos en ~46% de las configs (no en todas) — consistente con que el mecanismo depende de la distribución de longitudes de secuencia por fold, no de un sesgo constante.

**Recomendación para "lecciones aprendidas"/calidad de datos (CRISP-DM)**:
- Los resultados de `Resultados ANTIGUOS CON BUG F1_MACRO ALTO/` (incluido el "F1-macro=0.78" de MLP+wav2vec2-large-robust+raw) **deben tratarse como inválidos** y no citarse como resultado final.
- Todo el track de secuencia comparte la fuga de "umbral óptimo en validación" hasta el protocolo final; los números de `dl_script1`/`dl_script2`/rondas 1-4 (F1-macro 0.53-0.78) no son directamente comparables con la sección 4.2 (GroupKFold, F1-macro≈0.52) ni con `CNN2D_fase1`.
- El hallazgo Ronda4→Ronda4-fixed (caída de AUC ~0.09) es la evidencia cuantitativa más clara, auto-documentada por el propio equipo, de cuánto puede inflar una fuga de selección de modelo/umbral en un dataset pequeño (N=79) con CV repetida de alta varianza.
- Como estimación más conservadora del desempeño real, usar el protocolo final (`dl_script_bilstm_final.py`: AUC≈0.77, F1-macro≈0.52-0.58) o el GroupKFold a nivel de segmento (sección 4.2: F1-macro≈0.52, AUC≈0.58).

- Fuente: `Wav2Vec/results_DL/Resultados ANTIGUOS CON BUG F1_MACRO ALTO/dl_script1/summary.csv` (+`raw_folds.csv`, 1800 filas, y `raw_folds_parciales_incompletos_D:.csv`, 550 filas — nombre con ruta de Windows filtrada, indicio adicional de higiene de datos deficiente), `dl_script2/summary.csv` (+`raw_folds.csv`, 4800 filas), `dl_script3_plots/summary_all_configs.csv` y `metrics_per_fold.csv` por config; scripts `dl_script2_architectures.py` vs `dl_script2_architectures_fixed.py`, `dl_script1_feature_reduction.py`, `dl_script2b_bilstm_optim.py`, `dl_script2c_bilstm_round2.py`, `dl_script2e_bilstm_round4_fixed.py`, `dl_script_bilstm_final.py`

---

## 7. Scripts adicionales (propósito, sin transcripción de código)

### Validación cruzada entre datasets
- **`cross_dataset_common.py`**: módulo compartido con toda la lógica anti-leakage: carga pooled/secuencia, preprocesamiento (fit sólo en dataset origen), arquitecturas (MLP/CNN1D/BiLSTM/BiGRU/CNN_BiGRU vía `pack_padded_sequence`), evaluación cross-dataset sin CV (split fijo por origen).
- **`daic_embedding_pipeline.py`**: adapta el pipeline propio a DAIC-WOZ: usa transcripts oficiales (sólo turnos "Participant", excluye a "Ellie"), denoising con `noisereduce`, etiquetas AVEC2017, mismo formato de manifest.
- **`daic_embedding_pipeline_merged.py`**: fusiona dos implementaciones previas: Demucs (`--two-stems=vocals`) en vez de `noisereduce`, corrige bug de `attention_mask` en wav2vec (pasar `None`), 0.1s de silencio entre turnos, embeddings de todas las capas del transformer, detección flexible de carpetas/columnas.
- **`dl_cross_dataset_all_archs.py`**: replica el cross-dataset (ya corrido con MLP) para las 5 arquitecturas, usando `cross_dataset_common.py` (sección 5.2).
- **`dl_script_cross_dataset.py`**: script original con experimentos A (DAIC→propio) y B (propio→DAIC), orden de preprocesamiento documentado como "OBLIGATORIO, no cambiar" (produjo `cross_dataset/`, sólo MLP).
- **`dl_combined_cv.py`**: mezcla DAIC+propio, `StratifiedGroupKFold(5)`×N repeticiones manuales (agrupado por `patient_id`); mean-pooling por paciente antes del CV para arquitecturas de secuencia (produjo `combined_cv/`).

### Búsqueda de arquitectura BiLSTM/BiGRU (rondas)
- **`dl_script1_feature_reduction.py`**: MLP baseline, 6 variantes A-F. Umbral óptimo buscado en validación (fuga documentada en sección 6).
- **`dl_script2_architectures.py`/`_fixed.py`**: arquitecturas de secuencia; versión fixed corrige fit de scaler/PCA y usa `pack_padded_sequence` (detalle en sección 6).
- **`dl_script2b→2c→2d→2e→2e_fixed→bilstm_final`**: seis iteraciones sucesivas documentadas en sección 5.4.
- **`dl_script3_eval_plots.py`**: genera ROC/matrices de confusión por fold para 4 configs top; comparte la fuga de threshold-en-val; generó `Resultados ANTIGUOS CON BUG F1_MACRO ALTO/dl_script3_plots/`.

### ML clásico sobre features acústicas (`MachineLearning/`)
- **`autoML.py`**: pipeline AutoML ambicioso (ColumnTransformer+imputación+VarianceThreshold/SelectFromModel, GroupKFold/StratifiedGroupKFold, RandomForest/ExtraTrees/XGBoost/LightGBM/CatBoost+StackingClassifier, CalibratedClassifierCV, SMOTE/RandomOverSampler vía imblearn, Optuna N_TRIALS=40, SHAP). Guardaría en `outputs_automl/` — **carpeta no encontrada en disco**, sin evidencia de ejecución completa.
- **`pipeline_feat_depresion.py`**: extractor de features acústicas (sección 0), produce `features_depresion_15.csv`.

---

## 8. Resumen ejecutivo — mejor modelo y hallazgos clave

| Track | Mejor config | F1-macro | AUC | Acc |
|---|---|---|---|---|
| CNN2D sobre espectrograma | log-mel, 4s, CNN2D | 0.5435±0.121 | 0.576 | 0.651 |
| ML clásico, features acústicas | BernoulliNB tuneado, GroupKFold | 0.5433±0.078 | 0.634 | 0.610 |
| ML clásico, embeddings Wav2Vec, GroupKFold por segmento | ExtraTrees, wav2vec2-large-robust | **0.5217±0.055** | 0.580 | 0.675 |
| DL BiLSTM, protocolo final "para paper" | pooling_meanmax, wav2vec2-large-robust | 0.518-0.582 | **0.766±0.129** | 0.737 |
| DL BiLSTM Ronda 4 fixed (última corrección leakage) | dropout=0.5, gamma=3, ens5 | 0.563-0.566 | 0.697-0.709 | 0.657-0.670 |
| Cross-dataset (DAIC↔propio), mejor caso | CNN_BiGRU/xlsr-300m, pca | 0.451 | 0.570 | 0.466 |
| ML embeddings sin agrupar por paciente (N=79, cautela) | ExtraTrees tuneado, wav2vec2-large-robust | 0.668±0.133 | 0.679 | 0.762 |
| **Resultados INVALIDADOS por bug (no usar)** | MLP, wav2vec2-large-robust, raw | 0.779±0.089 | 0.713 | 0.824 |

**Conclusiones para el paper (CRISP-DM):**
1. El problema es difícil: pipelines sin fugas conocidas dan F1-macro 0.52-0.58, AUC 0.58-0.77 — modesto pero sobre el azar.
2. Generalización cross-idioma (DAIC inglés ↔ propio español) es pobre (AUC 0.42-0.60 en todas las combinaciones).
3. **Bug de F1-macro inflado**: fuga en dos frentes (estadísticas de normalización contaminadas por padding en RNNs; umbral optimizado sobre validación) infló de forma no uniforme una generación completa de resultados (hasta F1-macro=0.78). El equipo lo detectó, corrigió en 6 rondas documentadas (FIX-A a FIX-F), y cuantificó el efecto (caída de AUC ~0.09 en Ronda4→Ronda4-fixed). Documentar como lección aprendida de validación en la fase de Evaluación del CRISP-DM.
4. Discrepancia en `EDA_Depresion_Voz15.ipynb`: celda con output cacheado del dataset de Ansiedad en vez de Depresión — verificar antes de citar.
5. Ejecuciones incompletas que no deben citarse como fuente de cifras finales: `Depresion1DServidor.ipynb` (nunca ejecutado), `ML_Wav2Vec_DAIC_merged.ipynb` (interrumpido), `dl_script2c_bilstm_round2.py` (sin resultados persistidos).

**Rutas de archivos fuente relevantes** (todas bajo `/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/`): `CNN2D_fase1/`, `Depresion1DServidor.ipynb`, `MachineLearning/EDA_Depresion_Voz15.ipynb`, `MachineLearning/ML-Wav2Vec.ipynb`, `MachineLearning/ML-Wav2Vec-GroupKFold.ipynb`, `MachineLearning/ML_Wav2Vec_DAIC_Holdout_1_.ipynb`, `MachineLearning/ML_Wav2Vec_DAIC_merged.ipynb`, `Wav2Vec/results_DL/` (combined_cv, cross_dataset, cross_dataset_all_archs, embeddings_nuevos, embeddings_viejos, nuevos_embeddings, `Resultados ANTIGUOS CON BUG F1_MACRO ALTO`), y todos los `dl_script*.py`, `Scripts_Embedding_nuevo/`, `Scripts_Embedding_viejo/`, `cross_dataset_common.py`, `daic_embedding_pipeline*.py`, `MachineLearning/autoML.py`, `MachineLearning/pipeline_feat_depresion.py`.


---

## ANEXO D — Dataset propio 'Clasificación Final' y minería de logs de ejecución (581 resultados extraídos)

# Reporte de recopilación: Dataset propio y logs de ejecución

Fecha de recopilación: 2026-08-06
Alcance: 100% lectura/investigación, ningún archivo del proyecto fue modificado.

---

## 1. Dataset propio (Clasificación Final)

Ruta raíz: `/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/`

Estructura de primer nivel:
```
Clasificación Final/
├── Ansiedad/
│   ├── m4a/{0,1}
│   ├── mp4/{0,1}
│   ├── wav/{0,1}
│   ├── specs/{cqt,gammatone,log-mel,pcen}/{2s,3s,4s,10s}/{0,1}
│   └── embeddings_v2/
│       ├── manifest.json
│       └── ansiedad/{class_0,class_1}/{wav2vec2-large-robust,xlsr-300m,whisper-large-encoder,wavlm-large,hubert-large,xlsr-53}
├── Depresión/  (misma estructura, carpeta "depresion" en vez de "ansiedad" dentro de embeddings_v2)
└── DatosPreprocesados/
    └── Muestras_Text(corregidos)/   (80 .txt + texto.zip)
```

Convención de clases confirmada por el propio log de ejecución (líneas `Cargados: N pacientes | ... clase0=X | clase1=Y`): **clase 0 = sin la condición (controles), clase 1 = con la condición (casos)**. Esto es válido para ambos, Ansiedad y Depresión.

### 1.1 Conteo de archivos de audio/video crudos (por clase)

**Ansiedad**

| Formato | Clase 0 (archivos) | Clase 1 (archivos) | Total | Clase 0 (muestras únicas) | Clase 1 (muestras únicas) |
|---|---|---|---|---|---|
| wav | 180 | 57 | 237 | 61 | 19 |
| m4a | 44 | 10 | 54 | 44 | 10 |
| mp4 | 61 | 19 | 80 | 61 | 19 |

**Depresión**

| Formato | Clase 0 (archivos) | Clase 1 (archivos) | Total | Clase 0 (muestras únicas) | Clase 1 (muestras únicas) |
|---|---|---|---|---|---|
| wav | 174 | 63 | 237 | 58 | 21 |
| m4a | 42 | 12 | 54 | 42 | 12 |
| mp4 | 59 | 21 | 80 | 59 | 21 |

Notas sobre el conteo:
- **wav** contiene 3 variantes por muestra: el archivo base (`NNN.wav`), uno recortado (`NNN_recortado.wav`) y uno recortado+denoised (`NNN_recortado_denoised.wav`). Por eso el conteo de archivos wav es ~3× el número de muestras únicas (61 muestras × 3 = 183 ≈ 180 en Ansiedad clase 0, pequeñas discrepancias por archivos faltantes en alguna variante).
- **mp4** coincide en cantidad de muestras únicas con wav (61/19 en Ansiedad, 59/21 en Depresión) — es decir, el mp4 parece ser la fuente completa de la que se extrajo el wav.
- **m4a** es un subconjunto menor (44/10 en Ansiedad, 42/12 en Depresión) — probablemente grabaciones de audio-only disponibles solo para parte de los pacientes (posiblemente cuando no había video, o formato de grabación alternativo).
- Nombres de archivo son numéricos con padding a 3 dígitos (`001`...`080`), consistentes con los IDs usados en `DatosPreprocesados`.

**Balance de clases (dataset final, sobre muestras únicas ≈ pacientes):**
- Ansiedad: 61 controles (clase 0) vs 19 casos (clase 1) → ratio ≈ 3.2:1, fuertemente desbalanceado hacia la clase negativa. Total ≈ 80 muestras (coincide con manifest: 79 audio_ids únicos, ver 1.3).
- Depresión: 58-60 controles (clase 0) vs 21-63 casos según la fuente (ver discrepancia abajo) — el manifest de embeddings (fuente más confiable, ver 1.3) da 58 controles vs 21 casos → ratio ≈ 2.76:1. Total = 79 pacientes.

**Discrepancia detectada:** el conteo de archivos wav "en bruto" para Depresión da clase1=63 (57 en Ansiedad), pero al contar *muestras únicas* (quitando sufijos `_recortado`/`_recortado_denoised`) se obtiene clase1=21 para Depresión y clase1=19 para Ansiedad — coincidiendo exactamente con lo reportado por los logs de ejecución (`clase1=21` en Depresión, `clase1=19` en Ansiedad) y con el manifest.json de embeddings_v2. **Se recomienda usar el conteo de muestras únicas (pacientes) como cifra oficial de tamaño de dataset, no el conteo bruto de archivos wav.**

### 1.2 Espectrogramas generados (specs/{cqt,gammatone,log-mel,pcen})

Los 4 tipos de espectrograma (CQT, Gammatone, log-Mel, PCEN) se generaron de forma idéntica en cantidad — es decir, para cada segmento de audio se generaron las 4 representaciones en paralelo. Los espectrogramas están organizados por duración de ventana/segmento: 2s, 3s, 4s, 10s.

**Ansiedad — número de imágenes .png por tipo de spec, ventana y clase** (idéntico para cqt, gammatone, log-mel y pcen):

| Ventana | Clase 0 | Clase 1 | Total |
|---|---|---|---|
| 2s | 9,592 | 3,274 | 12,866 |
| 3s | 6,371 | 2,174 | 8,545 |
| 4s | 4,759 | 1,624 | 6,383 |
| 10s | 940 | 319 | 1,259 |

Total por tipo de spec (Ansiedad): 29,053 imágenes × 4 tipos = **116,212 espectrogramas en total** para Ansiedad.

**Depresión — número de imágenes .png por tipo de spec, ventana y clase** (idéntico para cqt, gammatone, log-mel y pcen):

| Ventana | Clase 0 | Clase 1 | Total |
|---|---|---|---|
| 2s | 9,314 | 3,549 | 12,863 |
| 3s | 6,187 | 2,357 | 8,544 |
| 4s | 4,622 | 1,761 | 6,383 |
| 10s | 912 | 347 | 1,259 |

Total por tipo de spec (Depresión): 29,049 imágenes × 4 tipos = **116,196 espectrogramas en total** para Depresión.

Observaciones:
- El desbalance de clases se mantiene consistente a nivel de segmento (~74-75% clase 0 / ~25-26% clase 1 en ventanas cortas), reflejando el desbalance a nivel de paciente.
- El número de segmentos por ventana decrece según aumenta la duración de la ventana (más segmentos con ventanas cortas de 2s, menos con ventanas de 10s), como es esperable de un slicing con stride fijo sobre audios de duración total similar.
- Los conteos entre Ansiedad y Depresión son muy similares en total (~29,050 c/u), sugiriendo que ambos parten de una base de audio similar en duración total agregada, pese a tener distinto balance de clases.

### 1.3 Embeddings v2 (embeddings_v2/) y manifest.json

Cada condición tiene embeddings pre-calculados con **6 modelos pre-entrenados de habla** (todos basados en transformers de audio self-supervised): `wav2vec2-large-robust`, `xlsr-300m`, `xlsr-53`, `whisper-large-encoder`, `wavlm-large`, `hubert-large`.

**Conteo de archivos de embedding por modelo/clase (recuento de directorios `segment_N/embedding.json` — vía `find -type f`):**

| Condición | Clase 0 (archivos por modelo) | Clase 1 (archivos por modelo) |
|---|---|---|
| Ansiedad | 3,911 | 1,341 |
| Depresión | 3,799 | 1,453 |

(Esta cifra se repite igual para cada uno de los 6 modelos, ya que todos procesan los mismos segmentos de audio.)

**Resumen de manifest.json (parseado con Python, no volcado completo por su tamaño ~13MB c/u):**

| Campo | Ansiedad | Depresión |
|---|---|---|
| Tamaño del archivo | 12,939,152 bytes (~12.3 MB) | 13,191,248 bytes (~12.6 MB) |
| Total de registros (list) | 31,512 | 31,512 |
| Modelos (6, ~5,252 registros c/u) | xlsr-300m, xlsr-53, whisper-large-encoder, wav2vec2-large-robust, wavlm-large, hubert-large | (idénticos) |
| Registros clase 0 | 23,466 | 22,794 |
| Registros clase 1 | 8,046 | 8,718 |
| audio_ids únicos (pacientes) | 79 (60 clase 0 / 19 clase 1) | 79 (58 clase 0 / 21 clase 1) |
| Duración de segmento predominante | 5.0s (30,744 de 31,512 registros); resto son segmentos residuales de duración variable (~1-5s, el "resto" al final de cada audio) | igual patrón |
| Segmentos por audio_id (min/max/media) | 138 / 942 / 398.9 | 138 / 942 / 398.9 |
| Campos del registro | `condition`, `audio_id`, `class_id`, `class_label`, `model`, `segment_id`, `start_time`, `end_time`, `file` | igual |

Ejemplo de un registro (primero del manifest de Ansiedad):
```json
{
  "condition": "ansiedad",
  "audio_id": "002_recortado_denoised",
  "class_id": "0",
  "class_label": "class_0",
  "model": "xlsr-300m",
  "segment_id": 0,
  "start_time": 0.0,
  "end_time": 5.0,
  "file": ".../Ansiedad/embeddings_v2/ansiedad/class_0/xlsr-300m/002_recortado_denoised/segment_0/embedding.json"
}
```

Nota: `audio_id` usa el archivo `_recortado_denoised` como fuente (i.e., los embeddings v2 se calcularon sobre el audio ya recortado y con reducción de ruido, no sobre el wav crudo). Esto confirma la cadena de preprocesamiento: **wav crudo → recorte (recortado) → denoising (recortado_denoised) → segmentación en ventanas de 5s → embeddings**.

**Confirmación de tamaño oficial del dataset propio (a nivel paciente):** 79 pacientes en ambas condiciones — Ansiedad: 60 controles / 19 casos; Depresión: 58 controles / 21 casos. Este es el N a reportar como tamaño muestral del dataset propio en el paper.

### 1.4 DatosPreprocesados / Muestras_Text(corregidos) — transcripciones

- Carpeta: `Clasificación Final/DatosPreprocesados/Muestras_Text(corregidos)/`
- Contiene **80 archivos .txt** (uno de ellos con nombre especial `066_1054544943(no tenia wav).txt`, indicando explícitamente que a esa muestra le faltaba el wav correspondiente) + 1 archivo `texto.zip` (backup/comprimido, no se extrajo, solo lectura de metadatos).
- Nomenclatura: `NNN_<ID_numerico_largo>.txt` (ej. `001_1053791621.txt`), donde NNN es el mismo índice de 3 dígitos usado en los audios (001-080), y el segundo número parece ser un ID de paciente/sesión.

**Contenido de los .txt (verificado leyendo 3 archivos completos):** Son transcripciones de una entrevista semiestructurada tipo cuestionario, con **7-8 preguntas numeradas** (1 a 7 u 8) y la respuesta transcrita del paciente en texto corrido (sin timestamps, sin metadatos de audio, solo texto plano separado por tabulador `numero\trespuesta`). Las preguntas no aparecen explícitas en el .txt, pero por el contenido de las respuestas se infieren preguntas del tipo: actividades diarias de la última semana, planes para las próximas semanas, planes de Navidad/fin de año, qué le hace feliz, qué le pone triste, etc. — un patrón temático muy similar al protocolo de entrevista DAIC-WOZ (ver sección DATA-DAIC). El contenido está en español, con muletillas y disfluencias conservadas (transcripción fiel, "corregida" probablemente en el sentido de corrección ortográfica/de errores de transcripción automática, no de contenido).

### 1.5 Resumen de la cadena de preprocesamiento (m4a/mp4/wav)

En base a los conteos y a los nombres de archivo:
1. **m4a** / **mp4**: formatos de grabación cruda de origen (probablemente exportados de una app de grabación de voz — m4a — o de videollamada/videograbación — mp4). El mp4 está disponible para todas las 80 muestras (61/19 Ansiedad, 59/21 Depresión); el m4a solo para un subconjunto (44/10 y 42/12 respectivamente), sugiriendo que hay dos fuentes/modalidades de captura distintas según el caso.
2. **wav**: audio extraído del mp4 (o m4a) y luego procesado en 2 pasos adicionales: recorte de silencios/segmentos no útiles (`_recortado`) y reducción de ruido (`_recortado_denoised`).
3. **specs**: espectrogramas (CQT, Gammatone, log-Mel, PCEN) generados a partir del wav final (`_recortado_denoised`), segmentado en ventanas de 2s/3s/4s/10s.
4. **embeddings_v2**: embeddings de 6 modelos de habla pre-entrenados, calculados sobre el mismo wav final segmentado en ventanas de 5s.

---

### 1.6 Dataset público DATA-DAIC (DAIC-WOZ) — para contexto/comparación

Ruta: `/home/ci2dt2-ai/Proyectos/Psiquiatria/DATA-DAIC/`

- **189 carpetas de pacientes** (`ls DATA-DAIC | grep _P | wc -l` = 189; nota: `ls | wc -l` total da 193, la diferencia de 4 corresponde a archivos/carpetas sueltas que no siguen el patrón `_P`, no verificadas en detalle por estar fuera del alcance pedido).
- Rango de IDs de paciente observado: desde `300_P` hasta `492_P` (numeración consistente con el dataset público DAIC-WOZ/E-DAIC, que usa IDs de paciente en el rango 300-492).
- Cada carpeta `NNN_P/` contiene, de forma consistente, los siguientes archivos (confirmado en `300_P`, `320_P`, `350_P`, `400_P` sin leer contenido):
  - `NNN_AUDIO.wav` — audio de la entrevista (~20 MB por paciente en el caso de 300_P)
  - `NNN_CLNF_AUs.txt` — Action Units faciales (CLNF)
  - `NNN_CLNF_features.txt` — features faciales 2D
  - `NNN_CLNF_features3D.txt` — features faciales 3D
  - `NNN_CLNF_gaze.txt` — dirección de mirada
  - `NNN_CLNF_hog.txt` **o** `NNN_CLNF_hog.bin` — descriptores HOG (formato varía: `.txt` en `300_P`, `.bin` en `320_P`/`350_P`/`400_P` — posible cambio de formato de exportación entre lotes de pacientes, a verificar si se usa este feature)
  - `NNN_CLNF_pose.txt` — pose de cabeza
  - `NNN_COVAREP.csv` — features acústicos COVAREP
  - `NNN_FORMANT.csv` — formantes
  - `NNN_TRANSCRIPT.csv` — transcripción con timestamps (entrevistador + paciente)
- No se leyó contenido de ningún archivo de DATA-DAIC (solo se listó estructura), conforme a lo solicitado.

---

## 2. Línea de tiempo y resultados agregados de los logs de ejecución

Archivos analizados (solo lectura, sin modificar):
- `/home/ci2dt2-ai/Proyectos/Psiquiatria/log_general.txt` — 345,719 bytes, 5,305 líneas, mtime `jun 4 11:17`.
- `/home/ci2dt2-ai/Proyectos/Psiquiatria/log_ejecucion_final.txt` — 209,680 bytes, 3,224 líneas, mtime `jun 8 21:54`.

Ambos son logs de `nohup` de un runner que ejecuta "en fila india" (secuencialmente) dos scripts de deep learning (`dl_script1_feature_reduction.py` y `dl_script2_architectures_fixed.py`, ubicados en `Ansiedad/Scripts_Embedding_{viejo,nuevo}/` y `Depresion/Scripts_Embedding_{viejo,nuevo}/`) sobre 4 combinaciones de {condición, versión de embeddings}: Depresión-Viejos, Depresión-Nuevos, Ansiedad-Viejos, Ansiedad-Nuevos. "Viejos" = `embeddings/` (una versión anterior, pooled) y "Nuevos" = `embeddings_v2/` (la descrita en la sección 1.3, con manifest.json y 6 modelos).

**Búsqueda de errores:** se ejecutó `grep -n -i "error|traceback|exception|❌|warn|fallo|fail"` sobre ambos archivos completos → **0 coincidencias en ambos archivos**. No se registraron errores, excepciones ni advertencias en ninguna de las corridas capturadas en estos logs.

### 2.1 Estructura de pipelines identificada

Cada línea `->` marca el inicio de un bloque {condición × versión de embeddings}. Dentro de cada bloque se ejecutan, en orden, `dl_script1` y luego `dl_script2`:

- **dl_script1_feature_reduction.py**: agrupa los embeddings por paciente (pooling) y evalúa, para cada uno de los 6 modelos de embedding (xlsr-300m, xlsr-53, whisper-large-encoder, wav2vec2-large-robust, wavlm-large, hubert-large), 6 técnicas de reducción/balanceo de features sobre un clasificador fijo: `A_raw` (features crudas), `B_pca` (PCA), `C_sfs` (Sequential Feature Selection — la más lenta, decenas a >600 min por combinación), `D_raw_mixup`, `E_pca_smote`, `F_pca_adasyn` (técnicas de aumento de datos / balanceo de clases). CV = 10 repeticiones × 5 folds = 50 evaluaciones por configuración. 6 modelos × 6 experimentos = 36 resultados `✅` por bloque completo (excepto en un bloque de Ansiedad-Nuevos donde `C_sfs` se excluyó del listado de experimentos, dando 30 resultados).
- **dl_script2_architectures_fixed.py**: reconstruye secuencias cronológicas de embeddings por paciente y evalúa 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) × 4 variantes de datos (raw, pca, raw_mixup, pca_mixup) × 6 modelos de embedding = 96 resultados `✅` por bloque completo. Al final de cada corrida completa de dl_script2 se imprime un ranking propio "TOP 20 CONFIGURACIONES (por F1-macro)" (ver sección 3.4).

Orden real de ejecución encontrado (marcadores `->`):

**log_general.txt:**
1. `Depresión: Embeddings Viejo` → dl_script1 (36/36 ✅) → dl_script2 (96/96 ✅)
2. `Depresión: Embeddings Nuevos` → dl_script1 (36/36 ✅) → dl_script2 (96/96 ✅)
3. `Ansiedad: Embeddings Viejos` → dl_script1 (36/36 ✅) → dl_script2 (**59/96 ✅ — interrumpido**, el log termina abruptamente a mitad del modelo `wav2vec2-large-robust` sin línea de cierre "REPRODUCCIÓN COMPLETADA")

**log_ejecucion_final.txt** (corrida separada, 4 días después según mtimes):
1. `Ansiedad: Embeddings Viejos` → dl_script2 (96/96 ✅, **repite y completa** lo que quedó inconcluso en log_general.txt; no incluye dl_script1 porque ya se había completado en la corrida anterior)
2. `Ansiedad: Embeddings Nuevos` → dl_script1 (30/30 ✅, sin `C_sfs`) → dl_script2 (96/96 ✅) → termina con "📊 TOP 20 CONFIGURACIONES", "📄 Summary", "⏱️ Tiempo total: 0.7 h" y "¡REPRODUCCIÓN COMPLETADA!"

**Total de resultados `✅` extraídos y tabulados: 581** (359 en log_general.txt + 222 en log_ejecucion_final.txt).

### 2.2 Cronología y duración por fase

No hay timestamps con fecha dentro de los logs (solo `HH:MM:SS`), por lo que las fechas se infieren de la fecha de modificación (mtime) de cada archivo y de la consistencia interna de los acumulados de tiempo (`⏱️ Tiempo total`).

| # | Archivo | Bloque | Script | Inicio | Fin | Duración reportada |
|---|---|---|---|---|---|---|
| 1 | log_general.txt | Depresión Viejos | dl_script1 | 21:52:02 | ~21:07 (día sig.) | 1395.3 min (23.3 h) |
| 2 | log_general.txt | Depresión Viejos | dl_script2 | 21:08:27 | 14:10:43 | 17.0 h |
| 3 | log_general.txt | Depresión Nuevos | dl_script1 | 14:10:46 | ~00:04 | 594.2 min (9.9 h) |
| 4 | log_general.txt | Depresión Nuevos | dl_script2 | 00:05:21 | 00:52:25 | 0.8 h (47 min) |
| 5 | log_general.txt | Ansiedad Viejos | dl_script1 | 00:52:26 | 10:45:35 | 593.2 min (9.9 h) |
| 6 | log_general.txt | Ansiedad Viejos | dl_script2 | 10:45:59 | **11:17:37 (interrumpido, sin cierre)** | incompleto (59/96) |
| 7 | log_ejecucion_final.txt | Ansiedad Viejos | dl_script2 (redo) | 20:18:27 | 21:04:04 | 0.8 h (45 min) |
| 8 | log_ejecucion_final.txt | Ansiedad Nuevos | dl_script1 | 21:04:06 | 21:09:25 | 5.3 min (0.1 h) |
| 9 | log_ejecucion_final.txt | Ansiedad Nuevos | dl_script2 | 21:09:25 | 21:54:05 | 0.7 h (42 min) |

**Tiempo total acumulado en log_general.txt** (fases 1-6, hasta la interrupción): ≈ 61 horas (~2.5 días) de cómputo continuo, coherente con un salto de mtime de aproximadamente 3 días desde el inicio del `nohup` (`INICIANDO REPRODUCCIÓN...`) hasta `jun 4 11:17`.

**Tiempo total en log_ejecucion_final.txt** (fases 7-9): 1 h 36 min, coherente con el mtime `jun 8 21:54` prácticamente idéntico al timestamp final `21:54:05` del propio log.

**Observaciones relevantes para el paper:**
- Hay un **gap de ~4 días** entre el corte de `log_general.txt` (≈ jun 4) y el inicio de `log_ejecucion_final.txt` (≈ jun 8), sugiriendo que la ejecución fue retomada manualmente tras la interrupción.
- El experimento `C_sfs` (Sequential Feature Selection) dentro de `dl_script1` es, con diferencia, el más costoso computacionalmente: sus tiempos individuales van de ~53 min hasta >600 min (10 h) por combinación modelo×condición, mientras el resto de experimentos (`A_raw`, `B_pca`, `D_raw_mixup`, `E_pca_smote`, `F_pca_adasyn`) toman típicamente 0.1-1.2 min cada uno. Esto explica la mayor parte de las 23.3h y 9.9h de las fases dl_script1.
- Se observa una **anomalía de tiempos** en `dl_script2`: la fase "Depresión Viejos dl_script2" tomó 17.0 h para 96 configuraciones, mientras que "Depresión Nuevos dl_script2" (mismo script, mismo número de configuraciones) tomó solo 0.8 h, y "Ansiedad Viejos dl_script2" (re-ejecutado en log_ejecucion_final.txt) tomó 0.8h también. No hay errores en el log que expliquen esta diferencia de ~20x; podría deberse a contención de GPU/disco durante esa corrida específica, y es un punto a mencionar con cautela en el paper (no se encontró causa explícita en los logs).
- No se registró ningún archivo de log con marca de fecha explícita (`YYYY-MM-DD`), por lo que las fechas exactas de cada fase deben tomarse como aproximadas, derivadas de mtimes.

### 2.3 Configuración de carga de datos por bloque (tal como aparece en los logs)

| Bloque | Pacientes | clase0 | clase1 | dim / shape embeddings |
|---|---|---|---|---|
| Depresión, embeddings viejos (dl_script1) | 79 | 58 | 21 | dim=1024 |
| Depresión, embeddings viejos (dl_script2) | 79 | 58 | 21 | shape=(79, 157, 1024) |
| Depresión, embeddings nuevos (dl_script1) | 79 | 58 | 21 | dim=1024 |
| Depresión, embeddings nuevos (dl_script2) | 79 | 58 | 21 | shape=(79, 157, 1024) |
| Ansiedad, embeddings viejos (dl_script1) | 79 | 60 | 19 | dim=1024 |
| Ansiedad, embeddings viejos (dl_script2) | 79 | 60 | 19 | shape=(79, 157, 1024) |
| Ansiedad, embeddings nuevos (dl_script1/2) | 79 | 60 | 19 | dim=1024 / shape=(79,157,1024) |

Estos conteos por paciente (79 totales, 58/21 en Depresión, 60/19 en Ansiedad) **coinciden exactamente** con los del manifest.json de embeddings_v2 descritos en la sección 1.3, confirmando la consistencia del dataset a través de todo el pipeline (de archivos crudos → specs → embeddings → entrenamiento).

### 2.4 Resultados destacados (mejor configuración por fase, según el propio TOP-20 del log)

Cada corrida completa de `dl_script2` termina con un bloque propio "📊 TOP 20 CONFIGURACIONES (por F1-macro)" generado por el script (agregando sobre 50 evaluaciones = 10 repeats × 5 folds). Se transcribe la fila #1 (mejor configuración) de cada una de las 4 corridas completas de dl_script2 encontradas:

| Bloque | Mejor configuración | F1-macro (mean) | std | F1-minoría | AUC |
|---|---|---|---|---|---|
| Depresión, Viejos, dl_script2 | wav2vec2-large-robust + BiLSTM + raw | 0.7549 | 0.1049 | 0.6782 | 0.7396 |
| Depresión, Nuevos, dl_script2 | wav2vec2-large-robust + BiLSTM + raw | 0.7638 | 0.1123 | 0.6888 | 0.7459 |
| Ansiedad, Viejos, dl_script2 | xlsr-300m + CNN1D + raw | 0.7151 | 0.1199 | 0.6048 | 0.6142 |
| Ansiedad, Nuevos, dl_script2 | xlsr-300m + CNN1D + raw | 0.7115 | 0.1178 | 0.5992 | 0.6207 |

Patrón consistente: para **Depresión**, la mejor arquitectura es **BiLSTM sobre embeddings de wav2vec2-large-robust en crudo (raw)**; para **Ansiedad**, la mejor es **CNN1D sobre embeddings de xlsr-300m en crudo (raw)**. En ambas condiciones, los resultados con embeddings "nuevos" (v2) son muy similares o ligeramente mejores que con embeddings "viejos", y en ambas la variante "raw" (sin reducción/balanceo) supera a las variantes con PCA o mixup en la mejor configuración global.

Mejor resultado de `dl_script1` (reducción de features, clasificador fijo) por fase, calculado sobre los 581 resultados parseados:

| Bloque | Mejor configuración | F1-macro | AUC |
|---|---|---|---|
| Depresión, Viejos, dl_script1 | A_raw + wav2vec2-large-robust | 0.757±0.113 | 0.694 |
| Depresión, Nuevos, dl_script1 | A_raw + wav2vec2-large-robust | 0.777±0.128 | 0.740 |
| Ansiedad, Viejos, dl_script1 | D_raw_mixup + xlsr-53 | 0.721±0.099 | 0.655 |
| Ansiedad, Nuevos, dl_script1 | D_raw_mixup + wav2vec2-large-robust | 0.723±0.085 | 0.668 |

### 2.5 Tablas completas de resultados ✅ (las 581 filas, organizadas por bloque)

A continuación se listan **todos** los resultados con marca `✅` encontrados en ambos logs (fuente de verdad), agrupados por condición → versión de embeddings → script, en el mismo orden en que aparecen en el log (por modelo de embedding, y dentro de cada modelo por arquitectura/experimento). Cada tabla incluye F1-macro, desviación estándar (sobre 10 repeats), F1 de la clase minoritaria, AUC, tiempo de cómputo de esa configuración y la hora (HH:MM:SS) del log en que se registró.

#### Depresión: Embeddings Viejo | embeddings_viejos | dl_script1  (archivo: `log_general.txt`, 36 resultados ✅)

- Script: dl_script1_feature_reduction.py — reduccion/seleccion de features sobre embeddings pooled por paciente (clasificador fijo) con 6 tecnicas: A_raw, B_pca, C_sfs, D_raw_mixup, E_pca_smote, F_pca_adasyn
- Carga de datos: `21:52:24 [INFO]    Cargados: 79 pacientes | dim=1024 | clase0=58 | clase1=21`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|
| 1 | A_raw | xlsr-300m | 0.662 | 0.110 | 0.578 | 0.595 | 0.1 min | 21:52:30 |
| 2 | B_pca | xlsr-300m | 0.617 | 0.137 | 0.559 | 0.542 | 0.4 min | 21:52:53 |
| 3 | C_sfs | xlsr-300m | 0.643 | 0.144 | 0.577 | 0.565 | 56.4 min | 22:49:20 |
| 4 | D_raw_mixup | xlsr-300m | 0.660 | 0.124 | 0.573 | 0.569 | 0.1 min | 22:49:26 |
| 5 | E_pca_smote | xlsr-300m | 0.642 | 0.145 | 0.570 | 0.580 | 0.4 min | 22:49:53 |
| 6 | F_pca_adasyn | xlsr-300m | 0.620 | 0.170 | 0.570 | 0.568 | 0.4 min | 22:50:17 |
| 7 | A_raw | xlsr-53 | 0.636 | 0.152 | 0.568 | 0.555 | 0.1 min | 22:50:45 |
| 8 | B_pca | xlsr-53 | 0.583 | 0.154 | 0.538 | 0.524 | 0.4 min | 22:51:08 |
| 9 | C_sfs | xlsr-53 | 0.612 | 0.181 | 0.574 | 0.563 | 73.7 min | 00:04:49 |
| 10 | D_raw_mixup | xlsr-53 | 0.661 | 0.114 | 0.572 | 0.573 | 0.1 min | 00:04:56 |
| 11 | E_pca_smote | xlsr-53 | 0.609 | 0.162 | 0.557 | 0.541 | 0.4 min | 00:05:22 |
| 12 | F_pca_adasyn | xlsr-53 | 0.618 | 0.155 | 0.559 | 0.562 | 0.4 min | 00:05:49 |
| 13 | A_raw | whisper-large-encoder | 0.665 | 0.124 | 0.575 | 0.561 | 0.1 min | 00:06:27 |
| 14 | B_pca | whisper-large-encoder | 0.599 | 0.150 | 0.550 | 0.555 | 0.4 min | 00:06:51 |
| 15 | C_sfs | whisper-large-encoder | 0.634 | 0.167 | 0.578 | 0.575 | 92.2 min | 01:39:04 |
| 16 | D_raw_mixup | whisper-large-encoder | 0.664 | 0.158 | 0.590 | 0.575 | 0.1 min | 01:39:11 |
| 17 | E_pca_smote | whisper-large-encoder | 0.655 | 0.165 | 0.585 | 0.581 | 0.5 min | 01:39:38 |
| 18 | F_pca_adasyn | whisper-large-encoder | 0.646 | 0.168 | 0.579 | 0.573 | 0.4 min | 01:40:05 |
| 19 | A_raw | wav2vec2-large-robust | 0.757 | 0.113 | 0.667 | 0.694 | 0.1 min | 01:40:31 |
| 20 | B_pca | wav2vec2-large-robust | 0.692 | 0.147 | 0.622 | 0.634 | 0.4 min | 01:40:54 |
| 21 | C_sfs | wav2vec2-large-robust | 0.729 | 0.145 | 0.656 | 0.678 | 109.6 min | 03:30:30 |
| 22 | D_raw_mixup | wav2vec2-large-robust | 0.745 | 0.124 | 0.659 | 0.698 | 0.1 min | 03:30:36 |
| 23 | E_pca_smote | wav2vec2-large-robust | 0.709 | 0.144 | 0.632 | 0.666 | 0.4 min | 03:31:02 |
| 24 | F_pca_adasyn | wav2vec2-large-robust | 0.717 | 0.113 | 0.633 | 0.658 | 0.4 min | 03:31:29 |
| 25 | A_raw | wavlm-large | 0.697 | 0.130 | 0.616 | 0.636 | 0.1 min | 03:32:02 |
| 26 | B_pca | wavlm-large | 0.677 | 0.164 | 0.608 | 0.622 | 0.5 min | 03:32:35 |
| 27 | C_sfs | wavlm-large | 0.682 | 0.143 | 0.602 | 0.623 | 444.3 min | 10:56:51 |
| 28 | D_raw_mixup | wavlm-large | 0.680 | 0.131 | 0.599 | 0.634 | 0.3 min | 10:57:11 |
| 29 | E_pca_smote | wavlm-large | 0.675 | 0.167 | 0.618 | 0.627 | 1.1 min | 10:58:15 |
| 30 | F_pca_adasyn | wavlm-large | 0.641 | 0.176 | 0.593 | 0.589 | 1.2 min | 10:59:27 |
| 31 | A_raw | hubert-large | 0.654 | 0.161 | 0.590 | 0.591 | 0.3 min | 11:00:31 |
| 32 | B_pca | hubert-large | 0.629 | 0.175 | 0.585 | 0.585 | 0.9 min | 11:01:27 |
| 33 | C_sfs | hubert-large | 0.622 | 0.151 | 0.573 | 0.562 | 603.3 min | 21:04:45 |
| 34 | D_raw_mixup | hubert-large | 0.662 | 0.160 | 0.594 | 0.602 | 0.4 min | 21:05:06 |
| 35 | E_pca_smote | hubert-large | 0.656 | 0.152 | 0.600 | 0.599 | 1.1 min | 21:06:11 |
| 36 | F_pca_adasyn | hubert-large | 0.650 | 0.155 | 0.577 | 0.567 | 1.2 min | 21:07:23 |


#### Depresión: Embeddings Viejo | embeddings_viejos | dl_script2  (archivo: `log_general.txt`, 96 resultados ✅)

- Script: dl_script2_architectures_fixed.py — 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) sobre secuencias de embeddings, cada una con 4 variantes de datos: raw, pca, raw_mixup, pca_mixup
- Carga de datos: `21:09:16 [INFO]    Cargados: 79 pacientes | shape=(79, 157, 1024) | clase0=58 | clase1=21`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Arquitectura | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN1D | raw | xlsr-300m | 0.705 | 0.152 | 0.625 | 0.638 | 8.7 min | 21:18:01 |
| 2 | CNN1D | pca | xlsr-300m | 0.707 | 0.132 | 0.621 | 0.638 | 21.5 min | 21:39:31 |
| 3 | CNN1D | raw_mixup | xlsr-300m | 0.644 | 0.173 | 0.569 | 0.569 | 10.8 min | 21:50:19 |
| 4 | CNN1D | pca_mixup | xlsr-300m | 0.673 | 0.173 | 0.601 | 0.607 | 21.8 min | 22:12:08 |
| 5 | BiLSTM | raw | xlsr-300m | 0.661 | 0.136 | 0.596 | 0.608 | 9.5 min | 22:21:35 |
| 6 | BiLSTM | pca | xlsr-300m | 0.658 | 0.138 | 0.589 | 0.598 | 22.1 min | 22:43:43 |
| 7 | BiLSTM | raw_mixup | xlsr-300m | 0.662 | 0.122 | 0.587 | 0.605 | 12.7 min | 22:56:25 |
| 8 | BiLSTM | pca_mixup | xlsr-300m | 0.622 | 0.144 | 0.570 | 0.579 | 22.3 min | 23:18:44 |
| 9 | BiGRU | raw | xlsr-300m | 0.676 | 0.118 | 0.603 | 0.640 | 10.2 min | 23:28:58 |
| 10 | BiGRU | pca | xlsr-300m | 0.665 | 0.147 | 0.598 | 0.615 | 22.3 min | 23:51:15 |
| 11 | BiGRU | raw_mixup | xlsr-300m | 0.666 | 0.145 | 0.604 | 0.616 | 12.7 min | 00:03:57 |
| 12 | BiGRU | pca_mixup | xlsr-300m | 0.631 | 0.188 | 0.588 | 0.587 | 22.4 min | 00:26:24 |
| 13 | CNN_BiGRU | raw | xlsr-300m | 0.624 | 0.161 | 0.572 | 0.571 | 9.0 min | 00:35:26 |
| 14 | CNN_BiGRU | pca | xlsr-300m | 0.638 | 0.139 | 0.583 | 0.598 | 22.3 min | 00:57:42 |
| 15 | CNN_BiGRU | raw_mixup | xlsr-300m | 0.620 | 0.144 | 0.573 | 0.587 | 13.9 min | 01:11:38 |
| 16 | CNN_BiGRU | pca_mixup | xlsr-300m | 0.640 | 0.136 | 0.567 | 0.597 | 22.3 min | 01:33:55 |
| 17 | CNN1D | raw | xlsr-53 | 0.580 | 0.209 | 0.544 | 0.550 | 7.2 min | 01:41:58 |
| 18 | CNN1D | pca | xlsr-53 | 0.597 | 0.220 | 0.566 | 0.548 | 21.6 min | 02:03:37 |
| 19 | CNN1D | raw_mixup | xlsr-53 | 0.589 | 0.214 | 0.562 | 0.562 | 10.6 min | 02:14:13 |
| 20 | CNN1D | pca_mixup | xlsr-53 | 0.666 | 0.172 | 0.598 | 0.597 | 21.9 min | 02:36:07 |
| 21 | BiLSTM | raw | xlsr-53 | 0.651 | 0.156 | 0.592 | 0.602 | 8.8 min | 02:44:54 |
| 22 | BiLSTM | pca | xlsr-53 | 0.634 | 0.162 | 0.575 | 0.583 | 22.3 min | 03:07:09 |
| 23 | BiLSTM | raw_mixup | xlsr-53 | 0.647 | 0.144 | 0.577 | 0.609 | 13.8 min | 03:20:56 |
| 24 | BiLSTM | pca_mixup | xlsr-53 | 0.654 | 0.128 | 0.585 | 0.596 | 22.6 min | 03:43:34 |
| 25 | BiGRU | raw | xlsr-53 | 0.634 | 0.153 | 0.578 | 0.589 | 8.3 min | 03:51:52 |
| 26 | BiGRU | pca | xlsr-53 | 0.634 | 0.142 | 0.557 | 0.564 | 22.4 min | 04:14:18 |
| 27 | BiGRU | raw_mixup | xlsr-53 | 0.606 | 0.164 | 0.554 | 0.562 | 12.9 min | 04:27:14 |
| 28 | BiGRU | pca_mixup | xlsr-53 | 0.630 | 0.143 | 0.567 | 0.567 | 22.3 min | 04:49:35 |
| 29 | CNN_BiGRU | raw | xlsr-53 | 0.596 | 0.181 | 0.567 | 0.558 | 8.6 min | 04:58:11 |
| 30 | CNN_BiGRU | pca | xlsr-53 | 0.638 | 0.147 | 0.574 | 0.596 | 22.6 min | 05:20:45 |
| 31 | CNN_BiGRU | raw_mixup | xlsr-53 | 0.615 | 0.143 | 0.553 | 0.563 | 13.8 min | 05:34:33 |
| 32 | CNN_BiGRU | pca_mixup | xlsr-53 | 0.652 | 0.161 | 0.580 | 0.595 | 22.5 min | 05:57:00 |
| 33 | CNN1D | raw | whisper-large-encoder | 0.629 | 0.185 | 0.556 | 0.545 | 8.1 min | 06:06:19 |
| 34 | CNN1D | pca | whisper-large-encoder | 0.671 | 0.162 | 0.589 | 0.585 | 38.2 min | 06:44:31 |
| 35 | CNN1D | raw_mixup | whisper-large-encoder | 0.637 | 0.201 | 0.579 | 0.570 | 13.1 min | 06:57:40 |
| 36 | CNN1D | pca_mixup | whisper-large-encoder | 0.656 | 0.182 | 0.589 | 0.582 | 41.0 min | 07:38:41 |
| 37 | BiLSTM | raw | whisper-large-encoder | 0.663 | 0.144 | 0.591 | 0.589 | 10.0 min | 07:48:42 |
| 38 | BiLSTM | pca | whisper-large-encoder | 0.628 | 0.174 | 0.571 | 0.588 | 39.5 min | 08:28:09 |
| 39 | BiLSTM | raw_mixup | whisper-large-encoder | 0.671 | 0.143 | 0.597 | 0.589 | 14.9 min | 08:43:03 |
| 40 | BiLSTM | pca_mixup | whisper-large-encoder | 0.614 | 0.167 | 0.567 | 0.568 | 42.7 min | 09:25:42 |
| 41 | BiGRU | raw | whisper-large-encoder | 0.673 | 0.143 | 0.594 | 0.593 | 9.9 min | 09:35:36 |
| 42 | BiGRU | pca | whisper-large-encoder | 0.622 | 0.150 | 0.562 | 0.563 | 38.7 min | 10:14:16 |
| 43 | BiGRU | raw_mixup | whisper-large-encoder | 0.653 | 0.153 | 0.588 | 0.589 | 14.5 min | 10:28:44 |
| 44 | BiGRU | pca_mixup | whisper-large-encoder | 0.632 | 0.141 | 0.558 | 0.545 | 42.2 min | 11:10:57 |
| 45 | CNN_BiGRU | raw | whisper-large-encoder | 0.614 | 0.160 | 0.560 | 0.559 | 9.3 min | 11:20:15 |
| 46 | CNN_BiGRU | pca | whisper-large-encoder | 0.643 | 0.149 | 0.560 | 0.562 | 41.8 min | 12:02:03 |
| 47 | CNN_BiGRU | raw_mixup | whisper-large-encoder | 0.654 | 0.146 | 0.581 | 0.584 | 16.6 min | 12:18:39 |
| 48 | CNN_BiGRU | pca_mixup | whisper-large-encoder | 0.638 | 0.170 | 0.584 | 0.571 | 45.2 min | 13:03:52 |
| 49 | CNN1D | raw | wav2vec2-large-robust | 0.541 | 0.208 | 0.517 | 0.480 | 9.0 min | 13:13:41 |
| 50 | CNN1D | pca | wav2vec2-large-robust | 0.533 | 0.211 | 0.518 | 0.487 | 24.4 min | 13:38:04 |
| 51 | CNN1D | raw_mixup | wav2vec2-large-robust | 0.589 | 0.211 | 0.551 | 0.550 | 4.4 min | 13:42:28 |
| 52 | CNN1D | pca_mixup | wav2vec2-large-robust | 0.591 | 0.200 | 0.549 | 0.538 | 1.1 min | 13:43:36 |
| 53 | BiLSTM | raw | wav2vec2-large-robust | 0.755 | 0.105 | 0.678 | 0.740 | 0.5 min | 13:44:05 |
| 54 | BiLSTM | pca | wav2vec2-large-robust | 0.748 | 0.122 | 0.670 | 0.698 | 1.5 min | 13:45:35 |
| 55 | BiLSTM | raw_mixup | wav2vec2-large-robust | 0.734 | 0.110 | 0.657 | 0.703 | 0.8 min | 13:46:20 |
| 56 | BiLSTM | pca_mixup | wav2vec2-large-robust | 0.699 | 0.134 | 0.629 | 0.661 | 1.2 min | 13:47:32 |
| 57 | BiGRU | raw | wav2vec2-large-robust | 0.750 | 0.122 | 0.671 | 0.710 | 0.5 min | 13:48:01 |
| 58 | BiGRU | pca | wav2vec2-large-robust | 0.725 | 0.128 | 0.640 | 0.638 | 0.9 min | 13:48:52 |
| 59 | BiGRU | raw_mixup | wav2vec2-large-robust | 0.718 | 0.147 | 0.649 | 0.679 | 0.6 min | 13:49:28 |
| 60 | BiGRU | pca_mixup | wav2vec2-large-robust | 0.694 | 0.136 | 0.614 | 0.643 | 0.7 min | 13:50:08 |
| 61 | CNN_BiGRU | raw | wav2vec2-large-robust | 0.694 | 0.133 | 0.606 | 0.622 | 0.4 min | 13:50:32 |
| 62 | CNN_BiGRU | pca | wav2vec2-large-robust | 0.686 | 0.134 | 0.607 | 0.626 | 0.5 min | 13:51:03 |
| 63 | CNN_BiGRU | raw_mixup | wav2vec2-large-robust | 0.658 | 0.138 | 0.587 | 0.590 | 0.6 min | 13:51:41 |
| 64 | CNN_BiGRU | pca_mixup | wav2vec2-large-robust | 0.667 | 0.139 | 0.588 | 0.611 | 1.2 min | 13:52:50 |
| 65 | CNN1D | raw | wavlm-large | 0.710 | 0.140 | 0.633 | 0.653 | 0.2 min | 13:53:26 |
| 66 | CNN1D | pca | wavlm-large | 0.693 | 0.175 | 0.624 | 0.628 | 0.8 min | 13:54:11 |
| 67 | CNN1D | raw_mixup | wavlm-large | 0.730 | 0.143 | 0.652 | 0.688 | 0.3 min | 13:54:29 |
| 68 | CNN1D | pca_mixup | wavlm-large | 0.703 | 0.189 | 0.638 | 0.659 | 0.7 min | 13:55:10 |
| 69 | BiLSTM | raw | wavlm-large | 0.542 | 0.169 | 0.520 | 0.496 | 0.4 min | 13:55:36 |
| 70 | BiLSTM | pca | wavlm-large | 0.574 | 0.152 | 0.535 | 0.526 | 1.6 min | 13:57:12 |
| 71 | BiLSTM | raw_mixup | wavlm-large | 0.535 | 0.159 | 0.527 | 0.501 | 0.6 min | 13:57:47 |
| 72 | BiLSTM | pca_mixup | wavlm-large | 0.521 | 0.166 | 0.518 | 0.499 | 0.8 min | 13:58:35 |
| 73 | BiGRU | raw | wavlm-large | 0.591 | 0.163 | 0.548 | 0.545 | 0.3 min | 13:58:54 |
| 74 | BiGRU | pca | wavlm-large | 0.577 | 0.187 | 0.546 | 0.535 | 0.6 min | 13:59:32 |
| 75 | BiGRU | raw_mixup | wavlm-large | 0.545 | 0.138 | 0.516 | 0.486 | 0.5 min | 14:00:03 |
| 76 | BiGRU | pca_mixup | wavlm-large | 0.554 | 0.170 | 0.536 | 0.511 | 0.8 min | 14:00:51 |
| 77 | CNN_BiGRU | raw | wavlm-large | 0.616 | 0.139 | 0.563 | 0.561 | 0.3 min | 14:01:12 |
| 78 | CNN_BiGRU | pca | wavlm-large | 0.626 | 0.146 | 0.567 | 0.548 | 0.6 min | 14:01:49 |
| 79 | CNN_BiGRU | raw_mixup | wavlm-large | 0.661 | 0.155 | 0.602 | 0.623 | 0.5 min | 14:02:17 |
| 80 | CNN_BiGRU | pca_mixup | wavlm-large | 0.644 | 0.157 | 0.584 | 0.588 | 0.7 min | 14:03:02 |
| 81 | CNN1D | raw | hubert-large | 0.674 | 0.150 | 0.590 | 0.607 | 0.2 min | 14:03:36 |
| 82 | CNN1D | pca | hubert-large | 0.664 | 0.164 | 0.589 | 0.601 | 0.5 min | 14:04:04 |
| 83 | CNN1D | raw_mixup | hubert-large | 0.647 | 0.161 | 0.581 | 0.585 | 0.3 min | 14:04:22 |
| 84 | CNN1D | pca_mixup | hubert-large | 0.710 | 0.119 | 0.626 | 0.668 | 0.5 min | 14:04:49 |
| 85 | BiLSTM | raw | hubert-large | 0.547 | 0.159 | 0.529 | 0.487 | 0.3 min | 14:05:09 |
| 86 | BiLSTM | pca | hubert-large | 0.569 | 0.170 | 0.535 | 0.512 | 0.5 min | 14:05:37 |
| 87 | BiLSTM | raw_mixup | hubert-large | 0.535 | 0.155 | 0.528 | 0.501 | 0.5 min | 14:06:06 |
| 88 | BiLSTM | pca_mixup | hubert-large | 0.541 | 0.155 | 0.518 | 0.500 | 0.6 min | 14:06:44 |
| 89 | BiGRU | raw | hubert-large | 0.571 | 0.148 | 0.528 | 0.518 | 0.3 min | 14:07:04 |
| 90 | BiGRU | pca | hubert-large | 0.514 | 0.185 | 0.504 | 0.449 | 0.5 min | 14:07:35 |
| 91 | BiGRU | raw_mixup | hubert-large | 0.528 | 0.150 | 0.514 | 0.489 | 0.5 min | 14:08:04 |
| 92 | BiGRU | pca_mixup | hubert-large | 0.554 | 0.168 | 0.529 | 0.507 | 0.6 min | 14:08:40 |
| 93 | CNN_BiGRU | raw | hubert-large | 0.662 | 0.117 | 0.580 | 0.599 | 0.4 min | 14:09:04 |
| 94 | CNN_BiGRU | pca | hubert-large | 0.653 | 0.140 | 0.592 | 0.596 | 0.5 min | 14:09:36 |
| 95 | CNN_BiGRU | raw_mixup | hubert-large | 0.656 | 0.141 | 0.585 | 0.597 | 0.5 min | 14:10:05 |
| 96 | CNN_BiGRU | pca_mixup | hubert-large | 0.662 | 0.127 | 0.593 | 0.601 | 0.6 min | 14:10:43 |


#### Depresión: Embeddings Nuevos | embeddings_nuevos | dl_script1  (archivo: `log_general.txt`, 36 resultados ✅)

- Script: dl_script1_feature_reduction.py — reduccion/seleccion de features sobre embeddings pooled por paciente (clasificador fijo) con 6 tecnicas: A_raw, B_pca, C_sfs, D_raw_mixup, E_pca_smote, F_pca_adasyn
- Carga de datos: `14:11:06 [INFO]    Cargados NUEVOS: 79 pacientes | dim=1024 | clase0=58 | clase1=21`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|
| 1 | A_raw | xlsr-300m | 0.691 | 0.100 | 0.596 | 0.607 | 0.1 min | 14:11:11 |
| 2 | B_pca | xlsr-300m | 0.612 | 0.200 | 0.578 | 0.578 | 0.1 min | 14:11:18 |
| 3 | C_sfs | xlsr-300m | 0.614 | 0.193 | 0.564 | 0.543 | 53.4 min | 15:04:41 |
| 4 | D_raw_mixup | xlsr-300m | 0.681 | 0.117 | 0.591 | 0.587 | 0.1 min | 15:04:47 |
| 5 | E_pca_smote | xlsr-300m | 0.686 | 0.126 | 0.603 | 0.610 | 0.1 min | 15:04:54 |
| 6 | F_pca_adasyn | xlsr-300m | 0.650 | 0.143 | 0.580 | 0.582 | 0.1 min | 15:05:03 |
| 7 | A_raw | xlsr-53 | 0.634 | 0.164 | 0.579 | 0.575 | 0.1 min | 15:05:26 |
| 8 | B_pca | xlsr-53 | 0.596 | 0.194 | 0.565 | 0.563 | 0.1 min | 15:05:31 |
| 9 | C_sfs | xlsr-53 | 0.604 | 0.183 | 0.570 | 0.557 | 69.9 min | 16:15:25 |
| 10 | D_raw_mixup | xlsr-53 | 0.622 | 0.165 | 0.566 | 0.557 | 0.1 min | 16:15:30 |
| 11 | E_pca_smote | xlsr-53 | 0.616 | 0.181 | 0.582 | 0.579 | 0.1 min | 16:15:37 |
| 12 | F_pca_adasyn | xlsr-53 | 0.589 | 0.202 | 0.565 | 0.555 | 0.1 min | 16:15:45 |
| 13 | A_raw | whisper-large-encoder | 0.682 | 0.167 | 0.611 | 0.595 | 0.1 min | 16:16:18 |
| 14 | B_pca | whisper-large-encoder | 0.652 | 0.200 | 0.612 | 0.603 | 0.1 min | 16:16:24 |
| 15 | C_sfs | whisper-large-encoder | 0.653 | 0.162 | 0.585 | 0.567 | 89.1 min | 17:45:30 |
| 16 | D_raw_mixup | whisper-large-encoder | 0.673 | 0.171 | 0.598 | 0.582 | 0.1 min | 17:45:36 |
| 17 | E_pca_smote | whisper-large-encoder | 0.692 | 0.132 | 0.610 | 0.616 | 0.1 min | 17:45:45 |
| 18 | F_pca_adasyn | whisper-large-encoder | 0.691 | 0.167 | 0.616 | 0.599 | 0.1 min | 17:45:54 |
| 19 | A_raw | wav2vec2-large-robust | 0.777 | 0.128 | 0.695 | 0.740 | 0.1 min | 17:46:16 |
| 20 | B_pca | wav2vec2-large-robust | 0.727 | 0.137 | 0.650 | 0.674 | 0.1 min | 17:46:22 |
| 21 | C_sfs | wav2vec2-large-robust | 0.757 | 0.101 | 0.661 | 0.685 | 107.8 min | 19:34:09 |
| 22 | D_raw_mixup | wav2vec2-large-robust | 0.772 | 0.073 | 0.676 | 0.718 | 0.1 min | 19:34:14 |
| 23 | E_pca_smote | wav2vec2-large-robust | 0.742 | 0.142 | 0.662 | 0.688 | 0.1 min | 19:34:22 |
| 24 | F_pca_adasyn | wav2vec2-large-robust | 0.726 | 0.137 | 0.639 | 0.674 | 0.1 min | 19:34:30 |
| 25 | A_raw | wavlm-large | 0.677 | 0.136 | 0.604 | 0.646 | 0.1 min | 19:34:54 |
| 26 | B_pca | wavlm-large | 0.684 | 0.160 | 0.620 | 0.624 | 0.1 min | 19:35:01 |
| 27 | C_sfs | wavlm-large | 0.651 | 0.178 | 0.598 | 0.596 | 127.5 min | 21:42:30 |
| 28 | D_raw_mixup | wavlm-large | 0.675 | 0.139 | 0.604 | 0.632 | 0.1 min | 21:42:36 |
| 29 | E_pca_smote | wavlm-large | 0.686 | 0.158 | 0.620 | 0.651 | 0.1 min | 21:42:43 |
| 30 | F_pca_adasyn | wavlm-large | 0.664 | 0.162 | 0.601 | 0.626 | 0.1 min | 21:42:52 |
| 31 | A_raw | hubert-large | 0.655 | 0.135 | 0.585 | 0.611 | 0.1 min | 21:43:16 |
| 32 | B_pca | hubert-large | 0.619 | 0.162 | 0.574 | 0.563 | 0.1 min | 21:43:22 |
| 33 | C_sfs | hubert-large | 0.598 | 0.211 | 0.572 | 0.565 | 141.3 min | 00:04:39 |
| 34 | D_raw_mixup | hubert-large | 0.636 | 0.159 | 0.581 | 0.594 | 0.1 min | 00:04:45 |
| 35 | E_pca_smote | hubert-large | 0.611 | 0.190 | 0.580 | 0.590 | 0.1 min | 00:04:52 |
| 36 | F_pca_adasyn | hubert-large | 0.642 | 0.176 | 0.584 | 0.588 | 0.1 min | 00:05:01 |


#### Depresión: Embeddings Nuevos | embeddings_nuevos | dl_script2  (archivo: `log_general.txt`, 96 resultados ✅)

- Script: dl_script2_architectures_fixed.py — 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) sobre secuencias de embeddings, cada una con 4 variantes de datos: raw, pca, raw_mixup, pca_mixup
- Carga de datos: `00:05:41 [INFO]    Cargados NUEVOS: 79 pacientes | shape=(79, 157, 1024) | clase0=58 | clase1=21`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Arquitectura | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN1D | raw | xlsr-300m | 0.692 | 0.146 | 0.617 | 0.634 | 0.2 min | 00:05:53 |
| 2 | CNN1D | pca | xlsr-300m | 0.712 | 0.154 | 0.636 | 0.647 | 0.4 min | 00:06:17 |
| 3 | CNN1D | raw_mixup | xlsr-300m | 0.655 | 0.180 | 0.586 | 0.600 | 0.2 min | 00:06:31 |
| 4 | CNN1D | pca_mixup | xlsr-300m | 0.677 | 0.159 | 0.607 | 0.619 | 0.5 min | 00:06:59 |
| 5 | BiLSTM | raw | xlsr-300m | 0.675 | 0.119 | 0.603 | 0.617 | 0.3 min | 00:07:16 |
| 6 | BiLSTM | pca | xlsr-300m | 0.662 | 0.121 | 0.585 | 0.610 | 0.5 min | 00:07:46 |
| 7 | BiLSTM | raw_mixup | xlsr-300m | 0.631 | 0.156 | 0.574 | 0.594 | 0.5 min | 00:08:16 |
| 8 | BiLSTM | pca_mixup | xlsr-300m | 0.666 | 0.161 | 0.607 | 0.612 | 0.7 min | 00:08:56 |
| 9 | BiGRU | raw | xlsr-300m | 0.689 | 0.123 | 0.610 | 0.635 | 0.3 min | 00:09:13 |
| 10 | BiGRU | pca | xlsr-300m | 0.662 | 0.152 | 0.593 | 0.612 | 0.6 min | 00:09:47 |
| 11 | BiGRU | raw_mixup | xlsr-300m | 0.662 | 0.139 | 0.590 | 0.606 | 0.5 min | 00:10:15 |
| 12 | BiGRU | pca_mixup | xlsr-300m | 0.675 | 0.143 | 0.597 | 0.609 | 0.6 min | 00:10:52 |
| 13 | CNN_BiGRU | raw | xlsr-300m | 0.620 | 0.149 | 0.565 | 0.578 | 0.3 min | 00:11:13 |
| 14 | CNN_BiGRU | pca | xlsr-300m | 0.668 | 0.150 | 0.602 | 0.606 | 0.5 min | 00:11:42 |
| 15 | CNN_BiGRU | raw_mixup | xlsr-300m | 0.629 | 0.140 | 0.576 | 0.589 | 0.4 min | 00:12:06 |
| 16 | CNN_BiGRU | pca_mixup | xlsr-300m | 0.658 | 0.151 | 0.593 | 0.590 | 0.6 min | 00:12:41 |
| 17 | CNN1D | raw | xlsr-53 | 0.527 | 0.217 | 0.519 | 0.492 | 0.2 min | 00:13:09 |
| 18 | CNN1D | pca | xlsr-53 | 0.562 | 0.208 | 0.537 | 0.501 | 0.3 min | 00:13:30 |
| 19 | CNN1D | raw_mixup | xlsr-53 | 0.590 | 0.210 | 0.557 | 0.560 | 0.2 min | 00:13:43 |
| 20 | CNN1D | pca_mixup | xlsr-53 | 0.682 | 0.162 | 0.617 | 0.625 | 0.4 min | 00:14:07 |
| 21 | BiLSTM | raw | xlsr-53 | 0.662 | 0.145 | 0.586 | 0.591 | 0.4 min | 00:14:29 |
| 22 | BiLSTM | pca | xlsr-53 | 0.632 | 0.172 | 0.578 | 0.581 | 0.5 min | 00:14:58 |
| 23 | BiLSTM | raw_mixup | xlsr-53 | 0.662 | 0.153 | 0.603 | 0.601 | 0.5 min | 00:15:27 |
| 24 | BiLSTM | pca_mixup | xlsr-53 | 0.654 | 0.149 | 0.591 | 0.595 | 0.7 min | 00:16:09 |
| 25 | BiGRU | raw | xlsr-53 | 0.613 | 0.165 | 0.556 | 0.564 | 0.3 min | 00:16:28 |
| 26 | BiGRU | pca | xlsr-53 | 0.615 | 0.160 | 0.560 | 0.554 | 0.5 min | 00:16:56 |
| 27 | BiGRU | raw_mixup | xlsr-53 | 0.631 | 0.170 | 0.573 | 0.575 | 0.5 min | 00:17:26 |
| 28 | BiGRU | pca_mixup | xlsr-53 | 0.637 | 0.163 | 0.574 | 0.573 | 0.6 min | 00:17:59 |
| 29 | CNN_BiGRU | raw | xlsr-53 | 0.627 | 0.173 | 0.569 | 0.570 | 0.3 min | 00:18:20 |
| 30 | CNN_BiGRU | pca | xlsr-53 | 0.617 | 0.168 | 0.571 | 0.592 | 0.5 min | 00:18:48 |
| 31 | CNN_BiGRU | raw_mixup | xlsr-53 | 0.631 | 0.142 | 0.560 | 0.577 | 0.5 min | 00:19:19 |
| 32 | CNN_BiGRU | pca_mixup | xlsr-53 | 0.608 | 0.188 | 0.572 | 0.563 | 0.6 min | 00:19:53 |
| 33 | CNN1D | raw | whisper-large-encoder | 0.649 | 0.179 | 0.578 | 0.580 | 0.2 min | 00:20:32 |
| 34 | CNN1D | pca | whisper-large-encoder | 0.682 | 0.167 | 0.601 | 0.613 | 0.6 min | 00:21:10 |
| 35 | CNN1D | raw_mixup | whisper-large-encoder | 0.642 | 0.187 | 0.582 | 0.584 | 0.3 min | 00:21:28 |
| 36 | CNN1D | pca_mixup | whisper-large-encoder | 0.643 | 0.180 | 0.580 | 0.579 | 0.6 min | 00:22:03 |
| 37 | BiLSTM | raw | whisper-large-encoder | 0.655 | 0.156 | 0.579 | 0.592 | 0.4 min | 00:22:26 |
| 38 | BiLSTM | pca | whisper-large-encoder | 0.630 | 0.144 | 0.576 | 0.578 | 0.7 min | 00:23:10 |
| 39 | BiLSTM | raw_mixup | whisper-large-encoder | 0.653 | 0.152 | 0.591 | 0.593 | 0.5 min | 00:23:41 |
| 40 | BiLSTM | pca_mixup | whisper-large-encoder | 0.619 | 0.177 | 0.563 | 0.544 | 0.9 min | 00:24:37 |
| 41 | BiGRU | raw | whisper-large-encoder | 0.689 | 0.140 | 0.603 | 0.620 | 0.4 min | 00:24:59 |
| 42 | BiGRU | pca | whisper-large-encoder | 0.647 | 0.148 | 0.582 | 0.578 | 0.8 min | 00:25:48 |
| 43 | BiGRU | raw_mixup | whisper-large-encoder | 0.641 | 0.174 | 0.582 | 0.590 | 0.5 min | 00:26:17 |
| 44 | BiGRU | pca_mixup | whisper-large-encoder | 0.632 | 0.179 | 0.584 | 0.575 | 0.9 min | 00:27:11 |
| 45 | CNN_BiGRU | raw | whisper-large-encoder | 0.596 | 0.147 | 0.545 | 0.540 | 0.4 min | 00:27:33 |
| 46 | CNN_BiGRU | pca | whisper-large-encoder | 0.642 | 0.146 | 0.570 | 0.562 | 0.8 min | 00:28:22 |
| 47 | CNN_BiGRU | raw_mixup | whisper-large-encoder | 0.655 | 0.139 | 0.579 | 0.580 | 0.5 min | 00:28:49 |
| 48 | CNN_BiGRU | pca_mixup | whisper-large-encoder | 0.642 | 0.170 | 0.578 | 0.566 | 0.9 min | 00:29:42 |
| 49 | CNN1D | raw | wav2vec2-large-robust | 0.585 | 0.205 | 0.534 | 0.517 | 0.2 min | 00:30:11 |
| 50 | CNN1D | pca | wav2vec2-large-robust | 0.544 | 0.220 | 0.525 | 0.489 | 0.4 min | 00:30:32 |
| 51 | CNN1D | raw_mixup | wav2vec2-large-robust | 0.592 | 0.215 | 0.559 | 0.552 | 0.2 min | 00:30:45 |
| 52 | CNN1D | pca_mixup | wav2vec2-large-robust | 0.625 | 0.186 | 0.569 | 0.577 | 0.4 min | 00:31:09 |
| 53 | BiLSTM | raw | wav2vec2-large-robust | 0.764 | 0.112 | 0.689 | 0.746 | 0.3 min | 00:31:29 |
| 54 | BiLSTM | pca | wav2vec2-large-robust | 0.734 | 0.103 | 0.650 | 0.688 | 0.6 min | 00:32:06 |
| 55 | BiLSTM | raw_mixup | wav2vec2-large-robust | 0.752 | 0.129 | 0.682 | 0.728 | 0.5 min | 00:32:34 |
| 56 | BiLSTM | pca_mixup | wav2vec2-large-robust | 0.731 | 0.127 | 0.648 | 0.685 | 0.7 min | 00:33:14 |
| 57 | BiGRU | raw | wav2vec2-large-robust | 0.743 | 0.124 | 0.658 | 0.688 | 0.3 min | 00:33:35 |
| 58 | BiGRU | pca | wav2vec2-large-robust | 0.730 | 0.106 | 0.647 | 0.667 | 0.5 min | 00:34:04 |
| 59 | BiGRU | raw_mixup | wav2vec2-large-robust | 0.721 | 0.139 | 0.644 | 0.670 | 0.6 min | 00:34:38 |
| 60 | BiGRU | pca_mixup | wav2vec2-large-robust | 0.677 | 0.175 | 0.622 | 0.640 | 0.6 min | 00:35:16 |
| 61 | CNN_BiGRU | raw | wav2vec2-large-robust | 0.672 | 0.148 | 0.600 | 0.623 | 0.3 min | 00:35:36 |
| 62 | CNN_BiGRU | pca | wav2vec2-large-robust | 0.684 | 0.137 | 0.609 | 0.629 | 0.5 min | 00:36:04 |
| 63 | CNN_BiGRU | raw_mixup | wav2vec2-large-robust | 0.684 | 0.145 | 0.611 | 0.635 | 0.5 min | 00:36:32 |
| 64 | CNN_BiGRU | pca_mixup | wav2vec2-large-robust | 0.681 | 0.114 | 0.592 | 0.626 | 0.5 min | 00:37:04 |
| 65 | CNN1D | raw | wavlm-large | 0.721 | 0.147 | 0.640 | 0.673 | 0.2 min | 00:37:33 |
| 66 | CNN1D | pca | wavlm-large | 0.671 | 0.198 | 0.620 | 0.632 | 0.4 min | 00:38:00 |
| 67 | CNN1D | raw_mixup | wavlm-large | 0.728 | 0.118 | 0.644 | 0.682 | 0.3 min | 00:38:15 |
| 68 | CNN1D | pca_mixup | wavlm-large | 0.714 | 0.193 | 0.645 | 0.666 | 0.5 min | 00:38:45 |
| 69 | BiLSTM | raw | wavlm-large | 0.575 | 0.140 | 0.536 | 0.519 | 0.4 min | 00:39:07 |
| 70 | BiLSTM | pca | wavlm-large | 0.539 | 0.171 | 0.526 | 0.493 | 0.6 min | 00:39:43 |
| 71 | BiLSTM | raw_mixup | wavlm-large | 0.540 | 0.151 | 0.528 | 0.511 | 0.4 min | 00:40:09 |
| 72 | BiLSTM | pca_mixup | wavlm-large | 0.586 | 0.169 | 0.553 | 0.541 | 0.8 min | 00:40:54 |
| 73 | BiGRU | raw | wavlm-large | 0.596 | 0.147 | 0.549 | 0.545 | 0.3 min | 00:41:14 |
| 74 | BiGRU | pca | wavlm-large | 0.588 | 0.186 | 0.557 | 0.558 | 0.6 min | 00:41:51 |
| 75 | BiGRU | raw_mixup | wavlm-large | 0.570 | 0.152 | 0.539 | 0.518 | 0.5 min | 00:42:22 |
| 76 | BiGRU | pca_mixup | wavlm-large | 0.555 | 0.177 | 0.535 | 0.505 | 0.7 min | 00:43:06 |
| 77 | CNN_BiGRU | raw | wavlm-large | 0.646 | 0.117 | 0.575 | 0.595 | 0.4 min | 00:43:27 |
| 78 | CNN_BiGRU | pca | wavlm-large | 0.593 | 0.162 | 0.555 | 0.533 | 0.6 min | 00:44:04 |
| 79 | CNN_BiGRU | raw_mixup | wavlm-large | 0.656 | 0.132 | 0.588 | 0.612 | 0.4 min | 00:44:30 |
| 80 | CNN_BiGRU | pca_mixup | wavlm-large | 0.599 | 0.179 | 0.579 | 0.576 | 0.8 min | 00:45:15 |
| 81 | CNN1D | raw | hubert-large | 0.666 | 0.151 | 0.583 | 0.602 | 0.2 min | 00:45:45 |
| 82 | CNN1D | pca | hubert-large | 0.696 | 0.138 | 0.613 | 0.621 | 0.4 min | 00:46:10 |
| 83 | CNN1D | raw_mixup | hubert-large | 0.640 | 0.158 | 0.574 | 0.579 | 0.2 min | 00:46:25 |
| 84 | CNN1D | pca_mixup | hubert-large | 0.712 | 0.104 | 0.623 | 0.668 | 0.4 min | 00:46:51 |
| 85 | BiLSTM | raw | hubert-large | 0.552 | 0.147 | 0.518 | 0.489 | 0.3 min | 00:47:09 |
| 86 | BiLSTM | pca | hubert-large | 0.546 | 0.172 | 0.526 | 0.504 | 0.5 min | 00:47:38 |
| 87 | BiLSTM | raw_mixup | hubert-large | 0.537 | 0.162 | 0.520 | 0.502 | 0.5 min | 00:48:06 |
| 88 | BiLSTM | pca_mixup | hubert-large | 0.528 | 0.173 | 0.524 | 0.493 | 0.6 min | 00:48:39 |
| 89 | BiGRU | raw | hubert-large | 0.563 | 0.148 | 0.519 | 0.504 | 0.3 min | 00:48:58 |
| 90 | BiGRU | pca | hubert-large | 0.552 | 0.175 | 0.530 | 0.523 | 0.5 min | 00:49:28 |
| 91 | BiGRU | raw_mixup | hubert-large | 0.544 | 0.159 | 0.514 | 0.499 | 0.5 min | 00:49:55 |
| 92 | BiGRU | pca_mixup | hubert-large | 0.557 | 0.164 | 0.527 | 0.499 | 0.5 min | 00:50:28 |
| 93 | CNN_BiGRU | raw | hubert-large | 0.652 | 0.142 | 0.588 | 0.583 | 0.4 min | 00:50:50 |
| 94 | CNN_BiGRU | pca | hubert-large | 0.652 | 0.148 | 0.582 | 0.575 | 0.5 min | 00:51:22 |
| 95 | CNN_BiGRU | raw_mixup | hubert-large | 0.637 | 0.123 | 0.563 | 0.559 | 0.4 min | 00:51:48 |
| 96 | CNN_BiGRU | pca_mixup | hubert-large | 0.667 | 0.133 | 0.590 | 0.596 | 0.6 min | 00:52:25 |


#### Ansiedad: Embeddings Viejos | embeddings_viejos | dl_script1  (archivo: `log_general.txt`, 36 resultados ✅)

- Script: dl_script1_feature_reduction.py — reduccion/seleccion de features sobre embeddings pooled por paciente (clasificador fijo) con 6 tecnicas: A_raw, B_pca, C_sfs, D_raw_mixup, E_pca_smote, F_pca_adasyn
- Carga de datos: `00:52:48 [INFO]    Cargados: 79 pacientes | dim=1024 | clase0=60 | clase1=19`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|
| 1 | A_raw | xlsr-300m | 0.716 | 0.123 | 0.605 | 0.623 | 0.1 min | 00:52:53 |
| 2 | B_pca | xlsr-300m | 0.618 | 0.175 | 0.557 | 0.555 | 0.1 min | 00:52:59 |
| 3 | C_sfs | xlsr-300m | 0.599 | 0.193 | 0.539 | 0.533 | 53.3 min | 01:46:18 |
| 4 | D_raw_mixup | xlsr-300m | 0.708 | 0.136 | 0.597 | 0.632 | 0.1 min | 01:46:24 |
| 5 | E_pca_smote | xlsr-300m | 0.654 | 0.159 | 0.566 | 0.596 | 0.2 min | 01:46:34 |
| 6 | F_pca_adasyn | xlsr-300m | 0.634 | 0.194 | 0.562 | 0.563 | 0.2 min | 01:46:43 |
| 7 | A_raw | xlsr-53 | 0.682 | 0.148 | 0.594 | 0.619 | 0.1 min | 01:47:09 |
| 8 | B_pca | xlsr-53 | 0.647 | 0.152 | 0.564 | 0.587 | 0.1 min | 01:47:14 |
| 9 | C_sfs | xlsr-53 | 0.626 | 0.171 | 0.560 | 0.578 | 69.9 min | 02:57:07 |
| 10 | D_raw_mixup | xlsr-53 | 0.721 | 0.099 | 0.609 | 0.655 | 0.1 min | 02:57:13 |
| 11 | E_pca_smote | xlsr-53 | 0.609 | 0.166 | 0.535 | 0.541 | 0.1 min | 02:57:22 |
| 12 | F_pca_adasyn | xlsr-53 | 0.647 | 0.155 | 0.556 | 0.592 | 0.1 min | 02:57:29 |
| 13 | A_raw | whisper-large-encoder | 0.672 | 0.119 | 0.558 | 0.590 | 0.1 min | 02:58:06 |
| 14 | B_pca | whisper-large-encoder | 0.586 | 0.162 | 0.516 | 0.519 | 0.1 min | 02:58:12 |
| 15 | C_sfs | whisper-large-encoder | 0.577 | 0.175 | 0.527 | 0.526 | 88.9 min | 04:27:04 |
| 16 | D_raw_mixup | whisper-large-encoder | 0.662 | 0.118 | 0.547 | 0.555 | 0.1 min | 04:27:10 |
| 17 | E_pca_smote | whisper-large-encoder | 0.621 | 0.180 | 0.548 | 0.542 | 0.1 min | 04:27:19 |
| 18 | F_pca_adasyn | whisper-large-encoder | 0.610 | 0.173 | 0.542 | 0.554 | 0.1 min | 04:27:27 |
| 19 | A_raw | wav2vec2-large-robust | 0.708 | 0.098 | 0.600 | 0.645 | 0.1 min | 04:27:53 |
| 20 | B_pca | wav2vec2-large-robust | 0.624 | 0.174 | 0.562 | 0.563 | 0.1 min | 04:27:59 |
| 21 | C_sfs | wav2vec2-large-robust | 0.644 | 0.167 | 0.562 | 0.577 | 107.3 min | 06:15:19 |
| 22 | D_raw_mixup | wav2vec2-large-robust | 0.695 | 0.108 | 0.592 | 0.641 | 0.1 min | 06:15:26 |
| 23 | E_pca_smote | wav2vec2-large-robust | 0.662 | 0.134 | 0.569 | 0.618 | 0.1 min | 06:15:32 |
| 24 | F_pca_adasyn | wav2vec2-large-robust | 0.633 | 0.108 | 0.540 | 0.567 | 0.1 min | 06:15:40 |
| 25 | A_raw | wavlm-large | 0.615 | 0.160 | 0.550 | 0.572 | 0.1 min | 06:16:05 |
| 26 | B_pca | wavlm-large | 0.610 | 0.166 | 0.541 | 0.555 | 0.1 min | 06:16:11 |
| 27 | C_sfs | wavlm-large | 0.600 | 0.143 | 0.514 | 0.529 | 126.9 min | 08:23:08 |
| 28 | D_raw_mixup | wavlm-large | 0.605 | 0.141 | 0.529 | 0.532 | 0.1 min | 08:23:13 |
| 29 | E_pca_smote | wavlm-large | 0.602 | 0.160 | 0.526 | 0.545 | 0.1 min | 08:23:21 |
| 30 | F_pca_adasyn | wavlm-large | 0.635 | 0.172 | 0.563 | 0.574 | 0.1 min | 08:23:29 |
| 31 | A_raw | hubert-large | 0.515 | 0.115 | 0.479 | 0.459 | 0.1 min | 08:23:55 |
| 32 | B_pca | hubert-large | 0.601 | 0.169 | 0.527 | 0.546 | 0.1 min | 08:24:00 |
| 33 | C_sfs | hubert-large | 0.545 | 0.179 | 0.505 | 0.506 | 141.2 min | 10:45:13 |
| 34 | D_raw_mixup | hubert-large | 0.527 | 0.165 | 0.482 | 0.443 | 0.1 min | 10:45:18 |
| 35 | E_pca_smote | hubert-large | 0.570 | 0.169 | 0.504 | 0.513 | 0.1 min | 10:45:27 |
| 36 | F_pca_adasyn | hubert-large | 0.534 | 0.183 | 0.498 | 0.495 | 0.1 min | 10:45:35 |


#### Ansiedad: Embeddings Viejos | embeddings_viejos | dl_script2  (archivo: `log_general.txt`, 59 resultados ✅)

- Script: dl_script2_architectures_fixed.py — 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) sobre secuencias de embeddings, cada una con 4 variantes de datos: raw, pca, raw_mixup, pca_mixup
- Carga de datos: `10:46:20 [INFO]    Cargados: 79 pacientes | shape=(79, 157, 1024) | clase0=60 | clase1=19`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Arquitectura | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN1D | raw | xlsr-300m | 0.714 | 0.097 | 0.597 | 0.636 | 0.2 min | 10:46:32 |
| 2 | CNN1D | pca | xlsr-300m | 0.679 | 0.154 | 0.573 | 0.594 | 0.6 min | 10:47:06 |
| 3 | CNN1D | raw_mixup | xlsr-300m | 0.679 | 0.156 | 0.575 | 0.591 | 0.3 min | 10:47:21 |
| 4 | CNN1D | pca_mixup | xlsr-300m | 0.687 | 0.139 | 0.573 | 0.588 | 0.5 min | 10:47:50 |
| 5 | BiLSTM | raw | xlsr-300m | 0.548 | 0.140 | 0.496 | 0.496 | 0.3 min | 10:48:08 |
| 6 | BiLSTM | pca | xlsr-300m | 0.523 | 0.172 | 0.486 | 0.492 | 0.5 min | 10:48:40 |
| 7 | BiLSTM | raw_mixup | xlsr-300m | 0.557 | 0.141 | 0.497 | 0.495 | 0.5 min | 10:49:10 |
| 8 | BiLSTM | pca_mixup | xlsr-300m | 0.531 | 0.176 | 0.498 | 0.478 | 0.6 min | 10:49:49 |
| 9 | BiGRU | raw | xlsr-300m | 0.517 | 0.184 | 0.489 | 0.466 | 0.3 min | 10:50:10 |
| 10 | BiGRU | pca | xlsr-300m | 0.544 | 0.193 | 0.509 | 0.503 | 0.5 min | 10:50:41 |
| 11 | BiGRU | raw_mixup | xlsr-300m | 0.466 | 0.159 | 0.466 | 0.415 | 0.5 min | 10:51:10 |
| 12 | BiGRU | pca_mixup | xlsr-300m | 0.519 | 0.164 | 0.477 | 0.464 | 0.6 min | 10:51:48 |
| 13 | CNN_BiGRU | raw | xlsr-300m | 0.551 | 0.163 | 0.489 | 0.466 | 0.4 min | 10:52:09 |
| 14 | CNN_BiGRU | pca | xlsr-300m | 0.578 | 0.166 | 0.518 | 0.507 | 0.5 min | 10:52:41 |
| 15 | CNN_BiGRU | raw_mixup | xlsr-300m | 0.581 | 0.182 | 0.522 | 0.500 | 0.5 min | 10:53:11 |
| 16 | CNN_BiGRU | pca_mixup | xlsr-300m | 0.565 | 0.168 | 0.516 | 0.527 | 0.6 min | 10:53:49 |
| 17 | CNN1D | raw | xlsr-53 | 0.560 | 0.226 | 0.515 | 0.526 | 0.2 min | 10:54:19 |
| 18 | CNN1D | pca | xlsr-53 | 0.596 | 0.212 | 0.530 | 0.542 | 0.4 min | 10:54:41 |
| 19 | CNN1D | raw_mixup | xlsr-53 | 0.579 | 0.209 | 0.523 | 0.547 | 0.2 min | 10:54:54 |
| 20 | CNN1D | pca_mixup | xlsr-53 | 0.593 | 0.191 | 0.520 | 0.531 | 0.4 min | 10:55:19 |
| 21 | BiLSTM | raw | xlsr-53 | 0.607 | 0.172 | 0.534 | 0.540 | 0.4 min | 10:55:41 |
| 22 | BiLSTM | pca | xlsr-53 | 0.578 | 0.170 | 0.521 | 0.547 | 0.5 min | 10:56:12 |
| 23 | BiLSTM | raw_mixup | xlsr-53 | 0.594 | 0.180 | 0.525 | 0.512 | 0.5 min | 10:56:41 |
| 24 | BiLSTM | pca_mixup | xlsr-53 | 0.579 | 0.153 | 0.509 | 0.517 | 0.7 min | 10:57:22 |
| 25 | BiGRU | raw | xlsr-53 | 0.601 | 0.176 | 0.516 | 0.516 | 0.4 min | 10:57:44 |
| 26 | BiGRU | pca | xlsr-53 | 0.545 | 0.156 | 0.486 | 0.477 | 0.6 min | 10:58:20 |
| 27 | BiGRU | raw_mixup | xlsr-53 | 0.587 | 0.169 | 0.514 | 0.527 | 0.6 min | 10:58:54 |
| 28 | BiGRU | pca_mixup | xlsr-53 | 0.548 | 0.167 | 0.492 | 0.492 | 0.6 min | 10:59:32 |
| 29 | CNN_BiGRU | raw | xlsr-53 | 0.584 | 0.172 | 0.516 | 0.518 | 0.4 min | 10:59:54 |
| 30 | CNN_BiGRU | pca | xlsr-53 | 0.648 | 0.159 | 0.561 | 0.582 | 0.5 min | 11:00:26 |
| 31 | CNN_BiGRU | raw_mixup | xlsr-53 | 0.587 | 0.171 | 0.520 | 0.519 | 0.4 min | 11:00:52 |
| 32 | CNN_BiGRU | pca_mixup | xlsr-53 | 0.570 | 0.183 | 0.510 | 0.485 | 0.7 min | 11:01:33 |
| 33 | CNN1D | raw | whisper-large-encoder | 0.637 | 0.176 | 0.535 | 0.553 | 0.2 min | 11:02:16 |
| 34 | CNN1D | pca | whisper-large-encoder | 0.636 | 0.169 | 0.535 | 0.540 | 0.7 min | 11:02:56 |
| 35 | CNN1D | raw_mixup | whisper-large-encoder | 0.598 | 0.206 | 0.528 | 0.541 | 0.2 min | 11:03:11 |
| 36 | CNN1D | pca_mixup | whisper-large-encoder | 0.612 | 0.188 | 0.534 | 0.549 | 0.6 min | 11:03:49 |
| 37 | BiLSTM | raw | whisper-large-encoder | 0.527 | 0.162 | 0.477 | 0.449 | 0.4 min | 11:04:13 |
| 38 | BiLSTM | pca | whisper-large-encoder | 0.575 | 0.172 | 0.513 | 0.492 | 0.8 min | 11:05:03 |
| 39 | BiLSTM | raw_mixup | whisper-large-encoder | 0.493 | 0.184 | 0.477 | 0.447 | 0.5 min | 11:05:33 |
| 40 | BiLSTM | pca_mixup | whisper-large-encoder | 0.551 | 0.172 | 0.501 | 0.494 | 0.9 min | 11:06:30 |
| 41 | BiGRU | raw | whisper-large-encoder | 0.510 | 0.190 | 0.478 | 0.436 | 0.4 min | 11:06:51 |
| 42 | BiGRU | pca | whisper-large-encoder | 0.526 | 0.205 | 0.493 | 0.473 | 0.8 min | 11:07:38 |
| 43 | BiGRU | raw_mixup | whisper-large-encoder | 0.453 | 0.187 | 0.460 | 0.386 | 0.5 min | 11:08:07 |
| 44 | BiGRU | pca_mixup | whisper-large-encoder | 0.511 | 0.217 | 0.494 | 0.452 | 0.9 min | 11:09:00 |
| 45 | CNN_BiGRU | raw | whisper-large-encoder | 0.554 | 0.187 | 0.502 | 0.473 | 0.3 min | 11:09:20 |
| 46 | CNN_BiGRU | pca | whisper-large-encoder | 0.552 | 0.168 | 0.493 | 0.461 | 0.8 min | 11:10:09 |
| 47 | CNN_BiGRU | raw_mixup | whisper-large-encoder | 0.509 | 0.197 | 0.483 | 0.470 | 0.5 min | 11:10:37 |
| 48 | CNN_BiGRU | pca_mixup | whisper-large-encoder | 0.531 | 0.178 | 0.494 | 0.487 | 0.8 min | 11:11:27 |
| 49 | CNN1D | raw | wav2vec2-large-robust | 0.590 | 0.206 | 0.524 | 0.529 | 0.1 min | 11:11:57 |
| 50 | CNN1D | pca | wav2vec2-large-robust | 0.589 | 0.203 | 0.518 | 0.526 | 0.4 min | 11:12:19 |
| 51 | CNN1D | raw_mixup | wav2vec2-large-robust | 0.577 | 0.211 | 0.518 | 0.541 | 0.2 min | 11:12:33 |
| 52 | CNN1D | pca_mixup | wav2vec2-large-robust | 0.600 | 0.199 | 0.526 | 0.541 | 0.5 min | 11:13:01 |
| 53 | BiLSTM | raw | wav2vec2-large-robust | 0.574 | 0.143 | 0.520 | 0.512 | 0.3 min | 11:13:22 |
| 54 | BiLSTM | pca | wav2vec2-large-robust | 0.571 | 0.140 | 0.517 | 0.525 | 0.6 min | 11:13:58 |
| 55 | BiLSTM | raw_mixup | wav2vec2-large-robust | 0.537 | 0.172 | 0.503 | 0.478 | 0.6 min | 11:14:33 |
| 56 | BiLSTM | pca_mixup | wav2vec2-large-robust | 0.546 | 0.169 | 0.508 | 0.492 | 0.8 min | 11:15:19 |
| 57 | BiGRU | raw | wav2vec2-large-robust | 0.555 | 0.165 | 0.519 | 0.504 | 0.3 min | 11:15:39 |
| 58 | BiGRU | pca | wav2vec2-large-robust | 0.539 | 0.186 | 0.513 | 0.502 | 0.6 min | 11:16:14 |
| 59 | BiGRU | raw_mixup | wav2vec2-large-robust | 0.546 | 0.164 | 0.509 | 0.487 | 0.5 min | 11:16:44 |


#### Ansiedad: Embeddings Viejos | embeddings_viejos | dl_script2  (archivo: `log_ejecucion_final.txt`, 96 resultados ✅)

- Script: dl_script2_architectures_fixed.py — 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) sobre secuencias de embeddings, cada una con 4 variantes de datos: raw, pca, raw_mixup, pca_mixup
- Carga de datos: `20:18:48 [INFO]    Cargados: 79 pacientes | shape=(79, 157, 1024) | clase0=60 | clase1=19`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Arquitectura | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN1D | raw | xlsr-300m | 0.715 | 0.120 | 0.605 | 0.614 | 0.2 min | 20:19:02 |
| 2 | CNN1D | pca | xlsr-300m | 0.710 | 0.117 | 0.590 | 0.609 | 0.5 min | 20:19:33 |
| 3 | CNN1D | raw_mixup | xlsr-300m | 0.671 | 0.171 | 0.574 | 0.577 | 0.3 min | 20:19:48 |
| 4 | CNN1D | pca_mixup | xlsr-300m | 0.682 | 0.151 | 0.580 | 0.594 | 0.4 min | 20:20:13 |
| 5 | BiLSTM | raw | xlsr-300m | 0.558 | 0.143 | 0.513 | 0.494 | 0.3 min | 20:20:33 |
| 6 | BiLSTM | pca | xlsr-300m | 0.555 | 0.144 | 0.495 | 0.498 | 0.5 min | 20:21:04 |
| 7 | BiLSTM | raw_mixup | xlsr-300m | 0.560 | 0.147 | 0.510 | 0.508 | 0.5 min | 20:21:34 |
| 8 | BiLSTM | pca_mixup | xlsr-300m | 0.511 | 0.201 | 0.506 | 0.479 | 0.6 min | 20:22:07 |
| 9 | BiGRU | raw | xlsr-300m | 0.489 | 0.164 | 0.482 | 0.447 | 0.3 min | 20:22:25 |
| 10 | BiGRU | pca | xlsr-300m | 0.569 | 0.173 | 0.511 | 0.508 | 0.5 min | 20:22:53 |
| 11 | BiGRU | raw_mixup | xlsr-300m | 0.525 | 0.155 | 0.483 | 0.459 | 0.5 min | 20:23:22 |
| 12 | BiGRU | pca_mixup | xlsr-300m | 0.528 | 0.169 | 0.492 | 0.466 | 0.7 min | 20:24:02 |
| 13 | CNN_BiGRU | raw | xlsr-300m | 0.555 | 0.183 | 0.497 | 0.477 | 0.4 min | 20:24:25 |
| 14 | CNN_BiGRU | pca | xlsr-300m | 0.561 | 0.203 | 0.521 | 0.505 | 0.5 min | 20:24:54 |
| 15 | CNN_BiGRU | raw_mixup | xlsr-300m | 0.567 | 0.190 | 0.519 | 0.497 | 0.4 min | 20:25:18 |
| 16 | CNN_BiGRU | pca_mixup | xlsr-300m | 0.573 | 0.170 | 0.524 | 0.512 | 0.5 min | 20:25:50 |
| 17 | CNN1D | raw | xlsr-53 | 0.582 | 0.211 | 0.519 | 0.529 | 0.2 min | 20:26:21 |
| 18 | CNN1D | pca | xlsr-53 | 0.603 | 0.209 | 0.530 | 0.540 | 0.3 min | 20:26:42 |
| 19 | CNN1D | raw_mixup | xlsr-53 | 0.584 | 0.208 | 0.524 | 0.544 | 0.2 min | 20:26:55 |
| 20 | CNN1D | pca_mixup | xlsr-53 | 0.599 | 0.194 | 0.532 | 0.549 | 0.4 min | 20:27:17 |
| 21 | BiLSTM | raw | xlsr-53 | 0.610 | 0.162 | 0.528 | 0.549 | 0.3 min | 20:27:36 |
| 22 | BiLSTM | pca | xlsr-53 | 0.592 | 0.173 | 0.524 | 0.519 | 0.5 min | 20:28:04 |
| 23 | BiLSTM | raw_mixup | xlsr-53 | 0.590 | 0.171 | 0.531 | 0.532 | 0.5 min | 20:28:36 |
| 24 | BiLSTM | pca_mixup | xlsr-53 | 0.554 | 0.195 | 0.515 | 0.529 | 0.6 min | 20:29:10 |
| 25 | BiGRU | raw | xlsr-53 | 0.601 | 0.168 | 0.523 | 0.512 | 0.4 min | 20:29:32 |
| 26 | BiGRU | pca | xlsr-53 | 0.553 | 0.183 | 0.496 | 0.458 | 0.5 min | 20:30:01 |
| 27 | BiGRU | raw_mixup | xlsr-53 | 0.616 | 0.170 | 0.532 | 0.526 | 0.5 min | 20:30:31 |
| 28 | BiGRU | pca_mixup | xlsr-53 | 0.540 | 0.195 | 0.505 | 0.491 | 0.6 min | 20:31:06 |
| 29 | CNN_BiGRU | raw | xlsr-53 | 0.599 | 0.175 | 0.535 | 0.558 | 0.4 min | 20:31:29 |
| 30 | CNN_BiGRU | pca | xlsr-53 | 0.586 | 0.176 | 0.525 | 0.522 | 0.5 min | 20:31:57 |
| 31 | CNN_BiGRU | raw_mixup | xlsr-53 | 0.600 | 0.174 | 0.527 | 0.533 | 0.4 min | 20:32:19 |
| 32 | CNN_BiGRU | pca_mixup | xlsr-53 | 0.615 | 0.165 | 0.548 | 0.550 | 0.5 min | 20:32:52 |
| 33 | CNN1D | raw | whisper-large-encoder | 0.616 | 0.185 | 0.532 | 0.538 | 0.2 min | 20:33:35 |
| 34 | CNN1D | pca | whisper-large-encoder | 0.621 | 0.172 | 0.532 | 0.557 | 0.6 min | 20:34:10 |
| 35 | CNN1D | raw_mixup | whisper-large-encoder | 0.596 | 0.201 | 0.517 | 0.529 | 0.3 min | 20:34:25 |
| 36 | CNN1D | pca_mixup | whisper-large-encoder | 0.603 | 0.194 | 0.524 | 0.536 | 0.6 min | 20:35:00 |
| 37 | BiLSTM | raw | whisper-large-encoder | 0.509 | 0.174 | 0.471 | 0.431 | 0.3 min | 20:35:18 |
| 38 | BiLSTM | pca | whisper-large-encoder | 0.540 | 0.174 | 0.497 | 0.489 | 0.7 min | 20:36:02 |
| 39 | BiLSTM | raw_mixup | whisper-large-encoder | 0.429 | 0.189 | 0.450 | 0.393 | 0.5 min | 20:36:30 |
| 40 | BiLSTM | pca_mixup | whisper-large-encoder | 0.540 | 0.177 | 0.489 | 0.478 | 0.9 min | 20:37:25 |
| 41 | BiGRU | raw | whisper-large-encoder | 0.475 | 0.175 | 0.467 | 0.409 | 0.3 min | 20:37:43 |
| 42 | BiGRU | pca | whisper-large-encoder | 0.556 | 0.191 | 0.511 | 0.479 | 0.7 min | 20:38:25 |
| 43 | BiGRU | raw_mixup | whisper-large-encoder | 0.441 | 0.196 | 0.457 | 0.382 | 0.5 min | 20:38:54 |
| 44 | BiGRU | pca_mixup | whisper-large-encoder | 0.524 | 0.191 | 0.491 | 0.463 | 0.8 min | 20:39:41 |
| 45 | CNN_BiGRU | raw | whisper-large-encoder | 0.531 | 0.190 | 0.496 | 0.475 | 0.4 min | 20:40:05 |
| 46 | CNN_BiGRU | pca | whisper-large-encoder | 0.531 | 0.183 | 0.495 | 0.475 | 0.7 min | 20:40:46 |
| 47 | CNN_BiGRU | raw_mixup | whisper-large-encoder | 0.516 | 0.190 | 0.484 | 0.460 | 0.5 min | 20:41:14 |
| 48 | CNN_BiGRU | pca_mixup | whisper-large-encoder | 0.528 | 0.180 | 0.481 | 0.464 | 0.9 min | 20:42:05 |
| 49 | CNN1D | raw | wav2vec2-large-robust | 0.573 | 0.214 | 0.514 | 0.529 | 0.2 min | 20:42:36 |
| 50 | CNN1D | pca | wav2vec2-large-robust | 0.568 | 0.216 | 0.508 | 0.522 | 0.4 min | 20:42:58 |
| 51 | CNN1D | raw_mixup | wav2vec2-large-robust | 0.586 | 0.212 | 0.519 | 0.542 | 0.2 min | 20:43:11 |
| 52 | CNN1D | pca_mixup | wav2vec2-large-robust | 0.556 | 0.215 | 0.503 | 0.510 | 0.4 min | 20:43:33 |
| 53 | BiLSTM | raw | wav2vec2-large-robust | 0.584 | 0.147 | 0.525 | 0.538 | 0.3 min | 20:43:51 |
| 54 | BiLSTM | pca | wav2vec2-large-robust | 0.613 | 0.183 | 0.550 | 0.555 | 0.5 min | 20:44:20 |
| 55 | BiLSTM | raw_mixup | wav2vec2-large-robust | 0.560 | 0.149 | 0.498 | 0.481 | 0.5 min | 20:44:50 |
| 56 | BiLSTM | pca_mixup | wav2vec2-large-robust | 0.543 | 0.168 | 0.509 | 0.497 | 0.6 min | 20:45:24 |
| 57 | BiGRU | raw | wav2vec2-large-robust | 0.581 | 0.141 | 0.522 | 0.528 | 0.3 min | 20:45:42 |
| 58 | BiGRU | pca | wav2vec2-large-robust | 0.577 | 0.168 | 0.524 | 0.512 | 0.5 min | 20:46:12 |
| 59 | BiGRU | raw_mixup | wav2vec2-large-robust | 0.583 | 0.165 | 0.525 | 0.504 | 0.5 min | 20:46:42 |
| 60 | BiGRU | pca_mixup | wav2vec2-large-robust | 0.568 | 0.175 | 0.521 | 0.510 | 0.6 min | 20:47:15 |
| 61 | CNN_BiGRU | raw | wav2vec2-large-robust | 0.655 | 0.133 | 0.553 | 0.574 | 0.4 min | 20:47:36 |
| 62 | CNN_BiGRU | pca | wav2vec2-large-robust | 0.595 | 0.153 | 0.528 | 0.540 | 0.4 min | 20:48:02 |
| 63 | CNN_BiGRU | raw_mixup | wav2vec2-large-robust | 0.622 | 0.165 | 0.555 | 0.592 | 0.4 min | 20:48:29 |
| 64 | CNN_BiGRU | pca_mixup | wav2vec2-large-robust | 0.628 | 0.150 | 0.545 | 0.566 | 0.5 min | 20:49:00 |
| 65 | CNN1D | raw | wavlm-large | 0.635 | 0.149 | 0.529 | 0.535 | 0.2 min | 20:49:34 |
| 66 | CNN1D | pca | wavlm-large | 0.629 | 0.150 | 0.528 | 0.519 | 0.4 min | 20:49:59 |
| 67 | CNN1D | raw_mixup | wavlm-large | 0.662 | 0.106 | 0.556 | 0.597 | 0.3 min | 20:50:16 |
| 68 | CNN1D | pca_mixup | wavlm-large | 0.571 | 0.199 | 0.521 | 0.514 | 0.5 min | 20:50:48 |
| 69 | BiLSTM | raw | wavlm-large | 0.563 | 0.170 | 0.516 | 0.512 | 0.3 min | 20:51:05 |
| 70 | BiLSTM | pca | wavlm-large | 0.552 | 0.209 | 0.528 | 0.522 | 0.6 min | 20:51:40 |
| 71 | BiLSTM | raw_mixup | wavlm-large | 0.537 | 0.168 | 0.494 | 0.487 | 0.4 min | 20:52:05 |
| 72 | BiLSTM | pca_mixup | wavlm-large | 0.527 | 0.190 | 0.499 | 0.469 | 0.7 min | 20:52:48 |
| 73 | BiGRU | raw | wavlm-large | 0.591 | 0.145 | 0.513 | 0.544 | 0.3 min | 20:53:06 |
| 74 | BiGRU | pca | wavlm-large | 0.615 | 0.175 | 0.543 | 0.558 | 0.6 min | 20:53:42 |
| 75 | BiGRU | raw_mixup | wavlm-large | 0.560 | 0.161 | 0.507 | 0.510 | 0.5 min | 20:54:11 |
| 76 | BiGRU | pca_mixup | wavlm-large | 0.543 | 0.201 | 0.506 | 0.489 | 0.7 min | 20:54:54 |
| 77 | CNN_BiGRU | raw | wavlm-large | 0.623 | 0.167 | 0.555 | 0.566 | 0.4 min | 20:55:15 |
| 78 | CNN_BiGRU | pca | wavlm-large | 0.641 | 0.157 | 0.563 | 0.582 | 0.6 min | 20:55:52 |
| 79 | CNN_BiGRU | raw_mixup | wavlm-large | 0.679 | 0.145 | 0.586 | 0.612 | 0.4 min | 20:56:18 |
| 80 | CNN_BiGRU | pca_mixup | wavlm-large | 0.645 | 0.153 | 0.568 | 0.595 | 0.7 min | 20:57:01 |
| 81 | CNN1D | raw | hubert-large | 0.660 | 0.173 | 0.563 | 0.562 | 0.2 min | 20:57:33 |
| 82 | CNN1D | pca | hubert-large | 0.625 | 0.178 | 0.534 | 0.529 | 0.4 min | 20:57:55 |
| 83 | CNN1D | raw_mixup | hubert-large | 0.606 | 0.195 | 0.528 | 0.542 | 0.2 min | 20:58:10 |
| 84 | CNN1D | pca_mixup | hubert-large | 0.654 | 0.136 | 0.549 | 0.585 | 0.4 min | 20:58:37 |
| 85 | BiLSTM | raw | hubert-large | 0.512 | 0.158 | 0.481 | 0.462 | 0.3 min | 20:58:55 |
| 86 | BiLSTM | pca | hubert-large | 0.510 | 0.174 | 0.488 | 0.466 | 0.5 min | 20:59:26 |
| 87 | BiLSTM | raw_mixup | hubert-large | 0.512 | 0.186 | 0.482 | 0.467 | 0.5 min | 20:59:55 |
| 88 | BiLSTM | pca_mixup | hubert-large | 0.517 | 0.215 | 0.496 | 0.446 | 0.6 min | 21:00:34 |
| 89 | BiGRU | raw | hubert-large | 0.520 | 0.190 | 0.495 | 0.474 | 0.2 min | 21:00:49 |
| 90 | BiGRU | pca | hubert-large | 0.494 | 0.198 | 0.488 | 0.440 | 0.5 min | 21:01:18 |
| 91 | BiGRU | raw_mixup | hubert-large | 0.471 | 0.187 | 0.474 | 0.438 | 0.5 min | 21:01:47 |
| 92 | BiGRU | pca_mixup | hubert-large | 0.490 | 0.176 | 0.487 | 0.472 | 0.5 min | 21:02:19 |
| 93 | CNN_BiGRU | raw | hubert-large | 0.509 | 0.203 | 0.488 | 0.453 | 0.3 min | 21:02:39 |
| 94 | CNN_BiGRU | pca | hubert-large | 0.497 | 0.184 | 0.481 | 0.441 | 0.5 min | 21:03:07 |
| 95 | CNN_BiGRU | raw_mixup | hubert-large | 0.573 | 0.164 | 0.503 | 0.493 | 0.4 min | 21:03:33 |
| 96 | CNN_BiGRU | pca_mixup | hubert-large | 0.548 | 0.168 | 0.500 | 0.501 | 0.5 min | 21:04:04 |


#### Ansiedad: Embeddings Nuevos | embeddings_nuevos | dl_script1  (archivo: `log_ejecucion_final.txt`, 30 resultados ✅)

- Script: dl_script1_feature_reduction.py — reduccion/seleccion de features sobre embeddings pooled por paciente (clasificador fijo) con 6 tecnicas: A_raw, B_pca, C_sfs, D_raw_mixup, E_pca_smote, F_pca_adasyn
- Carga de datos: `21:04:25 [INFO]    Cargados NUEVOS: 79 pacientes | dim=1024 | clase0=60 | clase1=19`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|
| 1 | A_raw | xlsr-300m | 0.698 | 0.154 | 0.611 | 0.653 | 0.1 min | 21:04:31 |
| 2 | B_pca | xlsr-300m | 0.610 | 0.205 | 0.557 | 0.542 | 0.1 min | 21:04:36 |
| 3 | D_raw_mixup | xlsr-300m | 0.708 | 0.137 | 0.605 | 0.631 | 0.1 min | 21:04:43 |
| 4 | E_pca_smote | xlsr-300m | 0.618 | 0.189 | 0.551 | 0.560 | 0.1 min | 21:04:51 |
| 5 | F_pca_adasyn | xlsr-300m | 0.683 | 0.144 | 0.579 | 0.583 | 0.1 min | 21:05:00 |
| 6 | A_raw | xlsr-53 | 0.687 | 0.112 | 0.583 | 0.638 | 0.1 min | 21:05:24 |
| 7 | B_pca | xlsr-53 | 0.600 | 0.173 | 0.534 | 0.540 | 0.1 min | 21:05:29 |
| 8 | D_raw_mixup | xlsr-53 | 0.720 | 0.105 | 0.610 | 0.664 | 0.1 min | 21:05:35 |
| 9 | E_pca_smote | xlsr-53 | 0.610 | 0.174 | 0.552 | 0.572 | 0.1 min | 21:05:43 |
| 10 | F_pca_adasyn | xlsr-53 | 0.640 | 0.142 | 0.559 | 0.591 | 0.1 min | 21:05:50 |
| 11 | A_raw | whisper-large-encoder | 0.627 | 0.171 | 0.550 | 0.561 | 0.1 min | 21:06:23 |
| 12 | B_pca | whisper-large-encoder | 0.602 | 0.167 | 0.537 | 0.549 | 0.1 min | 21:06:29 |
| 13 | D_raw_mixup | whisper-large-encoder | 0.645 | 0.133 | 0.545 | 0.558 | 0.1 min | 21:06:34 |
| 14 | E_pca_smote | whisper-large-encoder | 0.610 | 0.143 | 0.533 | 0.557 | 0.1 min | 21:06:43 |
| 15 | F_pca_adasyn | whisper-large-encoder | 0.620 | 0.158 | 0.540 | 0.545 | 0.1 min | 21:06:51 |
| 16 | A_raw | wav2vec2-large-robust | 0.714 | 0.099 | 0.617 | 0.662 | 0.1 min | 21:07:16 |
| 17 | B_pca | wav2vec2-large-robust | 0.592 | 0.164 | 0.528 | 0.511 | 0.1 min | 21:07:21 |
| 18 | D_raw_mixup | wav2vec2-large-robust | 0.723 | 0.085 | 0.616 | 0.668 | 0.1 min | 21:07:27 |
| 19 | E_pca_smote | wav2vec2-large-robust | 0.660 | 0.170 | 0.578 | 0.588 | 0.1 min | 21:07:34 |
| 20 | F_pca_adasyn | wav2vec2-large-robust | 0.621 | 0.164 | 0.549 | 0.548 | 0.1 min | 21:07:42 |
| 21 | A_raw | wavlm-large | 0.649 | 0.145 | 0.560 | 0.597 | 0.1 min | 21:08:05 |
| 22 | B_pca | wavlm-large | 0.599 | 0.183 | 0.531 | 0.532 | 0.1 min | 21:08:11 |
| 23 | D_raw_mixup | wavlm-large | 0.620 | 0.152 | 0.546 | 0.551 | 0.1 min | 21:08:16 |
| 24 | E_pca_smote | wavlm-large | 0.629 | 0.164 | 0.557 | 0.582 | 0.1 min | 21:08:24 |
| 25 | F_pca_adasyn | wavlm-large | 0.639 | 0.174 | 0.567 | 0.597 | 0.1 min | 21:08:33 |
| 26 | A_raw | hubert-large | 0.528 | 0.158 | 0.479 | 0.468 | 0.1 min | 21:08:56 |
| 27 | B_pca | hubert-large | 0.573 | 0.179 | 0.516 | 0.521 | 0.1 min | 21:09:03 |
| 28 | D_raw_mixup | hubert-large | 0.524 | 0.171 | 0.492 | 0.449 | 0.1 min | 21:09:08 |
| 29 | E_pca_smote | hubert-large | 0.607 | 0.152 | 0.523 | 0.533 | 0.1 min | 21:09:16 |
| 30 | F_pca_adasyn | hubert-large | 0.594 | 0.178 | 0.531 | 0.542 | 0.1 min | 21:09:24 |


#### Ansiedad: Embeddings Nuevos | embeddings_nuevos | dl_script2  (archivo: `log_ejecucion_final.txt`, 96 resultados ✅)

- Script: dl_script2_architectures_fixed.py — 4 arquitecturas de deep learning (CNN1D, BiLSTM, BiGRU, CNN_BiGRU) sobre secuencias de embeddings, cada una con 4 variantes de datos: raw, pca, raw_mixup, pca_mixup
- Carga de datos: `21:09:42 [INFO]    Cargados NUEVOS: 79 pacientes | shape=(79, 157, 1024) | clase0=60 | clase1=19`
- Validacion: 10 repeats x 5 folds (50 evaluaciones por config)

| # | Arquitectura | Experimento | Modelo embedding | F1-macro | ±std | F1-min | AUC | Tiempo | Hora |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN1D | raw | xlsr-300m | 0.712 | 0.118 | 0.599 | 0.621 | 0.2 min | 21:09:54 |
| 2 | CNN1D | pca | xlsr-300m | 0.695 | 0.120 | 0.581 | 0.596 | 0.4 min | 21:10:17 |
| 3 | CNN1D | raw_mixup | xlsr-300m | 0.668 | 0.181 | 0.574 | 0.574 | 0.3 min | 21:10:34 |
| 4 | CNN1D | pca_mixup | xlsr-300m | 0.672 | 0.159 | 0.563 | 0.571 | 0.4 min | 21:10:59 |
| 5 | BiLSTM | raw | xlsr-300m | 0.542 | 0.114 | 0.483 | 0.481 | 0.3 min | 21:11:18 |
| 6 | BiLSTM | pca | xlsr-300m | 0.577 | 0.182 | 0.518 | 0.530 | 0.5 min | 21:11:46 |
| 7 | BiLSTM | raw_mixup | xlsr-300m | 0.550 | 0.146 | 0.505 | 0.496 | 0.4 min | 21:12:13 |
| 8 | BiLSTM | pca_mixup | xlsr-300m | 0.551 | 0.163 | 0.508 | 0.495 | 0.6 min | 21:12:50 |
| 9 | BiGRU | raw | xlsr-300m | 0.544 | 0.149 | 0.488 | 0.480 | 0.3 min | 21:13:09 |
| 10 | BiGRU | pca | xlsr-300m | 0.558 | 0.169 | 0.502 | 0.519 | 0.5 min | 21:13:37 |
| 11 | BiGRU | raw_mixup | xlsr-300m | 0.497 | 0.172 | 0.479 | 0.444 | 0.4 min | 21:14:01 |
| 12 | BiGRU | pca_mixup | xlsr-300m | 0.579 | 0.176 | 0.518 | 0.508 | 0.6 min | 21:14:37 |
| 13 | CNN_BiGRU | raw | xlsr-300m | 0.560 | 0.187 | 0.502 | 0.473 | 0.3 min | 21:14:52 |
| 14 | CNN_BiGRU | pca | xlsr-300m | 0.556 | 0.187 | 0.506 | 0.477 | 0.5 min | 21:15:21 |
| 15 | CNN_BiGRU | raw_mixup | xlsr-300m | 0.585 | 0.166 | 0.513 | 0.506 | 0.4 min | 21:15:47 |
| 16 | CNN_BiGRU | pca_mixup | xlsr-300m | 0.570 | 0.170 | 0.511 | 0.516 | 0.6 min | 21:16:21 |
| 17 | CNN1D | raw | xlsr-53 | 0.615 | 0.198 | 0.538 | 0.564 | 0.2 min | 21:16:49 |
| 18 | CNN1D | pca | xlsr-53 | 0.610 | 0.198 | 0.536 | 0.550 | 0.3 min | 21:17:09 |
| 19 | CNN1D | raw_mixup | xlsr-53 | 0.580 | 0.212 | 0.523 | 0.541 | 0.2 min | 21:17:21 |
| 20 | CNN1D | pca_mixup | xlsr-53 | 0.593 | 0.191 | 0.526 | 0.545 | 0.4 min | 21:17:43 |
| 21 | BiLSTM | raw | xlsr-53 | 0.636 | 0.163 | 0.548 | 0.571 | 0.3 min | 21:18:00 |
| 22 | BiLSTM | pca | xlsr-53 | 0.580 | 0.162 | 0.518 | 0.527 | 0.5 min | 21:18:31 |
| 23 | BiLSTM | raw_mixup | xlsr-53 | 0.608 | 0.182 | 0.541 | 0.555 | 0.5 min | 21:18:59 |
| 24 | BiLSTM | pca_mixup | xlsr-53 | 0.625 | 0.164 | 0.542 | 0.565 | 0.6 min | 21:19:33 |
| 25 | BiGRU | raw | xlsr-53 | 0.620 | 0.160 | 0.539 | 0.532 | 0.3 min | 21:19:52 |
| 26 | BiGRU | pca | xlsr-53 | 0.586 | 0.187 | 0.521 | 0.537 | 0.4 min | 21:20:19 |
| 27 | BiGRU | raw_mixup | xlsr-53 | 0.578 | 0.188 | 0.509 | 0.506 | 0.5 min | 21:20:48 |
| 28 | BiGRU | pca_mixup | xlsr-53 | 0.602 | 0.177 | 0.532 | 0.524 | 0.6 min | 21:21:23 |
| 29 | CNN_BiGRU | raw | xlsr-53 | 0.585 | 0.167 | 0.518 | 0.521 | 0.4 min | 21:21:44 |
| 30 | CNN_BiGRU | pca | xlsr-53 | 0.590 | 0.173 | 0.523 | 0.534 | 0.5 min | 21:22:12 |
| 31 | CNN_BiGRU | raw_mixup | xlsr-53 | 0.602 | 0.156 | 0.521 | 0.527 | 0.4 min | 21:22:37 |
| 32 | CNN_BiGRU | pca_mixup | xlsr-53 | 0.625 | 0.169 | 0.543 | 0.561 | 0.5 min | 21:23:10 |
| 33 | CNN1D | raw | whisper-large-encoder | 0.617 | 0.183 | 0.528 | 0.538 | 0.2 min | 21:23:46 |
| 34 | CNN1D | pca | whisper-large-encoder | 0.653 | 0.144 | 0.549 | 0.562 | 0.6 min | 21:24:20 |
| 35 | CNN1D | raw_mixup | whisper-large-encoder | 0.584 | 0.203 | 0.512 | 0.524 | 0.2 min | 21:24:35 |
| 36 | CNN1D | pca_mixup | whisper-large-encoder | 0.626 | 0.171 | 0.533 | 0.553 | 0.6 min | 21:25:10 |
| 37 | BiLSTM | raw | whisper-large-encoder | 0.518 | 0.181 | 0.489 | 0.442 | 0.3 min | 21:25:31 |
| 38 | BiLSTM | pca | whisper-large-encoder | 0.560 | 0.183 | 0.515 | 0.493 | 0.7 min | 21:26:14 |
| 39 | BiLSTM | raw_mixup | whisper-large-encoder | 0.475 | 0.190 | 0.465 | 0.392 | 0.5 min | 21:26:44 |
| 40 | BiLSTM | pca_mixup | whisper-large-encoder | 0.509 | 0.188 | 0.491 | 0.466 | 0.8 min | 21:27:34 |
| 41 | BiGRU | raw | whisper-large-encoder | 0.475 | 0.171 | 0.460 | 0.406 | 0.3 min | 21:27:54 |
| 42 | BiGRU | pca | whisper-large-encoder | 0.519 | 0.179 | 0.481 | 0.472 | 0.7 min | 21:28:36 |
| 43 | BiGRU | raw_mixup | whisper-large-encoder | 0.476 | 0.190 | 0.459 | 0.405 | 0.5 min | 21:29:07 |
| 44 | BiGRU | pca_mixup | whisper-large-encoder | 0.497 | 0.214 | 0.485 | 0.458 | 0.8 min | 21:29:57 |
| 45 | CNN_BiGRU | raw | whisper-large-encoder | 0.504 | 0.197 | 0.488 | 0.443 | 0.4 min | 21:30:18 |
| 46 | CNN_BiGRU | pca | whisper-large-encoder | 0.553 | 0.187 | 0.499 | 0.470 | 0.7 min | 21:30:58 |
| 47 | CNN_BiGRU | raw_mixup | whisper-large-encoder | 0.534 | 0.208 | 0.493 | 0.469 | 0.4 min | 21:31:22 |
| 48 | CNN_BiGRU | pca_mixup | whisper-large-encoder | 0.541 | 0.186 | 0.502 | 0.473 | 0.8 min | 21:32:12 |
| 49 | CNN1D | raw | wav2vec2-large-robust | 0.590 | 0.218 | 0.527 | 0.543 | 0.1 min | 21:32:37 |
| 50 | CNN1D | pca | wav2vec2-large-robust | 0.617 | 0.186 | 0.532 | 0.554 | 0.4 min | 21:32:58 |
| 51 | CNN1D | raw_mixup | wav2vec2-large-robust | 0.581 | 0.203 | 0.517 | 0.540 | 0.2 min | 21:33:09 |
| 52 | CNN1D | pca_mixup | wav2vec2-large-robust | 0.598 | 0.202 | 0.520 | 0.540 | 0.4 min | 21:33:31 |
| 53 | BiLSTM | raw | wav2vec2-large-robust | 0.584 | 0.125 | 0.512 | 0.520 | 0.3 min | 21:33:50 |
| 54 | BiLSTM | pca | wav2vec2-large-robust | 0.568 | 0.169 | 0.512 | 0.509 | 0.5 min | 21:34:20 |
| 55 | BiLSTM | raw_mixup | wav2vec2-large-robust | 0.527 | 0.169 | 0.496 | 0.477 | 0.5 min | 21:34:49 |
| 56 | BiLSTM | pca_mixup | wav2vec2-large-robust | 0.582 | 0.175 | 0.533 | 0.546 | 0.6 min | 21:35:23 |
| 57 | BiGRU | raw | wav2vec2-large-robust | 0.589 | 0.155 | 0.523 | 0.511 | 0.3 min | 21:35:41 |
| 58 | BiGRU | pca | wav2vec2-large-robust | 0.621 | 0.123 | 0.531 | 0.550 | 0.5 min | 21:36:12 |
| 59 | BiGRU | raw_mixup | wav2vec2-large-robust | 0.562 | 0.177 | 0.517 | 0.495 | 0.5 min | 21:36:40 |
| 60 | BiGRU | pca_mixup | wav2vec2-large-robust | 0.541 | 0.165 | 0.500 | 0.481 | 0.6 min | 21:37:17 |
| 61 | CNN_BiGRU | raw | wav2vec2-large-robust | 0.622 | 0.169 | 0.541 | 0.558 | 0.4 min | 21:37:39 |
| 62 | CNN_BiGRU | pca | wav2vec2-large-robust | 0.601 | 0.189 | 0.543 | 0.561 | 0.4 min | 21:38:06 |
| 63 | CNN_BiGRU | raw_mixup | wav2vec2-large-robust | 0.600 | 0.162 | 0.528 | 0.536 | 0.4 min | 21:38:32 |
| 64 | CNN_BiGRU | pca_mixup | wav2vec2-large-robust | 0.613 | 0.167 | 0.549 | 0.571 | 0.6 min | 21:39:06 |
| 65 | CNN1D | raw | wavlm-large | 0.639 | 0.139 | 0.538 | 0.549 | 0.2 min | 21:39:34 |
| 66 | CNN1D | pca | wavlm-large | 0.625 | 0.168 | 0.533 | 0.553 | 0.4 min | 21:39:59 |
| 67 | CNN1D | raw_mixup | wavlm-large | 0.660 | 0.139 | 0.565 | 0.593 | 0.2 min | 21:40:13 |
| 68 | CNN1D | pca_mixup | wavlm-large | 0.548 | 0.176 | 0.497 | 0.486 | 0.5 min | 21:40:41 |
| 69 | BiLSTM | raw | wavlm-large | 0.547 | 0.190 | 0.512 | 0.506 | 0.3 min | 21:41:00 |
| 70 | BiLSTM | pca | wavlm-large | 0.636 | 0.124 | 0.542 | 0.583 | 0.6 min | 21:41:36 |
| 71 | BiLSTM | raw_mixup | wavlm-large | 0.571 | 0.149 | 0.512 | 0.519 | 0.5 min | 21:42:04 |
| 72 | BiLSTM | pca_mixup | wavlm-large | 0.520 | 0.178 | 0.489 | 0.468 | 0.8 min | 21:42:55 |
| 73 | BiGRU | raw | wavlm-large | 0.584 | 0.180 | 0.537 | 0.554 | 0.3 min | 21:43:15 |
| 74 | BiGRU | pca | wavlm-large | 0.543 | 0.182 | 0.511 | 0.500 | 0.6 min | 21:43:52 |
| 75 | BiGRU | raw_mixup | wavlm-large | 0.578 | 0.166 | 0.530 | 0.537 | 0.5 min | 21:44:21 |
| 76 | BiGRU | pca_mixup | wavlm-large | 0.541 | 0.188 | 0.509 | 0.490 | 0.7 min | 21:45:04 |
| 77 | CNN_BiGRU | raw | wavlm-large | 0.640 | 0.148 | 0.556 | 0.576 | 0.4 min | 21:45:26 |
| 78 | CNN_BiGRU | pca | wavlm-large | 0.632 | 0.143 | 0.553 | 0.574 | 0.6 min | 21:46:01 |
| 79 | CNN_BiGRU | raw_mixup | wavlm-large | 0.670 | 0.149 | 0.575 | 0.607 | 0.4 min | 21:46:25 |
| 80 | CNN_BiGRU | pca_mixup | wavlm-large | 0.638 | 0.160 | 0.563 | 0.576 | 0.7 min | 21:47:07 |
| 81 | CNN1D | raw | hubert-large | 0.653 | 0.170 | 0.557 | 0.567 | 0.2 min | 21:47:35 |
| 82 | CNN1D | pca | hubert-large | 0.614 | 0.182 | 0.520 | 0.522 | 0.4 min | 21:47:58 |
| 83 | CNN1D | raw_mixup | hubert-large | 0.589 | 0.209 | 0.527 | 0.545 | 0.2 min | 21:48:11 |
| 84 | CNN1D | pca_mixup | hubert-large | 0.657 | 0.124 | 0.556 | 0.609 | 0.5 min | 21:48:39 |
| 85 | BiLSTM | raw | hubert-large | 0.502 | 0.172 | 0.483 | 0.462 | 0.3 min | 21:48:57 |
| 86 | BiLSTM | pca | hubert-large | 0.493 | 0.176 | 0.482 | 0.469 | 0.5 min | 21:49:27 |
| 87 | BiLSTM | raw_mixup | hubert-large | 0.481 | 0.171 | 0.468 | 0.424 | 0.5 min | 21:49:55 |
| 88 | BiLSTM | pca_mixup | hubert-large | 0.514 | 0.192 | 0.496 | 0.479 | 0.5 min | 21:50:28 |
| 89 | BiGRU | raw | hubert-large | 0.493 | 0.197 | 0.481 | 0.463 | 0.3 min | 21:50:46 |
| 90 | BiGRU | pca | hubert-large | 0.476 | 0.181 | 0.465 | 0.436 | 0.5 min | 21:51:13 |
| 91 | BiGRU | raw_mixup | hubert-large | 0.505 | 0.177 | 0.482 | 0.465 | 0.4 min | 21:51:40 |
| 92 | BiGRU | pca_mixup | hubert-large | 0.533 | 0.195 | 0.501 | 0.486 | 0.6 min | 21:52:13 |
| 93 | CNN_BiGRU | raw | hubert-large | 0.535 | 0.197 | 0.507 | 0.485 | 0.3 min | 21:52:34 |
| 94 | CNN_BiGRU | pca | hubert-large | 0.538 | 0.187 | 0.499 | 0.467 | 0.5 min | 21:53:01 |
| 95 | CNN_BiGRU | raw_mixup | hubert-large | 0.579 | 0.183 | 0.525 | 0.523 | 0.4 min | 21:53:28 |
| 96 | CNN_BiGRU | pca_mixup | hubert-large | 0.571 | 0.159 | 0.505 | 0.481 | 0.6 min | 21:54:05 |

### 2.6 Bloques "TOP 20 CONFIGURACIONES" impresos por el propio script (dl_script2, agregado sobre 6 modelos × 4 arquitecturas × 4 experimentos)

Estos bloques son generados automáticamente por `dl_script2_architectures_fixed.py` al finalizar cada corrida completa, y constituyen un ranking cruzado (no específico de un solo modelo de embedding) de las mejores 20 combinaciones {modelo_embedding, arquitectura, experimento} por F1-macro medio. Se transcriben completos, tal como aparecen en el log.

#### TOP 20 — Depresión, Embeddings Viejos, dl_script2 (log_general.txt, línea 1914; cierre 14:10:43, duración total del bloque 17.0 h)

| # | Modelo embedding | Arquitectura | Experimento | mean_f1_macro | std_f1_macro | mean_f1_minority | mean_auc |
|---|---|---|---|---|---|---|---|
| 1 | wav2vec2-large-robust | BiLSTM | raw | 0.754928 | 0.104941 | 0.678192 | 0.739621 |
| 2 | wav2vec2-large-robust | BiGRU | raw | 0.750370 | 0.121545 | 0.670716 | 0.709985 |
| 3 | wav2vec2-large-robust | BiLSTM | pca | 0.748491 | 0.121778 | 0.670427 | 0.697826 |
| 4 | wav2vec2-large-robust | BiLSTM | raw_mixup | 0.734021 | 0.109871 | 0.656580 | 0.703500 |
| 5 | wavlm-large | CNN1D | raw_mixup | 0.729954 | 0.142894 | 0.652298 | 0.687795 |
| 6 | wav2vec2-large-robust | BiGRU | pca | 0.724740 | 0.128446 | 0.640181 | 0.637871 |
| 7 | wav2vec2-large-robust | BiGRU | raw_mixup | 0.718097 | 0.147245 | 0.648919 | 0.678985 |
| 8 | wavlm-large | CNN1D | raw | 0.710030 | 0.139722 | 0.633264 | 0.653106 |
| 9 | hubert-large | CNN1D | pca_mixup | 0.709500 | 0.119407 | 0.625887 | 0.667720 |
| 10 | xlsr-300m | CNN1D | pca | 0.707158 | 0.131694 | 0.620627 | 0.638220 |
| 11 | xlsr-300m | CNN1D | raw | 0.705090 | 0.151635 | 0.624760 | 0.638409 |
| 12 | wavlm-large | CNN1D | pca_mixup | 0.703326 | 0.188707 | 0.637812 | 0.658606 |
| 13 | wav2vec2-large-robust | BiLSTM | pca_mixup | 0.699268 | 0.134327 | 0.629060 | 0.660939 |
| 14 | wav2vec2-large-robust | BiGRU | pca_mixup | 0.694331 | 0.136253 | 0.614008 | 0.642947 |
| 15 | wav2vec2-large-robust | CNN_BiGRU | raw | 0.693787 | 0.132658 | 0.606349 | 0.621883 |
| 16 | wavlm-large | CNN1D | pca | 0.693046 | 0.174815 | 0.624009 | 0.627614 |
| 17 | wav2vec2-large-robust | CNN_BiGRU | pca | 0.685682 | 0.134410 | 0.607011 | 0.625939 |
| 18 | xlsr-300m | BiGRU | raw | 0.676498 | 0.117837 | 0.602990 | 0.639864 |
| 19 | hubert-large | CNN1D | raw | 0.674124 | 0.149845 | 0.589989 | 0.606985 |
| 20 | xlsr-300m | CNN1D | pca_mixup | 0.672966 | 0.173350 | 0.600825 | 0.606545 |

#### TOP 20 — Depresión, Embeddings Nuevos, dl_script2 (log_general.txt, línea 3866; cierre 00:52:25, duración total del bloque 0.8 h)

| # | Modelo embedding | Arquitectura | Experimento | mean_f1_macro | std_f1_macro | mean_f1_minority | mean_auc |
|---|---|---|---|---|---|---|---|
| 1 | wav2vec2-large-robust | BiLSTM | raw | 0.763789 | 0.112298 | 0.688828 | 0.745917 |
| 2 | wav2vec2-large-robust | BiLSTM | raw_mixup | 0.752398 | 0.129447 | 0.682009 | 0.727606 |
| 3 | wav2vec2-large-robust | BiGRU | raw | 0.742952 | 0.124331 | 0.658069 | 0.688250 |
| 4 | wav2vec2-large-robust | BiLSTM | pca | 0.733877 | 0.102805 | 0.649771 | 0.687530 |
| 5 | wav2vec2-large-robust | BiLSTM | pca_mixup | 0.730918 | 0.126815 | 0.647618 | 0.685364 |
| 6 | wav2vec2-large-robust | BiGRU | pca | 0.730402 | 0.106339 | 0.646940 | 0.667341 |
| 7 | wavlm-large | CNN1D | raw_mixup | 0.727849 | 0.117966 | 0.643648 | 0.682379 |
| 8 | wavlm-large | CNN1D | raw | 0.721349 | 0.147278 | 0.640226 | 0.672841 |
| 9 | wav2vec2-large-robust | BiGRU | raw_mixup | 0.720876 | 0.139008 | 0.644169 | 0.670068 |
| 10 | wavlm-large | CNN1D | pca_mixup | 0.714406 | 0.192914 | 0.644784 | 0.665765 |
| 11 | xlsr-300m | CNN1D | pca | 0.712126 | 0.153979 | 0.635731 | 0.647038 |
| 12 | hubert-large | CNN1D | pca_mixup | 0.711743 | 0.103630 | 0.622884 | 0.667591 |
| 13 | hubert-large | CNN1D | pca | 0.696357 | 0.137593 | 0.612642 | 0.620977 |
| 14 | xlsr-300m | CNN1D | raw | 0.691543 | 0.146336 | 0.616719 | 0.634114 |
| 15 | whisper-large-encoder | BiGRU | raw | 0.688956 | 0.140325 | 0.603246 | 0.619765 |
| 16 | xlsr-300m | BiGRU | raw | 0.688529 | 0.123032 | 0.609776 | 0.634568 |
| 17 | wav2vec2-large-robust | CNN_BiGRU | raw_mixup | 0.684331 | 0.144874 | 0.610812 | 0.634830 |
| 18 | wav2vec2-large-robust | CNN_BiGRU | pca | 0.684069 | 0.136769 | 0.609249 | 0.629182 |
| 19 | whisper-large-encoder | CNN1D | pca | 0.682191 | 0.167158 | 0.600505 | 0.612636 |
| 20 | xlsr-53 | CNN1D | pca_mixup | 0.682177 | 0.161537 | 0.616554 | 0.624508 |

#### TOP 20 — Ansiedad, Embeddings Viejos, dl_script2 (log_ejecucion_final.txt, línea 1333; cierre 21:04:04, duración total del bloque 0.8 h)

| # | Modelo embedding | Arquitectura | Experimento | mean_f1_macro | std_f1_macro | mean_f1_minority | mean_auc |
|---|---|---|---|---|---|---|---|
| 1 | xlsr-300m | CNN1D | raw | 0.715072 | 0.119863 | 0.604757 | 0.614167 |
| 2 | xlsr-300m | CNN1D | pca | 0.709730 | 0.117255 | 0.589909 | 0.609306 |
| 3 | xlsr-300m | CNN1D | pca_mixup | 0.682345 | 0.151217 | 0.579750 | 0.593611 |
| 4 | wavlm-large | CNN_BiGRU | raw_mixup | 0.679127 | 0.145032 | 0.586022 | 0.611667 |
| 5 | xlsr-300m | CNN1D | raw_mixup | 0.671029 | 0.171378 | 0.573562 | 0.576528 |
| 6 | wavlm-large | CNN1D | raw_mixup | 0.661954 | 0.105632 | 0.555813 | 0.596528 |
| 7 | hubert-large | CNN1D | raw | 0.659972 | 0.173434 | 0.562580 | 0.561667 |
| 8 | wav2vec2-large-robust | CNN_BiGRU | raw | 0.655085 | 0.132931 | 0.553322 | 0.574306 |
| 9 | hubert-large | CNN1D | pca_mixup | 0.653991 | 0.136381 | 0.548960 | 0.584861 |
| 10 | wavlm-large | CNN_BiGRU | pca_mixup | 0.645212 | 0.153149 | 0.567724 | 0.595417 |
| 11 | wavlm-large | CNN_BiGRU | pca | 0.640786 | 0.157000 | 0.562566 | 0.582222 |
| 12 | wavlm-large | CNN1D | raw | 0.635078 | 0.148501 | 0.529003 | 0.535278 |
| 13 | wavlm-large | CNN1D | pca | 0.628552 | 0.149949 | 0.527579 | 0.519167 |
| 14 | wav2vec2-large-robust | CNN_BiGRU | pca_mixup | 0.627876 | 0.149911 | 0.545407 | 0.566389 |
| 15 | hubert-large | CNN1D | pca | 0.624842 | 0.178091 | 0.533607 | 0.528889 |
| 16 | wavlm-large | CNN_BiGRU | raw | 0.623154 | 0.167478 | 0.554721 | 0.566250 |
| 17 | wav2vec2-large-robust | CNN_BiGRU | raw_mixup | 0.621933 | 0.164676 | 0.554898 | 0.592361 |
| 18 | whisper-large-encoder | CNN1D | pca | 0.620581 | 0.171633 | 0.531866 | 0.556806 |
| 19 | whisper-large-encoder | CNN1D | raw | 0.616457 | 0.184725 | 0.531824 | 0.537639 |
| 20 | xlsr-53 | BiGRU | raw_mixup | 0.615556 | 0.170011 | 0.531506 | 0.525972 |

#### TOP 20 — Ansiedad, Embeddings Nuevos, dl_script2 (log_ejecucion_final.txt, línea 3195; cierre 21:54:05, duración total del bloque 0.7 h — última fase, cierre global "REPRODUCCIÓN COMPLETADA")

| # | Modelo embedding | Arquitectura | Experimento | mean_f1_macro | std_f1_macro | mean_f1_minority | mean_auc |
|---|---|---|---|---|---|---|---|
| 1 | xlsr-300m | CNN1D | raw | 0.711534 | 0.117798 | 0.599159 | 0.620694 |
| 2 | xlsr-300m | CNN1D | pca | 0.695348 | 0.119601 | 0.581462 | 0.596389 |
| 3 | xlsr-300m | CNN1D | pca_mixup | 0.671804 | 0.159229 | 0.562977 | 0.571111 |
| 4 | wavlm-large | CNN_BiGRU | raw_mixup | 0.670315 | 0.149478 | 0.574796 | 0.607361 |
| 5 | xlsr-300m | CNN1D | raw_mixup | 0.667625 | 0.180778 | 0.574334 | 0.573889 |
| 6 | wavlm-large | CNN1D | raw_mixup | 0.659823 | 0.138561 | 0.564627 | 0.593333 |
| 7 | hubert-large | CNN1D | pca_mixup | 0.657153 | 0.123719 | 0.556332 | 0.608611 |
| 8 | whisper-large-encoder | CNN1D | pca | 0.653363 | 0.143656 | 0.549068 | 0.562361 |
| 9 | hubert-large | CNN1D | raw | 0.653063 | 0.170491 | 0.557094 | 0.566944 |
| 10 | wavlm-large | CNN_BiGRU | raw | 0.640306 | 0.147673 | 0.556385 | 0.575556 |
| 11 | wavlm-large | CNN1D | raw | 0.639018 | 0.139459 | 0.538232 | 0.549167 |
| 12 | wavlm-large | CNN_BiGRU | pca_mixup | 0.638466 | 0.159824 | 0.563187 | 0.576250 |
| 13 | wavlm-large | BiLSTM | pca | 0.636220 | 0.123888 | 0.541896 | 0.583194 |
| 14 | xlsr-53 | BiLSTM | raw | 0.635772 | 0.162529 | 0.547645 | 0.571389 |
| 15 | wavlm-large | CNN_BiGRU | pca | 0.631678 | 0.142647 | 0.552859 | 0.573750 |
| 16 | whisper-large-encoder | CNN1D | pca_mixup | 0.625619 | 0.170867 | 0.532555 | 0.553333 |
| 17 | xlsr-53 | BiLSTM | pca_mixup | 0.625250 | 0.163558 | 0.542135 | 0.564583 |
| 18 | wavlm-large | CNN1D | pca | 0.625153 | 0.168170 | 0.532825 | 0.553056 |
| 19 | xlsr-53 | CNN_BiGRU | pca_mixup | 0.624565 | 0.168962 | 0.543034 | 0.561250 |
| 20 | wav2vec2-large-robust | CNN_BiGRU | raw | 0.621615 | 0.168501 | 0.540526 | 0.557778 |

**Nota:** No se encontró bloque "TOP 20" para las corridas de `dl_script1` (ese script no imprime ranking agregado en el log, solo los 36 resultados `✅` individuales por modelo×experimento), ni para la fase "Ansiedad Viejos dl_script2" en `log_general.txt` (se interrumpió antes de llegar al cierre donde se imprime el TOP 20; el TOP 20 de esa misma fase sí se encuentra completo en `log_ejecucion_final.txt`, que la re-ejecutó del todo).

---

## 3. Resumen ejecutivo de hallazgos clave para el paper (complemento, no reemplaza las 2 secciones anteriores)

1. **Tamaño y balance del dataset propio:** 79 pacientes por condición. Ansiedad: 60 controles / 19 casos (23.75% positivos). Depresión: 58 controles / 21 casos (26.6% positivos). Ambos datasets están desbalanceados hacia la clase negativa en proporción ~3:1 y ~2.8:1 respectivamente — dato crítico para justificar el uso de técnicas de balanceo (mixup, SMOTE, ADASYN) evaluadas en `dl_script1`.
2. **Volumen de datos derivados:** ~116,200 espectrogramas por condición (4 tipos × 4 ventanas temporales × segmentos), y 31,512 registros de embeddings por condición (6 modelos × ~5,252 segmentos de 5s).
3. **Pipeline de preprocesamiento de audio confirmado por evidencia de archivos:** mp4/m4a (crudo) → wav → wav recortado → wav recortado+denoised → segmentación → espectrogramas / embeddings.
4. **Transcripciones de texto:** 80 muestras con transcripción manual corregida, estructura de entrevista semiestructurada de 7-8 preguntas temáticas (actividades, planes, Navidad, felicidad, tristeza) — compatible con un futuro análisis multimodal texto+audio, y temáticamente alineado con el protocolo DAIC-WOZ.
5. **Dataset público DAIC-WOZ disponible localmente:** 189 pacientes (IDs 300-492) con audio, features faciales/HOG/pose/gaze (OpenFace/CLNF), COVAREP, formantes y transcripción con timestamps — utilizable como dataset de referencia/comparación o para pre-entrenamiento/transfer learning.
6. **581 resultados experimentales completos** extraídos de los logs, cubriendo 2 condiciones (Ansiedad/Depresión) × 2 versiones de embeddings (viejos/nuevos) × 2 scripts (reducción de features / arquitecturas DL) × 6 modelos de embedding × (6 o 4-5 experimentos) × (4 arquitecturas cuando aplica).
7. **Mejor resultado global por condición (dl_script2, TOP-1 del ranking agregado):**
   - Depresión: wav2vec2-large-robust + BiLSTM + raw → F1-macro 0.764 (embeddings nuevos) / 0.755 (viejos), AUC ~0.74-0.75.
   - Ansiedad: xlsr-300m + CNN1D + raw → F1-macro 0.715 (viejos) / 0.712 (nuevos), AUC ~0.61-0.62.
   - Depresión obtiene sistemáticamente mejor desempeño que Ansiedad con esta metodología, y en ambas condiciones "raw" (sin PCA/mixup) en el mejor modelo supera a las variantes con reducción/balanceo.
8. **Sin errores de ejecución:** ninguno de los dos logs contiene errores, tracebacks ni excepciones; la única incidencia operativa es una interrupción no explicada de la corrida de `log_general.txt` (fase Ansiedad-Viejos-dl_script2, al 59/96 configuraciones) que fue retomada y completada 4 días después en `log_ejecucion_final.txt`.
9. **Costo computacional dominado por `C_sfs` (Sequential Feature Selection):** experimento más lento con diferencia (decenas de minutos a >10 horas por combinación modelo×condición), responsable de la mayor parte de las ~23h y ~10h que tomaron las fases de `dl_script1`.
