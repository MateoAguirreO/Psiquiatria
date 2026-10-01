# Multimodal_XAI: predicción, SHAP y LIME por muestra (ansiedad y depresión)

Reemplaza a `Depresion/Multimodal/build_oof_matrix_v4_balanceado.py`, que generaba `shap_matrix.csv`, `lime_matrix.csv` y `oof_matrix_*.csv` **solo para depresión**. Este módulo hace lo mismo **para los dos ejes**, cada uno con modelos independientes, y corrige los problemas de procedimiento de la v4 (ver "Diferencias con la v4").

## Esquema

- **75 / 5:** `RepeatedStratifiedKFold(16, R)`. En cada fold se entrena con 75 participantes y se prueba con 5. Las predicciones y explicaciones de esos 5 se agregan a las matrices, y así hasta cubrir a los 80. Con R repeticiones, cada participante queda explicado R veces y las matrices finales promedian.
- **Dentro de cada fold, por rama:**
  1. imputación con la mediana del train;
  2. elección de configuración por CV interna (5-fold) solo con los 75;
  3. predicción de los 5;
  4. SHAP y LIME por muestra, en las ramas interpretables.
- **Fusión:** soft vote (promedio de probabilidades de las ramas disponibles). Se evalúan las **127 combinaciones** de 1 a 7 ramas.
  - **AUTO** elige la combinación por AUC OOF dentro del train y la aplica al test. Es la cifra honesta de "quedarse con la mejor".
  - El máximo de la tabla de combinaciones es optimista por construcción.

## Ramas (modelos base)

| Rama | Modalidad | Contenido | Explicabilidad |
|---|---|---|---|
| `au` | rostro | 66 AU / pose / emoción (py-feat) | SHAP + LIME |
| `rostro_temporal` | rostro | 256 features temporales (MediaPipe) | SHAP + LIME |
| `egemaps` | voz | 88 eGeMAPS (media por participante, audio editado) | SHAP + LIME |
| `voz_microventanas` | voz | wav2vec2-large-robust, 0.19 s cada 2.5 s, última capa | proxy |
| `voz_densa` | voz | ídem, micro-ventanas de 0.19 s contiguas | proxy |
| `texto_emb` | texto | embeddings de frases (Whisper + mpnet multilingüe) | proxy |
| `texto_lexico` | texto | 7 features léxicas interpretables | SHAP + LIME |

**Explicabilidad "proxy" de los embeddings.** El SHAP por dimensión de un embedding no es interpretable. En su lugar:
1. la **contribución de cada modelo** a la fusión, por muestra (descomposición exacta del soft vote);
2. la **correlación** entre la predicción del modelo de embeddings y las features interpretables de su modalidad (eGeMAPS para voz, léxico para texto).

## Salidas (`stacking_output/<eje>/`, gitignored)

| Archivo | Contenido |
|---|---|
| `oof_matrix_detail_by_repeat_<eje>.csv` | probabilidad de cada rama y de AUTO, por participante y repetición |
| `oof_matrix_final_<eje>.csv` | promedio sobre repeticiones |
| `shap_matrix_<eje>.csv` / `lime_matrix_<eje>.csv` | atribución por participante y feature (`<rama>__<feature>`) |
| `contribucion_modelos_<eje>.csv` | aporte de cada modelo a la mejor combinación, por participante |
| `explicacion_embeddings_<eje>.csv` | correlaciones proxy de los embeddings |
| `combinaciones_<eje>.csv` | AUC de las 127 combinaciones + AUTO |
| `resumen_<eje>.txt` | resumen legible |

## Uso

```bash
# los datos por participante se leen de ../Multimodal (o de la variable MULTIMODAL_DATA)
python Multimodal_XAI/xai_por_muestra.py --dx depresion --repeticiones 5
python Multimodal_XAI/xai_por_muestra.py --dx ansiedad  --repeticiones 5
```

El script guarda cada fold en caché y se puede cortar y reanudar (`--minutos`).

## Diferencias con la v4

- Ambos ejes, cada uno con su etiqueta. Sin destilación: la v4 tenía una fuga de segundo orden que inflaba el AUC en +0.078.
- Sin `n_frames_detected`, que es un confusor ligado a la duración del clip.
- Embeddings deterministas: la v4 usaba un attention pooling con pesos aleatorios nunca guardados, lo que los hacía no desplegables.
- La selección de configuración y de combinación nunca mira el test.

## Advertencias al interpretar

- **Etiqueta.** `target_*` es la columna "DX IA", de origen desconocido y sin relación con el PHQ-9 (r = −0.04). Ver `Multimodal/experimento_embeddings/RESULTADOS.md` §13–14.
- **La rama `voz_microventanas` no es robusta.** Su AUC depende del punto de muestreo: desplazar la micro-ventana 0.5–2 s la baja de 0.68 a 0.50–0.59.
- **N ≈ 80:** con etiquetas al azar, el AUC varía entre ~0.31 y ~0.64. Diferencias de ±0.05 entre combinaciones son ruido.

## Privacidad

Este repositorio es público. **Nunca subir `stacking_output/` ni datos por participante.** El `.gitignore` ya cubre `stacking_output/`.
