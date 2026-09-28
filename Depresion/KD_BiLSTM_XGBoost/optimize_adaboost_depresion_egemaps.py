"""
==============================================================================
Optimización del mejor modelo de la corrida KD:
    condicion=depresion | dataset=egemaps | tipo=KD_regressor | alpha=0.00
    modelo_student=AdaBoostRegressor  (mean_auc=0.648, mean_f1=0.389, n=50)
==============================================================================

Qué hace este script:
  1. Reutiliza el Teacher OOF (teacher_oof_depresion.csv) --
     no vuelve a entrenar el BiLSTM. alpha=0.00 significa y_KD = p_teacher_OOF
     (imitación pura del Teacher), que fue la mejor configuración encontrada.
  2. Feature engineering LIGERO sobre eGeMAPS (ver justificación abajo).
  3. Optimización de hiperparámetros de AdaBoostRegressor con
     RandomizedSearchCV, agrupado por paciente (GroupKFold) para no filtrar
     información entre segmentos del mismo audio.
  4. Evaluación final honesta con el MISMO protocolo que el resto del paper:
     RepeatedStratifiedKFold(5, 10) a nivel de paciente, usando exactamente
     los folds/repeats ya generados por el Teacher (columnas 'repeat'/'fold'
     de teacher_oof_depresion.csv), comparando ANTES vs DESPUÉS del tuning.

------------------------------------------------------------------------------
FEATURE ENGINEERING -- por qué así y no más "detallado"
------------------------------------------------------------------------------
Con ~80 pacientes (n pequeño) y eGeMAPS (~88 features ya diseñadas a mano por
expertos y con literatura extensa detrás -- Eyben et al. 2016, el estándar de
facto en paralingüística), el riesgo dominante NO es "faltan features", es
overfitting / multicolinealidad. Por eso la literatura reciente en detección
de depresión/ansiedad por voz con eGeMAPS (DAIC-WOZ, AVEC, y trabajos 2022-24)
recomienda estos pasos -- ligeros pero con evidencia -- antes de complicar:

  (a) Filtrar features de varianza casi nula (no aportan señal).
  (b) Filtrar multicolinealidad (pares con |corr| > 0.95): AdaBoost con base
      DecisionTree tolera colinealidad, pero reduce ruido/varianza del
      ensemble y acelera el tuning.
  (c) Selección por Mutual Information contra el target de destilación
      (y_KD, continuo) calculada SOLO en train de cada fold -- reduce
      dimensionalidad de forma no lineal (a diferencia de correlación de
      Pearson), consistente con que el target ya no es binario.
  (d) Normalización intra-paciente opcional (z-score de cada feature dentro
      de cada audio_id) -- técnica reportada en estudios de voz clínica
      (DAIC-WOZ, Cummins et al. 2015) para reducir variabilidad de canal/
      hablante cuando hay múltiples segmentos por sesión; aquí se deja
      DESACTIVADA por defecto (FEATURE_ENG["per_patient_zscore"]=False)
      porque with AdaBoost+árboles el escalado no cambia el resultado y
      puede borrar diferencias de nivel real entre pacientes -- pruébala si
      quieres exprimir un poco más, pero no es la ganancia principal.

No se agregan features sintéticas/ratios ad-hoc (p.ej. F1/F2, jitter*shimmer)
porque eGeMAPS ya las incluye en su mayoría y con n=80 pacientes agregar más
columnas sin selección agresiva tiende a empeorar la generalización.

Instala si falta: pip install scikit-learn scipy --break-system-packages
==============================================================================
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("optimize_adaboost")

SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIG -- ajusta rutas si es necesario (mismas que usaste en el KD)
# =============================================================================
OUTPUT_DIR   = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/KD_LazyPredict_Results")
TEACHER_OOF_PATH = OUTPUT_DIR / "teacher_oof_depresion.csv"
FEATURES_EGEMAPS_PATH = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"

PATIENT_ID_COL = "audio_id"
LABEL_COL      = "label"
DROP_COLS      = ["seg_idx"]
ALPHA          = 0.00     # la mejor config encontrada: imitación pura del Teacher

N_FOLDS   = 5
N_REPEATS = 10   # igual que el resto del paper

FEATURE_ENG = dict(
    variance_threshold=1e-6,     # (a) elimina features casi constantes
    corr_threshold=0.95,         # (b) elimina una de cada par muy correlacionado
    mi_top_k=40,                 # (c) top-K features por mutual information (None = sin selección)
    per_patient_zscore=False,    # (d) normalización intra-paciente -- opcional, ver docstring
)

N_ITER_RANDOM_SEARCH = 60        # combinaciones probadas por RandomizedSearchCV
INNER_CV_FOLDS = 3               # folds internos (agrupados por paciente) para el tuning

_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")


def _normalize_audio_id(value) -> str:
    s = str(value).strip()
    low = s.lower()
    for ext in _AUDIO_EXTENSIONS:
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    return s.strip().lower()


# =============================================================================
# 1. CARGA Y MERGE (Teacher OOF ya generado x eGeMAPS)
# =============================================================================

def load_data():
    df_oof = pd.read_csv(TEACHER_OOF_PATH)
    df_feat = pd.read_csv(FEATURES_EGEMAPS_PATH)

    if LABEL_COL in df_feat.columns:
        df_feat = df_feat.drop(columns=[LABEL_COL])  # la etiqueta real viene de df_oof (y_real)

    df_feat["_merge_key"] = df_feat[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof = df_oof.copy()
    df_oof["_merge_key"] = df_oof[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof_key = df_oof.drop(columns=[PATIENT_ID_COL])

    merged = df_feat.merge(df_oof_key, on="_merge_key", how="inner").drop(columns=["_merge_key"])
    if merged.empty:
        raise ValueError("El merge produjo 0 filas -- revisa que TEACHER_OOF_PATH corresponda "
                          "a la misma corrida y condición que FEATURES_EGEMAPS_PATH.")

    merged["y_KD"] = ALPHA * merged["y_real"] + (1 - ALPHA) * merged["p_teacher_OOF"]

    feature_cols = [c for c in df_feat.columns
                    if c not in ({PATIENT_ID_COL} | set(DROP_COLS) | {"_merge_key"})]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]

    logger.info(f"Datos cargados: {len(merged)} filas (segmentos x repeats), "
                f"{merged[PATIENT_ID_COL].nunique()} audios únicos, {len(feature_cols)} features crudas.")
    return merged, feature_cols


# =============================================================================
# 2. FEATURE ENGINEERING LIGERO (se ajusta SOLO con datos de train de cada fold)
# =============================================================================

def fit_feature_engineering(X_train: pd.DataFrame, y_train: np.ndarray, cfg: dict):
    """
    Devuelve la lista final de columnas a usar y (opcional) los estadísticos
    de z-score por paciente, ajustados SOLO con train (sin leakage).
    """
    cols = list(X_train.columns)

    # (a) varianza casi nula
    variances = X_train[cols].var()
    cols = [c for c in cols if variances[c] > cfg["variance_threshold"]]

    # (b) multicolinealidad: de cada par con |corr|>threshold, se descarta la
    # segunda columna (orden alfabético para que sea determinístico)
    if len(cols) > 1:
        corr = X_train[cols].corr().abs()
        to_drop = set()
        for i, ci in enumerate(cols):
            if ci in to_drop:
                continue
            for cj in cols[i + 1:]:
                if cj in to_drop:
                    continue
                if corr.loc[ci, cj] > cfg["corr_threshold"]:
                    to_drop.add(cj)
        cols = [c for c in cols if c not in to_drop]

    # (c) selección por mutual information contra y_KD (target continuo)
    if cfg["mi_top_k"] is not None and len(cols) > cfg["mi_top_k"]:
        mi = mutual_info_regression(X_train[cols].values, y_train, random_state=SEED)
        top_idx = np.argsort(mi)[::-1][: cfg["mi_top_k"]]
        cols = [cols[i] for i in top_idx]

    return cols


def apply_feature_engineering(X: pd.DataFrame, cols: list, cfg: dict,
                               patient_ids: np.ndarray = None) -> pd.DataFrame:
    X_out = X[cols].copy()
    if cfg["per_patient_zscore"] and patient_ids is not None:
        X_out = X_out.groupby(patient_ids).transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8) if s.std() > 0 else s * 0.0
        )
    return X_out


# =============================================================================
# 3. UMBRAL ÓPTIMO (buscado SOLO en train, igual que el resto del paper)
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1[:-1])])


# =============================================================================
# 4. HYPERPARAMETER TUNING (AdaBoostRegressor, GroupKFold por paciente)
# =============================================================================

def tune_adaboost(X_train, y_kd_train, groups_train):
    """
    RandomizedSearchCV agrupado por paciente. Se afina simultáneamente el
    AdaBoostRegressor y la profundidad de su árbol base (DecisionTreeRegressor),
    que suele ser el hiperparámetro con mayor impacto en AdaBoost.
    """
    base_estimators = [DecisionTreeRegressor(max_depth=d, random_state=SEED)
                        for d in (1, 2, 3, 4)]

    param_dist = {
        "estimator": base_estimators,
        "n_estimators": randint(30, 400),
        "learning_rate": loguniform(1e-3, 2.0),
        "loss": ["linear", "square", "exponential"],
    }

    inner_cv = GroupKFold(n_splits=INNER_CV_FOLDS)
    search = RandomizedSearchCV(
        estimator=AdaBoostRegressor(random_state=SEED),
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring="neg_mean_squared_error",   # y_KD es continuo en [0,1]
        cv=list(inner_cv.split(X_train, y_kd_train, groups=groups_train)),
        random_state=SEED,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_kd_train)
    return search.best_estimator_, search.best_params_, -search.best_score_


# =============================================================================
# 5. EVALUACIÓN CON EL PROTOCOLO COMPLETO (5 folds x 10 repeats)
# =============================================================================

def evaluate_protocol(df: pd.DataFrame, feature_cols_raw: list, use_tuning: bool,
                       fixed_hparams: dict = None):
    """
    Recorre los 50 (repeat, fold) ya definidos por el Teacher. Si
    use_tuning=True, en CADA fold se re-ajusta el feature engineering y se
    vuelve a tunear el modelo con RandomizedSearchCV (más lento, pero es la
    forma correcta y sin leakage). Si use_tuning=False, se usa un modelo con
    hiperparámetros fijos (fixed_hparams) -- útil para el "baseline" antes
    del tuning con hiperparámetros por defecto de AdaBoostRegressor.
    """
    rows = []
    best_params_per_fold = []

    for repeat_num in sorted(df["repeat"].unique()):
        df_rep = df[df["repeat"] == repeat_num]
        for fold_idx in sorted(df_rep["fold"].unique()):
            train_mask = df_rep["fold"] != fold_idx
            val_mask   = df_rep["fold"] == fold_idx

            X_train_raw = df_rep.loc[train_mask, feature_cols_raw]
            X_val_raw   = df_rep.loc[val_mask, feature_cols_raw]
            y_kd_train  = df_rep.loc[train_mask, "y_KD"].values
            y_real_train = df_rep.loc[train_mask, "y_real"].values
            y_real_val  = df_rep.loc[val_mask, "y_real"].values
            groups_train = df_rep.loc[train_mask, PATIENT_ID_COL].values

            # --- feature engineering (ajustado SOLO con train de este fold) ---
            sel_cols = fit_feature_engineering(X_train_raw, y_kd_train, FEATURE_ENG)
            X_train_fe = apply_feature_engineering(
                X_train_raw, sel_cols, FEATURE_ENG, groups_train)
            X_val_fe = apply_feature_engineering(
                X_val_raw, sel_cols, FEATURE_ENG,
                df_rep.loc[val_mask, PATIENT_ID_COL].values)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_fe.values)
            X_val_s   = scaler.transform(X_val_fe.values)

            if use_tuning:
                model, best_params, best_mse = tune_adaboost(X_train_s, y_kd_train, groups_train)
                best_params_per_fold.append(best_params)
            else:
                model = AdaBoostRegressor(random_state=SEED, **(fixed_hparams or {}))
                model.fit(X_train_s, y_kd_train)

            p_val = np.clip(model.predict(X_val_s), 0, 1)
            thr = find_best_threshold(y_real_train, model.predict(X_train_s))
            y_pred_class = (p_val >= thr).astype(int)

            rows.append({
                "repeat": repeat_num,
                "fold": fold_idx,
                "n_features_seleccionadas": len(sel_cols),
                "auc": (roc_auc_score(y_real_val, p_val)
                        if len(np.unique(y_real_val)) > 1 else np.nan),
                "f1": f1_score(y_real_val, y_pred_class, zero_division=0),
                "acc": accuracy_score(y_real_val, y_pred_class),
            })

        logger.info(f"   repeat {repeat_num}/{N_REPEATS} listo "
                    f"({'con tuning' if use_tuning else 'sin tuning'})")

    df_results = pd.DataFrame(rows)
    return df_results, best_params_per_fold


# =============================================================================
# MAIN
# =============================================================================

def main():
    df, feature_cols_raw = load_data()

    logger.info("=" * 70)
    logger.info("BASELINE -- AdaBoostRegressor con hiperparámetros por defecto "
                "(para comparar contra el tuning)")
    logger.info("=" * 70)
    df_baseline, _ = evaluate_protocol(df, feature_cols_raw, use_tuning=False, fixed_hparams={})
    logger.info(f"Baseline: AUC medio = {df_baseline['auc'].mean():.4f} ± {df_baseline['auc'].std():.4f} "
                f"(n={len(df_baseline)} evaluaciones)")

    logger.info("=" * 70)
    logger.info("TUNING -- RandomizedSearchCV por fold (GroupKFold interno por paciente)")
    logger.info("=" * 70)
    df_tuned, best_params_per_fold = evaluate_protocol(df, feature_cols_raw, use_tuning=True)
    logger.info(f"Tuned:    AUC medio = {df_tuned['auc'].mean():.4f} ± {df_tuned['auc'].std():.4f} "
                f"(n={len(df_tuned)} evaluaciones)")
    logger.info(f"          F1  medio = {df_tuned['f1'].mean():.4f} | "
                f"Acc medio = {df_tuned['acc'].mean():.4f} | "
                f"#features medio = {df_tuned['n_features_seleccionadas'].mean():.1f}")

    # Hiperparámetro "consenso": el modo de cada hiperparámetro entre los 50 folds
    params_df = pd.DataFrame(best_params_per_fold)
    logger.info("\nHiperparámetros más frecuentes entre los 50 folds tuneados:")
    for col in params_df.columns:
        try:
            moda = params_df[col].mode(dropna=True).iloc[0]
        except Exception:
            moda = params_df[col].astype(str).mode().iloc[0]
        logger.info(f"   {col}: {moda}")

    out_dir = OUTPUT_DIR / "adaboost_egemaps_depresion_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_baseline.to_csv(out_dir / "resultados_baseline.csv", index=False)
    df_tuned.to_csv(out_dir / "resultados_tuned.csv", index=False)
    params_df.to_csv(out_dir / "hiperparametros_por_fold.csv", index=False)
    logger.info(f"\nResultados guardados en {out_dir}")


if __name__ == "__main__":
    main()