"""
==============================================================================
Optimización del mejor modelo de la corrida KD -- v2 (SIN nested-CV)
    condicion=depresion | dataset=egemaps | tipo=KD_regressor | alpha=0.00
    modelo_student=AdaBoostRegressor  (mean_auc=0.648, mean_f1=0.389, n=50)
==============================================================================

QUÉ CAMBIÓ RESPECTO A LA v1 (y por qué)
------------------------------------------------------------------------------
La v1 hacía CV anidada: 50 particiones externas (RepeatedStratifiedKFold del
Teacher, 5x10) x un GroupKFold(3) interno NUEVO por cada una, para elegir
hiperparámetros. Eso tenía tres problemas:

  1. El inner CV de 3 folds sin repetición, sobre ~63 pacientes con una clase
     minoritaria pequeña, da un AUC extremadamente ruidoso. Elegir el "mejor"
     de 61 configuraciones contra esa métrica ruidosa favorece ganar por azar
     (winner's curse), no por mérito real.
  2. El GroupKFold(3) era ADEMÁS redundante: el 'repeat'/'fold' que ya viene
     en teacher_oof_*.csv proviene de un RepeatedStratifiedKFold hecho A NIVEL
     DE PACIENTE (cada fila de esa CV era un paciente completo, nunca un
     segmento) -- es decir, YA está agrupado por paciente y estratificado por
     clase. No hacía falta re-agrupar con GroupKFold.
  3. La selección de features se recalculaba de forma distinta en cada uno de
     los 3 inner-folds para el MISMO candidato, así que en la práctica se
     promediaban 3 modelos con features diferentes entre sí.

v2 elimina el nivel de CV anidado y usa un solo nivel, reutilizando
directamente las particiones (repeat, fold) que ya trae teacher_oof_*.csv:

  FASE 1 (selección, barata):
      cada configuración candidata (incluida la "campeona" == config ganadora
      del experimento exploratorio) se evalúa sobre TUNING_REPEATS
      repeticiones (subconjunto de las 10 ya existentes) x 5 folds.
      Se promedia el AUC de esas evaluaciones. Se aplica una REGLA DE NO
      REGRESIÓN: un challenger solo reemplaza al campeón si le gana por un
      margen (NO_REGRESSION_MARGIN), para no cambiar de modelo por ruido.

  FASE 2 (reporte final, una sola vez):
      la configuración ganadora de la Fase 1 se evalúa contra las 10
      repeticiones COMPLETAS (50 folds) -- ese es el número que se reporta
      y se compara contra el 0.648443 de referencia.

Nota de honestidad estadística: como TUNING_REPEATS es un subconjunto de las
10 repeticiones usadas en el reporte final, existe una leve fuga optimista
(esas repeticiones "vieron" el proceso de selección). Es un trade-off
deliberado pedido explícitamente para bajar el costo computacional; si
quieres cero fuga, pon EXCLUDE_TUNING_REPEATS_FROM_FINAL=True (entonces el
reporte final promedia solo las repeticiones que NO se usaron para tunear).
==============================================================================
"""

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("optimize_adaboost_v2")

SEED = 42
np.random.seed(SEED)

N_JOBS = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/KD_LazyPredict_Results")
TEACHER_OOF_PATH = OUTPUT_DIR / "teacher_oof_depresion.csv"
FEATURES_EGEMAPS_PATH = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"

PATIENT_ID_COL = "audio_id"
LABEL_COL = "label"
DROP_COLS = ["seg_idx"]
ALPHA = 0.00  # imitación pura del Teacher -- la mejor config encontrada

N_FOLDS = 5
N_REPEATS = 10          # total de repeticiones ya presentes en teacher_oof_*.csv
TUNING_REPEATS = 3      # subconjunto usado SOLO para elegir hiperparámetros (barato)
TUNING_REPEAT_IDS = list(range(1, TUNING_REPEATS + 1))   # repeats 1..3
EXCLUDE_TUNING_REPEATS_FROM_FINAL = False   # True = reporte final 100% "limpio" (usa solo repeats 4..10)

N_CANDIDATE_CONFIGS = 60     # challengers además del campeón
NO_REGRESSION_MARGIN = 0.005  # un challenger debe superar al campeón por esto para reemplazarlo

FEATURE_ENG = dict(
    variance_threshold=1e-6,
    corr_threshold=0.95,
)

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
# 1. CARGA Y MERGE
# =============================================================================

def load_data():
    df_oof = pd.read_csv(TEACHER_OOF_PATH)
    df_feat = pd.read_csv(FEATURES_EGEMAPS_PATH)

    if LABEL_COL in df_feat.columns:
        df_feat = df_feat.drop(columns=[LABEL_COL])

    df_feat["_merge_key"] = df_feat[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof = df_oof.copy()
    df_oof["_merge_key"] = df_oof[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof_key = df_oof.drop(columns=[PATIENT_ID_COL])

    merged = df_feat.merge(df_oof_key, on="_merge_key", how="inner").drop(columns=["_merge_key"])
    if merged.empty:
        raise ValueError("El merge produjo 0 filas -- revisa TEACHER_OOF_PATH / FEATURES_EGEMAPS_PATH.")

    merged["y_KD"] = ALPHA * merged["y_real"] + (1 - ALPHA) * merged["p_teacher_OOF"]

    feature_cols = [c for c in df_feat.columns
                    if c not in ({PATIENT_ID_COL} | set(DROP_COLS) | {"_merge_key"})]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]

    logger.info(f"Datos cargados: {len(merged)} filas, "
                f"{merged[PATIENT_ID_COL].nunique()} audios únicos, {len(feature_cols)} features crudas.")
    return merged, feature_cols


# =============================================================================
# 2. FEATURE ENGINEERING (se ajusta SOLO con train de cada fold, sin leakage)
# =============================================================================

def select_features(X_train: pd.DataFrame, y_kd_train: np.ndarray, feature_mode: str,
                     all_cols: list) -> list:
    """
    feature_mode:
      'all'        -> las 88 features de eGeMAPS tal cual (config campeona)
      'engineered' -> tras filtrar varianza casi nula + multicolinealidad
      'topK'       -> 'engineered' + top-K por mutual information (K en {20..80})
    """
    if feature_mode == "all":
        return all_cols

    cols = list(all_cols)
    variances = X_train[cols].var()
    cols = [c for c in cols if variances[c] > FEATURE_ENG["variance_threshold"]]

    if len(cols) > 1:
        corr = X_train[cols].corr().abs()
        to_drop = set()
        for i, ci in enumerate(cols):
            if ci in to_drop:
                continue
            for cj in cols[i + 1:]:
                if cj in to_drop:
                    continue
                if corr.loc[ci, cj] > FEATURE_ENG["corr_threshold"]:
                    to_drop.add(cj)
        cols = [c for c in cols if c not in to_drop]

    if feature_mode == "engineered":
        return cols

    if feature_mode.startswith("top"):
        k = int(feature_mode.replace("top", ""))
        if len(cols) <= k:
            return cols
        X_mi = X_train[cols].fillna(X_train[cols].median(numeric_only=True))
        mi = mutual_info_regression(X_mi.values, y_kd_train, random_state=SEED)
        order = np.argsort(mi)[::-1]
        return [cols[i] for i in order[:k]]

    raise ValueError(f"feature_mode desconocido: {feature_mode}")


# =============================================================================
# 3. UMBRAL ÓPTIMO
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1[:-1])])


# =============================================================================
# 4. UN SOLO ENTRENAMIENTO/EVALUACIÓN PARA (config, repeat, fold)
# =============================================================================

def run_one_fold(df_rep: pd.DataFrame, fold_idx: int, feature_cols_raw: list, params: dict):
    train_mask = df_rep["fold"] != fold_idx
    val_mask = df_rep["fold"] == fold_idx

    X_train_raw = df_rep.loc[train_mask, feature_cols_raw]
    X_val_raw = df_rep.loc[val_mask, feature_cols_raw]
    y_kd_train = df_rep.loc[train_mask, "y_KD"].to_numpy()
    y_real_train = df_rep.loc[train_mask, "y_real"].to_numpy()
    y_real_val = df_rep.loc[val_mask, "y_real"].to_numpy()

    sel_cols = select_features(X_train_raw, y_kd_train, params["feature_mode"], feature_cols_raw)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw[sel_cols])
    X_val = imputer.transform(X_val_raw[sel_cols])

    base = DecisionTreeRegressor(max_depth=params["depth"], random_state=SEED)
    model = AdaBoostRegressor(
        estimator=base,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        loss=params["loss"],
        random_state=SEED,
    )
    model.fit(X_train, y_kd_train)

    p_train = np.clip(model.predict(X_train), 0, 1)
    p_val = np.clip(model.predict(X_val), 0, 1)
    thr = find_best_threshold(y_real_train, p_train)
    y_pred_class = (p_val >= thr).astype(int)

    auc = roc_auc_score(y_real_val, p_val) if len(np.unique(y_real_val)) > 1 else np.nan
    f1 = f1_score(y_real_val, y_pred_class, zero_division=0)
    acc = accuracy_score(y_real_val, y_pred_class)
    return auc, f1, acc, len(sel_cols)


def evaluate_config(params: dict, df: pd.DataFrame, feature_cols_raw: list, repeat_ids: list):
    """Evalúa UNA configuración a través de (repeat_ids x N_FOLDS) particiones."""
    aucs, f1s, accs, nfeats = [], [], [], []
    for repeat_num in repeat_ids:
        df_rep = df[df["repeat"] == repeat_num]
        for fold_idx in sorted(df_rep["fold"].unique()):
            auc, f1, acc, nf = run_one_fold(df_rep, fold_idx, feature_cols_raw, params)
            if not np.isnan(auc):
                aucs.append(auc)
                f1s.append(f1)
                accs.append(acc)
                nfeats.append(nf)
    return {
        "params": params,
        "mean_auc": float(np.mean(aucs)) if aucs else -np.inf,
        "std_auc": float(np.std(aucs)) if aucs else np.nan,
        "mean_f1": float(np.mean(f1s)) if f1s else np.nan,
        "mean_acc": float(np.mean(accs)) if accs else np.nan,
        "mean_features": float(np.mean(nfeats)) if nfeats else np.nan,
        "n_evaluaciones": len(aucs),
    }


# =============================================================================
# 5. ESPACIO DE BÚSQUEDA
# =============================================================================

def build_candidate_configs(n_candidates: int, seed: int):
    champion = {
        "name": "champion",
        "depth": 3, "n_estimators": 50, "learning_rate": 1.0,
        "loss": "linear", "feature_mode": "all",
    }
    configs = [champion]

    rng = np.random.default_rng(seed)
    losses = ["linear", "square", "exponential"]
    feature_modes = ["all", "engineered", "top80", "top60", "top50", "top40", "top30", "top20"]

    for i in range(n_candidates):
        configs.append({
            "name": f"challenger_{i}",
            "depth": int(rng.integers(1, 5)),
            "n_estimators": int(rng.integers(30, 400)),
            "learning_rate": float(np.exp(rng.uniform(np.log(1e-3), np.log(2.0)))),
            "loss": str(rng.choice(losses)),
            "feature_mode": str(rng.choice(feature_modes)),
        })
    return configs


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("EXPERIMENTO A MEJORAR: depresion | eGeMAPS | AdaBoostRegressor | alpha=0.00")
    logger.info(f"Referencia (exploración, 50 evals): AUC=0.648443 ± 0.118090")

    df, feature_cols_raw = load_data()

    # =========================================================================
    # FASE 1 -- Selección de configuración (barata: TUNING_REPEATS x 5 folds)
    # =========================================================================
    configs = build_candidate_configs(N_CANDIDATE_CONFIGS, seed=SEED)
    logger.info(f"FASE 1: evaluando {len(configs)} configuraciones "
                f"x {TUNING_REPEATS} repeats x {N_FOLDS} folds "
                f"({TUNING_REPEATS * N_FOLDS} evals/config) | n_jobs={N_JOBS}")

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(evaluate_config)(cfg, df, feature_cols_raw, TUNING_REPEAT_IDS)
        for cfg in configs
    )

    selection_df = pd.DataFrame([
        {"name": r["params"]["name"], **{k: v for k, v in r["params"].items() if k != "name"},
         "mean_auc": r["mean_auc"], "std_auc": r["std_auc"],
         "mean_f1": r["mean_f1"], "mean_acc": r["mean_acc"],
         "mean_features": r["mean_features"], "n_evaluaciones": r["n_evaluaciones"]}
        for r in results
    ]).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    champion_result = next(r for r in results if r["params"]["name"] == "champion")
    best_challenger = max(
        (r for r in results if r["params"]["name"] != "champion"),
        key=lambda r: r["mean_auc"],
    )

    logger.info(f"   Campeón   (config original): AUC={champion_result['mean_auc']:.4f}")
    logger.info(f"   Mejor challenger:             AUC={best_challenger['mean_auc']:.4f} "
                f"({best_challenger['params']['name']}, feature_mode={best_challenger['params']['feature_mode']})")

    if best_challenger["mean_auc"] > champion_result["mean_auc"] + NO_REGRESSION_MARGIN:
        winner = best_challenger
        logger.info(f"   -> El challenger SUPERA al campeón por más de {NO_REGRESSION_MARGIN} "
                     f"(regla de no-regresión). Se usa el challenger.")
    else:
        winner = champion_result
        logger.info(f"   -> Ningún challenger superó al campeón por el margen requerido "
                     f"({NO_REGRESSION_MARGIN}). Se conserva la config original (evita overfitting "
                     f"al proceso de selección).")

    winner_params = winner["params"]

    # =========================================================================
    # FASE 2 -- Reporte final: config ganadora, evaluada UNA sola vez
    # =========================================================================
    if EXCLUDE_TUNING_REPEATS_FROM_FINAL:
        final_repeat_ids = [r for r in range(1, N_REPEATS + 1) if r not in TUNING_REPEAT_IDS]
        logger.info(f"FASE 2: reporte final SIN fuga -- usando solo repeats {final_repeat_ids} "
                     f"(excluye los usados en la selección)")
    else:
        final_repeat_ids = list(range(1, N_REPEATS + 1))
        logger.info(f"FASE 2: reporte final con las {N_REPEATS} repeticiones completas "
                     f"(incluye las {TUNING_REPEATS} usadas también en la selección -- "
                     f"leve sesgo optimista aceptado por costo computacional)")

    final = evaluate_config(winner_params, df, feature_cols_raw, final_repeat_ids)

    logger.info("=" * 70)
    logger.info(f"AUC referencia (config original, exploración completa): 0.648443 ± 0.118090")
    logger.info(f"AUC config ganadora (Fase 2, {len(final_repeat_ids)} repeats): "
                f"{final['mean_auc']:.4f} ± {final['std_auc']:.4f}")
    logger.info(f"F1 medio: {final['mean_f1']:.4f} | Acc media: {final['mean_acc']:.4f} | "
                f"#features medio: {final['mean_features']:.1f}")
    logger.info(f"Config ganadora: {winner_params}")

    out_dir = OUTPUT_DIR / "adaboost_egemaps_depresion_tuning_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_df.to_csv(out_dir / "fase1_seleccion_configs.csv", index=False)
    pd.DataFrame([{
        "referencia_auc": 0.648443, "referencia_std": 0.118090,
        "ganadora_auc": final["mean_auc"], "ganadora_std": final["std_auc"],
        "ganadora_f1": final["mean_f1"], "ganadora_acc": final["mean_acc"],
        "n_evaluaciones_finales": final["n_evaluaciones"],
        **{f"param_{k}": v for k, v in winner_params.items()},
    }]).to_csv(out_dir / "fase2_resultado_final.csv", index=False)
    logger.info(f"Resultados guardados en {out_dir}")


if __name__ == "__main__":
    main()