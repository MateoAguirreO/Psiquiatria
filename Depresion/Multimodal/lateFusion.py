"""
==============================================================================
late_fusion_nested.py
==============================================================================
Fusión tardía (late fusion) con CV ANIDADO sobre la(s) matriz(ces) OOF que
genera build_oof_matrix.py.

Externo: 5 folds x 10 repeticiones (los mismos de folds_manifest_v2.csv)
    -> define los subjects de TEST. El test nunca se toca para entrenar
       ni para calibrar el umbral; solo se predice sobre él al final.

Interno: dentro de cada train externo, un split estratificado 80/20
    -> train_interno: ajusta el modelo de fusión.
    -> val_interno:   calcula el umbral óptimo de Youden (J = TPR - FPR).

──────────────────────────────────────────────────────────────────────────
NUEVO: comparación automática con/sin destilación
──────────────────────────────────────────────────────────────────────────
Este script ya NO importa fusion_tardia.py (evita depender de un archivo
que no viaja junto a este). Define sus propios modelos de fusión y, si
encuentra las dos matrices detalle que produce build_oof_matrix.py
(_con_kd y _sin_kd), corre la fusión en AMBAS y deja un resumen comparativo
en ablation_con_vs_sin_kd.csv -- así "¿vale la pena la destilación?" se
responde con el AUC de fusión real, no a priori.
==============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
INNER_VAL_SIZE = 0.20
METHODS = ["perceptron", "logreg", "voting", "gboost", "mlp"]

# Rutas producidas por build_oof_matrix.py (mismo OUTPUT_DIR que allá)
STACKING_OUTPUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/stacking_output")
DETAIL_CON_KD = STACKING_OUTPUT_DIR / "oof_matrix_detail_by_repeat_con_kd.csv"
DETAIL_SIN_KD = STACKING_OUTPUT_DIR / "oof_matrix_detail_by_repeat_sin_kd.csv"
OUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/fusion_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Modelos de fusión (antes vivían en fusion_tardia.py, ahora autocontenidos)
# ──────────────────────────────────────────────────────────────────────────

def fit_perceptron(X, y):
    return Perceptron(random_state=RANDOM_STATE).fit(X, y)


def fit_logreg(X, y):
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(X, y)


def fit_gradient_boosting(X, y):
    # árboles: se entrena sin escalar (el caller ya le pasa X_raw)
    return GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X, y)


def fit_mlp(X, y):
    return MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000,
                          random_state=RANDOM_STATE).fit(X, y)


def fusion_voting(X_raw):
    """Regla fija sin entrenamiento: promedio simple de las columnas base."""
    return np.mean(X_raw, axis=1)


def fit_and_calibrate(method, X_train_inner, y_train_inner, X_train_inner_raw):
    if method == "perceptron":
        return fit_perceptron(X_train_inner, y_train_inner)
    if method == "logreg":
        return fit_logreg(X_train_inner, y_train_inner)
    if method == "gboost":
        return fit_gradient_boosting(X_train_inner_raw, y_train_inner)
    if method == "mlp":
        return fit_mlp(X_train_inner, y_train_inner)
    if method == "voting":
        return None
    raise ValueError(method)


def predict_proba_method(method, model, X, X_raw):
    if method == "voting":
        return fusion_voting(X_raw)
    if method == "gboost":
        return model.predict_proba(X_raw)[:, 1]
    # Perceptron no tiene predict_proba -> usar decision_function + sigmoide
    if method == "perceptron":
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict_proba(X)[:, 1]


def youden_threshold(y_val, probs_val):
    fpr, tpr, thr = roc_curve(y_val, probs_val)
    finite_mask = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite_mask], tpr[finite_mask], thr[finite_mask]
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thr[best_idx]), float(j[best_idx])


# ──────────────────────────────────────────────────────────────────────────
# Fusión anidada para una matriz detalle dada
# ──────────────────────────────────────────────────────────────────────────

def run_nested_late_fusion(detail_path: Path, feature_cols: list, out_prefix: str):
    det = pd.read_csv(detail_path)
    for c in feature_cols:
        assert c in det.columns, f"{detail_path}: falta la columna '{c}'."

    for m in METHODS:
        det[f"fusion_{m}"] = np.nan
        det[f"fusion_{m}_pred"] = np.nan

    thresholds_rows = []

    for repeat, g_repeat in det.groupby("repeat"):
        outer_folds = sorted(g_repeat["fold"].unique())

        for outer_fold in outer_folds:
            test_mask = (det["repeat"] == repeat) & (det["fold"] == outer_fold)
            train_mask = (det["repeat"] == repeat) & (det["fold"] != outer_fold)

            train_idx = det.index[train_mask]
            X_train_full = det.loc[train_idx, feature_cols].values
            y_train_full = det.loc[train_idx, "y_true"].values
            X_test_raw = det.loc[test_mask, feature_cols].values

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=INNER_VAL_SIZE,
                random_state=RANDOM_STATE + repeat * 100 + outer_fold,
            )
            inner_train_pos, inner_val_pos = next(sss.split(X_train_full, y_train_full))

            X_inner_train_raw = X_train_full[inner_train_pos]
            y_inner_train = y_train_full[inner_train_pos]
            X_inner_val_raw = X_train_full[inner_val_pos]
            y_inner_val = y_train_full[inner_val_pos]

            scaler = StandardScaler()
            X_inner_train_s = scaler.fit_transform(X_inner_train_raw)
            X_inner_val_s = scaler.transform(X_inner_val_raw)
            X_test_s = scaler.transform(X_test_raw)

            for method in METHODS:
                model = fit_and_calibrate(method, X_inner_train_s, y_inner_train, X_inner_train_raw)

                probs_inner_val = predict_proba_method(method, model, X_inner_val_s, X_inner_val_raw)
                thr, j_stat = youden_threshold(y_inner_val, probs_inner_val)

                probs_test = predict_proba_method(method, model, X_test_s, X_test_raw)
                preds_test = (probs_test >= thr).astype(int)

                det.loc[test_mask, f"fusion_{method}"] = probs_test
                det.loc[test_mask, f"fusion_{method}_pred"] = preds_test

                thresholds_rows.append({
                    "repeat": repeat, "fold": outer_fold, "metodo": method,
                    "youden_threshold": thr, "youden_j": j_stat,
                })

    thresholds_df = pd.DataFrame(thresholds_rows)

    baseline_rows = []
    for repeat, g_repeat in det.groupby("repeat"):
        y_true_repeat = g_repeat["y_true"].values
        for col in feature_cols:
            auc = roc_auc_score(y_true_repeat, g_repeat[col].values)
            baseline_rows.append({"repeat": repeat, "metodo": col, "auc": auc})
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_summary_df = (
        baseline_df.groupby("metodo")["auc"].agg(auc_media="mean", auc_std="std").reset_index()
    )

    per_repeat_rows = []
    for repeat, g_repeat in det.groupby("repeat"):
        y_true_repeat = g_repeat["y_true"].values
        for method in METHODS:
            probs = g_repeat[f"fusion_{method}"].values
            preds = g_repeat[f"fusion_{method}_pred"].values
            auc = roc_auc_score(y_true_repeat, probs)
            tn, fp, fn, tp = confusion_matrix(y_true_repeat, preds).ravel()
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            per_repeat_rows.append({"repeat": repeat, "metodo": method,
                                     "auc": auc, "sensibilidad": sens, "especificidad": spec})
    per_repeat_df = pd.DataFrame(per_repeat_rows)

    summary_rows = []
    for method in METHODS:
        sub = per_repeat_df[per_repeat_df["metodo"] == method]
        thr_sub = thresholds_df[thresholds_df["metodo"] == method]["youden_threshold"]
        summary_rows.append({
            "metodo": method,
            "auc_media": sub["auc"].mean(), "auc_std": sub["auc"].std(),
            "sens_media": sub["sensibilidad"].mean(), "sens_std": sub["sensibilidad"].std(),
            "spec_media": sub["especificidad"].mean(), "spec_std": sub["especificidad"].std(),
            "youden_thr_media": thr_sub.mean(), "youden_thr_std": thr_sub.std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    det.to_csv(f"{out_prefix}_nested_oof_detail.csv", index=False)
    thresholds_df.to_csv(f"{out_prefix}_nested_youden_por_fold.csv", index=False)
    per_repeat_df.to_csv(f"{out_prefix}_nested_metricas_por_repeat.csv", index=False)
    summary_df.to_csv(f"{out_prefix}_nested_resumen.csv", index=False)
    baseline_summary_df.to_csv(f"{out_prefix}_nested_baseline_individual.csv", index=False)

    return summary_df, baseline_summary_df


# ──────────────────────────────────────────────────────────────────────────
# Orquestador: corre con_kd y sin_kd (si existen) y compara
# ──────────────────────────────────────────────────────────────────────────

FEATURE_COLS_BASE = ["prob_extratrees_embeddings", "prob_bilstm_embeddings", "prob_bigru_video"]
FEATURE_COLS_CON_KD = FEATURE_COLS_BASE + ["prob_destilacion_estudiante"]


def main():
    results = {}

    if DETAIL_CON_KD.exists():
        print(f"=== Fusión CON destilación ({DETAIL_CON_KD}) ===")
        summary_con, baseline_con = run_nested_late_fusion(
            DETAIL_CON_KD, FEATURE_COLS_CON_KD, str(OUT_DIR / "con_kd")
        )
        print(baseline_con.to_string(index=False))
        print(summary_con.to_string(index=False))
        results["con_kd"] = summary_con
    else:
        print(f"(no encontrado, se omite: {DETAIL_CON_KD})")

    if DETAIL_SIN_KD.exists():
        print(f"\n=== Fusión SIN destilación ({DETAIL_SIN_KD}) ===")
        summary_sin, baseline_sin = run_nested_late_fusion(
            DETAIL_SIN_KD, FEATURE_COLS_BASE, str(OUT_DIR / "sin_kd")
        )
        print(baseline_sin.to_string(index=False))
        print(summary_sin.to_string(index=False))
        results["sin_kd"] = summary_sin
    else:
        print(f"(no encontrado, se omite: {DETAIL_SIN_KD})")

    if "con_kd" in results and "sin_kd" in results:
        merged = results["con_kd"][["metodo", "auc_media", "auc_std"]].merge(
            results["sin_kd"][["metodo", "auc_media", "auc_std"]],
            on="metodo", suffixes=("_con_kd", "_sin_kd"),
        )
        merged["delta_auc_con_menos_sin"] = merged["auc_media_con_kd"] - merged["auc_media_sin_kd"]
        merged.to_csv(OUT_DIR / "ablation_con_vs_sin_kd.csv", index=False)

        print("\n" + "=" * 70)
        print("¿VALE LA PENA LA DESTILACIÓN? (delta de AUC de fusión, con - sin)")
        print("=" * 70)
        print(merged.to_string(index=False))
        print(
            "\nRegla de decisión sugerida: si delta_auc_con_menos_sin es <= 0 (o "
            "menor que el ruido, ~auc_std) para la mayoría de los métodos de "
            "fusión, sácala del ensamble final -- no está aportando lo "
            "suficiente para justificar la complejidad y el riesgo de "
            "circularidad (el Teacher que la genera es la misma columna "
            "prob_bilstm_embeddings)."
        )
    else:
        print(
            "\nPara la comparación automática, corre build_oof_matrix.py dos veces: "
            "una con INCLUDE_DESTILACION = True y otra con False."
        )


if __name__ == "__main__":
    main()