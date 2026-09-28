"""
Fusión tardía (late fusion) con CV ANIDADO.

Externo: 5 folds x 10 repeticiones (los mismos de oof_matrix_detail_by_repeat.csv)
    -> define los subjects de TEST. El test nunca se toca para entrenar
       ni para calibrar el umbral; solo se predice sobre él al final.

Interno: dentro de cada train externo, un split estratificado 80/20
    -> train_interno: ajusta el modelo de fusión.
    -> val_interno:   calcula el umbral óptimo de Youden (J = TPR - FPR)
       para ESE modelo, sin haber visto el test externo.

El modelo que predice sobre el test externo es el mismo que se entrenó
con train_interno (no se reentrena con train+val juntos) -- esto es
consistente porque el umbral de Youden reportado corresponde exactamente
a ese modelo, y el objetivo es el análisis del paper (no un modelo de
despliegue), así que no hace falta reconciliar "más datos para el modelo
final" con "el umbral que lo calibró".

Al unir las 50 predicciones de test (5 folds x 10 repeats) se obtiene la
matriz OOF anidada definitiva -> sobre esa se sacan ROC, matrices de
confusión, sensibilidad/especificidad, etc.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Reutiliza las funciones de ajuste de modelos ya definidas en fusion_tardia.py
from fusion_tardia import (
    FEATURE_COLS,
    RANDOM_STATE,
    fusion_voting,
    fit_perceptron,
    fit_logreg,
    fit_gradient_boosting,
    fit_mlp,
)

INNER_VAL_SIZE = 0.20  # 80/20 estratificado para el split interno

METHODS = ["perceptron", "logreg", "voting", "gboost", "mlp"]


def youden_threshold(y_val, probs_val):
    """
    Umbral óptimo de Youden: maximiza J = sensibilidad + especificidad - 1
                                        = TPR - FPR
    sobre la curva ROC calculada en el val interno.

    roc_curve agrega un primer punto artificial con threshold=+inf (nadie
    clasificado como positivo). En folds internos chicos ese punto puede
    empatar/ganar el argmax por ruido de muestreo, devolviendo un umbral
    inalcanzable -- se descarta antes de buscar el máximo.
    """
    fpr, tpr, thr = roc_curve(y_val, probs_val)
    finite_mask = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite_mask], tpr[finite_mask], thr[finite_mask]
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thr[best_idx]), float(j[best_idx])


def fit_and_calibrate(method, X_train_inner, y_train_inner, scaler):
    """Ajusta el modelo del método dado sobre train_interno (ya escalado si aplica)."""
    if method == "perceptron":
        return fit_perceptron(X_train_inner, y_train_inner)
    if method == "logreg":
        return fit_logreg(X_train_inner, y_train_inner)
    if method == "gboost":
        # árboles: se entrena sin escalar (igual que en fusion_tardia.py)
        return fit_gradient_boosting(X_train_inner, y_train_inner)
    if method == "mlp":
        return fit_mlp(X_train_inner, y_train_inner)
    if method == "voting":
        return None  # regla fija, no requiere modelo entrenado
    raise ValueError(method)


def predict_proba_method(method, model, X, X_raw):
    """
    X: features escaladas (para métodos lineales/mlp).
    X_raw: features sin escalar (para gboost y voting).
    """
    if method == "voting":
        return fusion_voting(X_raw)
    if method == "gboost":
        return model.predict_proba(X_raw)[:, 1]
    return model.predict_proba(X)[:, 1]


def run_nested_late_fusion(detail_path, out_prefix):
    det = pd.read_csv(detail_path)

    # columnas de salida: probabilidad OOF de test externo + umbral aplicado + clase
    for m in METHODS:
        det[f"fusion_{m}"] = np.nan
        det[f"fusion_{m}_pred"] = np.nan

    thresholds_rows = []  # un registro por (repeat, fold externo, método)

    for repeat, g_repeat in det.groupby("repeat"):
        outer_folds = sorted(g_repeat["fold"].unique())

        for outer_fold in outer_folds:
            test_mask = (det["repeat"] == repeat) & (det["fold"] == outer_fold)
            train_mask = (det["repeat"] == repeat) & (det["fold"] != outer_fold)

            train_idx = det.index[train_mask]
            X_train_full = det.loc[train_idx, FEATURE_COLS].values
            y_train_full = det.loc[train_idx, "y_true"].values
            X_test_raw = det.loc[test_mask, FEATURE_COLS].values

            # --- split interno estratificado 80/20 dentro del train externo ---
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=INNER_VAL_SIZE,
                random_state=RANDOM_STATE + repeat * 100 + outer_fold,
            )
            inner_train_pos, inner_val_pos = next(sss.split(X_train_full, y_train_full))

            X_inner_train_raw = X_train_full[inner_train_pos]
            y_inner_train = y_train_full[inner_train_pos]
            X_inner_val_raw = X_train_full[inner_val_pos]
            y_inner_val = y_train_full[inner_val_pos]

            # escalado ajustado SOLO con train_interno (evita fuga hacia val/test)
            scaler = StandardScaler()
            X_inner_train_s = scaler.fit_transform(X_inner_train_raw)
            X_inner_val_s = scaler.transform(X_inner_val_raw)
            X_test_s = scaler.transform(X_test_raw)

            for method in METHODS:
                model = fit_and_calibrate(method, X_inner_train_s, y_inner_train, scaler)

                # Youden se calcula con el val interno (nunca visto por el modelo)
                probs_inner_val = predict_proba_method(
                    method, model, X_inner_val_s, X_inner_val_raw
                )
                thr, j_stat = youden_threshold(y_inner_val, probs_inner_val)

                # predicción final sobre el test externo (el modelo NO se reentrena)
                probs_test = predict_proba_method(method, model, X_test_s, X_test_raw)
                preds_test = (probs_test >= thr).astype(int)

                det.loc[test_mask, f"fusion_{method}"] = probs_test
                det.loc[test_mask, f"fusion_{method}_pred"] = preds_test

                thresholds_rows.append(
                    {
                        "repeat": repeat,
                        "fold": outer_fold,
                        "metodo": method,
                        "youden_threshold": thr,
                        "youden_j": j_stat,
                    }
                )

    thresholds_df = pd.DataFrame(thresholds_rows)

    # ---- BASELINE: AUC de cada modelo base SOLO (sin fusión), sobre los
    # mismos folds de test que usa la fusión -- para comparar limpio contra
    # limpio (misma matriz OOF corregida para ambos lados) ----
    baseline_rows = []
    for repeat, g_repeat in det.groupby("repeat"):
        y_true_repeat = g_repeat["y_true"].values
        for col in FEATURE_COLS:
            auc = roc_auc_score(y_true_repeat, g_repeat[col].values)
            baseline_rows.append({"repeat": repeat, "metodo": col, "auc": auc})
    baseline_df = pd.DataFrame(baseline_rows)

    print("=== AUC individual por modelo base (sin fusión, mismos folds de test) ===")
    baseline_summary_rows = []
    for col in FEATURE_COLS:
        sub = baseline_df[baseline_df["metodo"] == col]
        row = {"metodo": col, "auc_media": sub["auc"].mean(), "auc_std": sub["auc"].std()}
        print(f"{col:32s} AUC={row['auc_media']:.4f}±{row['auc_std']:.4f}")
        baseline_summary_rows.append(row)
    baseline_summary_df = pd.DataFrame(baseline_summary_rows)

    # ---- métricas por repetición (uniendo los 5 folds externos de cada repeat) ----
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
            per_repeat_rows.append(
                {"repeat": repeat, "metodo": method, "auc": auc,
                 "sensibilidad": sens, "especificidad": spec}
            )
    per_repeat_df = pd.DataFrame(per_repeat_rows)

    print("=== Resumen anidado por método de fusión (media ± DE sobre 10 repeats) ===")
    summary_rows = []
    for method in METHODS:
        sub = per_repeat_df[per_repeat_df["metodo"] == method]
        thr_sub = thresholds_df[thresholds_df["metodo"] == method]["youden_threshold"]
        row = {
            "metodo": method,
            "auc_media": sub["auc"].mean(), "auc_std": sub["auc"].std(),
            "sens_media": sub["sensibilidad"].mean(), "sens_std": sub["sensibilidad"].std(),
            "spec_media": sub["especificidad"].mean(), "spec_std": sub["especificidad"].std(),
            "youden_thr_media": thr_sub.mean(), "youden_thr_std": thr_sub.std(),
        }
        print(f"{method:12s} AUC={row['auc_media']:.4f}±{row['auc_std']:.4f}  "
              f"Sens={row['sens_media']:.4f}  Spec={row['spec_media']:.4f}  "
              f"Thr={row['youden_thr_media']:.4f}±{row['youden_thr_std']:.4f}")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    det.to_csv(f"{out_prefix}_nested_oof_detail.csv", index=False)
    thresholds_df.to_csv(f"{out_prefix}_nested_youden_por_fold.csv", index=False)
    per_repeat_df.to_csv(f"{out_prefix}_nested_metricas_por_repeat.csv", index=False)
    summary_df.to_csv(f"{out_prefix}_nested_resumen.csv", index=False)
    baseline_summary_df.to_csv(f"{out_prefix}_nested_baseline_individual.csv", index=False)

    return det, thresholds_df, per_repeat_df, summary_df, baseline_summary_df


if __name__ == "__main__":
    run_nested_late_fusion(
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/stacking_output/oof_matrix_detail_by_repeat.csv",
        out_prefix="fusion_nested",
    )