"""
Fusión tardía (late fusion) de modelos usando las probabilidades OOF
(out-of-fold) de los 4 modelos base:
    - prob_extratrees_embeddings
    - prob_bilstm_embeddings
    - prob_destilacion_estudiante
    - prob_bigru_video

Se implementan 5 estrategias de fusión:
    1. Perceptrón (modelo lineal simple, calibrado para dar probabilidades)
    2. Regresión logística (modelo lineal)
    3. Votación (mayoría simple / soft voting basada en las 4 probabilidades)
    4. Gradient Boosting (ensemble de árboles, captura relaciones no lineales)
    5. MLP - perceptrón multicapa (red neuronal con una capa oculta)

Para evitar fuga de información, los modelos de fusión se entrenan y evalúan
respetando la misma estructura de repeticiones (repeat) y particiones (fold)
que ya trae el archivo `oof_matrix_detail_by_repeat.csv`:
    - Para cada repeat, se hace un 5-fold CV: en cada iteración se entrena
      la fusión con 4 folds y se predice el fold restante (out-of-fold).
    - Así, cada sujeto obtiene una predicción de fusión para modelos que
      nunca lo vieron en el entrenamiento, tanto a nivel de modelos base
      como a nivel del meta-modelo de fusión (evaluación anidada / nested CV).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

FEATURE_COLS = [
    "prob_extratrees_embeddings",
    "prob_bilstm_embeddings",
    "prob_destilacion_estudiante",
    "prob_bigru_video",
]


def fusion_voting(X):
    """
    Fusión por votación (soft voting):
    - Cada modelo base "vota" 1 si su probabilidad > 0.5, si no vota 0.
    - La probabilidad de fusión es la proporción de modelos que votaron 1.
    No requiere entrenamiento, es una regla fija.
    """
    votes = (X > 0.5).astype(int)
    return votes.mean(axis=1)


def fit_perceptron(X_train, y_train):
    """
    Perceptrón simple (modelo lineal, sin capa oculta) calibrado con
    CalibratedClassifierCV para poder obtener probabilidades
    (el Perceptrón clásico solo da clases, no probabilidades).
    """
    base = Perceptron(random_state=RANDOM_STATE, max_iter=1000)
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X_train, y_train)
    return clf


def fit_logreg(X_train, y_train):
    """Regresión logística estándar (modelo lineal para clasificación)."""
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def fit_gradient_boosting(X_train, y_train):
    """
    Gradient Boosting: ensemble de árboles poco profundos entrenados
    secuencialmente, cada uno corrigiendo los errores del anterior.
    Captura relaciones no lineales entre los 4 modelos base.
    """
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def fit_mlp(X_train, y_train):
    """
    MLP (perceptrón multicapa): red neuronal con una capa oculta,
    permite capturar interacciones no lineales entre los modelos base
    (a diferencia del Perceptrón simple, que es puramente lineal).
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def run_late_fusion(detail_path, out_prefix="/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/fusion"):
    det = pd.read_csv(detail_path)

    # Guardamos las predicciones OOF de fusión para cada método
    det["fusion_perceptron"] = np.nan
    det["fusion_logreg"] = np.nan
    det["fusion_voting"] = np.nan
    det["fusion_gboost"] = np.nan
    det["fusion_mlp"] = np.nan

    per_repeat_auc = {
        "perceptron": [],
        "logreg": [],
        "voting": [],
        "gboost": [],
        "mlp": [],
    }

    for repeat, g_repeat in det.groupby("repeat"):
        folds = sorted(g_repeat["fold"].unique())

        for fold in folds:
            test_mask = (det["repeat"] == repeat) & (det["fold"] == fold)
            train_mask = (det["repeat"] == repeat) & (det["fold"] != fold)

            X_train = det.loc[train_mask, FEATURE_COLS].values
            y_train = det.loc[train_mask, "y_true"].values
            X_test = det.loc[test_mask, FEATURE_COLS].values

            # Escalamos para los modelos lineales (perceptrón / regresión)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # --- Perceptrón ---
            perc = fit_perceptron(X_train_s, y_train)
            det.loc[test_mask, "fusion_perceptron"] = perc.predict_proba(X_test_s)[:, 1]

            # --- Regresión logística ---
            logreg = fit_logreg(X_train_s, y_train)
            det.loc[test_mask, "fusion_logreg"] = logreg.predict_proba(X_test_s)[:, 1]

            # --- Votación (no requiere entrenamiento) ---
            det.loc[test_mask, "fusion_voting"] = fusion_voting(X_test)

            # --- Gradient Boosting ---
            # Los árboles no necesitan features escaladas, se usa X sin escalar
            gboost = fit_gradient_boosting(X_train, y_train)
            det.loc[test_mask, "fusion_gboost"] = gboost.predict_proba(X_test)[:, 1]

            # --- MLP (perceptrón multicapa) ---
            mlp = fit_mlp(X_train_s, y_train)
            det.loc[test_mask, "fusion_mlp"] = mlp.predict_proba(X_test_s)[:, 1]

        # AUC de la repetición completa (uniendo todos los folds del repeat)
        y_repeat = g_repeat["y_true"].values
        idx_repeat = det["repeat"] == repeat
        for name, col in [
            ("perceptron", "fusion_perceptron"),
            ("logreg", "fusion_logreg"),
            ("voting", "fusion_voting"),
            ("gboost", "fusion_gboost"),
            ("mlp", "fusion_mlp"),
        ]:
            auc = roc_auc_score(det.loc[idx_repeat, "y_true"], det.loc[idx_repeat, col])
            per_repeat_auc[name].append(auc)

    # ---- Resumen por repetición ----
    print("=== AUC de fusión por repetición (media ± DE sobre 10 repeats) ===")
    summary_rows = []
    for name, aucs in per_repeat_auc.items():
        mean, std = np.mean(aucs), np.std(aucs)
        print(f"{name:12s} AUC medio = {mean:.4f}  ± {std:.4f}")
        summary_rows.append({"metodo": name, "auc_medio": mean, "auc_std": std})

    pd.DataFrame(summary_rows).to_csv(f"{out_prefix}_resumen_por_repeat.csv", index=False)

    # ---- Predicción final consolidada por sujeto (promedio entre repeats) ----
    final = (
        det.groupby("subject_id")[
            [
                "fusion_perceptron",
                "fusion_logreg",
                "fusion_voting",
                "fusion_gboost",
                "fusion_mlp",
                "y_true",
            ]
        ]
        .mean()  # y_true es constante por sujeto, promediar no cambia el valor
        .reset_index()
    )

    print("\n=== AUC de fusión (predicción final consolidada por sujeto) ===")
    final_rows = []
    for name, col in [
        ("perceptron", "fusion_perceptron"),
        ("logreg", "fusion_logreg"),
        ("voting", "fusion_voting"),
        ("gboost", "fusion_gboost"),
        ("mlp", "fusion_mlp"),
    ]:
        auc = roc_auc_score(final["y_true"], final[col])
        print(f"{name:12s} AUC = {auc:.4f}")
        final_rows.append({"metodo": name, "auc_final": auc})

    pd.DataFrame(final_rows).to_csv(f"{out_prefix}_resumen_final.csv", index=False)
    final.to_csv(f"{out_prefix}_predicciones_finales.csv", index=False)
    det.to_csv(f"{out_prefix}_predicciones_detalle.csv", index=False)

    return det, final, per_repeat_auc


if __name__ == "__main__":
    run_late_fusion("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/stacking_output/oof_matrix_detail_by_repeat.csv")