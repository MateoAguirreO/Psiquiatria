"""
==============================================================================
late_fusion_final.py
==============================================================================
Fusión tardía (late fusion) con CV ANIDADO sobre la MATRIZ FINAL (agregada
por sujeto) que genera build_oof_matrix.py -- es decir, sobre
oof_matrix_final.csv / oof_matrix_final_con_kd.csv, NO sobre el detalle por
repetición.

──────────────────────────────────────────────────────────────────────────
DIFERENCIA CLAVE respecto a late_fusion_nested.py
──────────────────────────────────────────────────────────────────────────
La matriz final NO trae columnas "repeat"/"fold": cada prob_* es el
PROMEDIO de esa probabilidad a través de las 10 repeticiones
(build_final_matrix hace detail.groupby("subject_id")[prob_cols].mean()).
Por lo tanto aquí no existe ya un split externo que "reusar" -- hay que
generar uno nuevo con StratifiedKFold sobre la matriz final.

Esto NO reintroduce leakage: cada valor prob_* de un sujeto, en cada una
de las 10 repeticiones que promedia, fue calculado por un modelo base que
tenía a ese sujeto en el fold de test (es decir, nunca fue usado para
entrenar ese modelo en ninguna repetición). Da igual qué partición nueva
usemos para evaluar el FUSOR: el sujeto de test del fusor nunca vio sus
propios prob_* "contaminados" con su propia etiqueta.

Igual que en late_fusion_nested.py:
  Externo: 5 folds estratificados (por defecto SIN repetir, para mantener
      la misma filosofía "5 folds, una sola partición" del script
      original). N_OUTER_REPEATS es configurable si luego se quiere más
      robustez estadística (el dataset es chico: 79 sujetos).
  Interno: dentro de cada train externo, split estratificado 80/20
      -> train_interno: ajusta el modelo de fusión.
      -> val_interno:   calcula el umbral óptimo de Youden (J = TPR - FPR).
El test externo nunca se toca para entrenar ni para calibrar el umbral.

Reutiliza el mismo diseño de reportes (classification_report por fold +
matrices de confusión + Excel + gráficos) que late_fusion_nested.py.
==============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
INNER_VAL_SIZE = 0.20
METHODS = ["voting"]

# Externo: N_SPLITS folds x N_OUTER_REPEATS repeticiones. Por defecto 1
# repetición (5 folds "puros", sin repetir) para no alejarse del diseño
# de late_fusion_nested.py. Si más adelante quieres una estimación más
# estable (79 sujetos es poco), sube N_OUTER_REPEATS (p.ej. 10) -- el
# resto del script no necesita cambios, cada combinación (repeat, fold)
# simplemente se trata como un "fold" más a la hora de agregar métricas.
N_SPLITS = 5
N_OUTER_REPEATS = 1

# Rutas producidas por build_oof_matrix.py (mismo OUTPUT_DIR que allá)
STACKING_OUTPUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/stacking_output")


def find_existing(*candidates: Path) -> Path | None:
    """build_oof_matrix.py nombra los archivos con sufijo _con_kd/_sin_kd,
    pero corridas viejas pueden no tener sufijo (p.ej. 'oof_matrix_final.csv'
    a secas para la variante sin destilación). Probamos varios nombres."""
    for c in candidates:
        if c.exists():
            return c
    return None


FINAL_CON_KD = find_existing(
    STACKING_OUTPUT_DIR / "oof_matrix_final_con_kd.csv",
)
FINAL_SIN_KD = find_existing(
    STACKING_OUTPUT_DIR / "oof_matrix_final_sin_kd.csv",
    STACKING_OUTPUT_DIR / "oof_matrix_final.csv",  # nombre sin sufijo (corrida vieja)
)

OUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/fusion_output_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Modelos de fusión (idénticos a late_fusion_nested.py)
# ──────────────────────────────────────────────────────────────────────────
def fit_perceptron(X, y):
    return Perceptron(random_state=RANDOM_STATE).fit(X, y)


def fit_logreg(X, y):
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(X, y)


def fit_gradient_boosting(X, y):
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
# Reportes sklearn + matrices de confusión a nivel de fold externo
# (idéntico a late_fusion_nested.py: agrupa por "fold", que aquí es un id
# de iteración externa generado por nosotros, no por un manifest)
# ──────────────────────────────────────────────────────────────────────────
def per_fold_sklearn_reports(det: pd.DataFrame, methods: list, out_prefix: str):
    report_rows = []
    cm_rows = []
    for fold, g in det.groupby("fold"):
        y_true = g["y_true"].values
        for method in methods:
            y_pred = g[f"fusion_{method}_pred"].values
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            for label, metrics_dict in rep.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        report_rows.append({
                            "fold": fold, "metodo": method,
                            "clase": label, "metrica": metric_name, "valor": value,
                        })
                else:
                    report_rows.append({
                        "fold": fold, "metodo": method,
                        "clase": "accuracy", "metrica": "accuracy", "valor": metrics_dict,
                    })
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            cm_rows.append({
                "fold": fold, "metodo": method,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "n_test": len(y_true),
            })
    report_df = pd.DataFrame(report_rows)
    cm_df = pd.DataFrame(cm_rows)
    resumen_df = (
        report_df.groupby(["metodo", "clase", "metrica"])["valor"]
        .agg(media="mean", std="std", n_folds="count")
        .reset_index()
        .sort_values(["metodo", "clase", "metrica"])
    )
    cm_resumen_df = (
        cm_df.groupby("metodo")[["TN", "FP", "FN", "TP"]]
        .agg(["mean", "std"])
    )
    cm_resumen_df.columns = ["_".join(c) for c in cm_resumen_df.columns]
    cm_resumen_df = cm_resumen_df.reset_index()

    report_df.to_csv(f"{out_prefix}_sklearn_reporte_por_fold.csv", index=False)
    cm_df.to_csv(f"{out_prefix}_sklearn_matrices_confusion_por_fold.csv", index=False)
    resumen_df.to_csv(f"{out_prefix}_sklearn_resumen_metricas.csv", index=False)
    cm_resumen_df.to_csv(f"{out_prefix}_sklearn_resumen_matrices_confusion.csv", index=False)

    excel_path = f"{out_prefix}_sklearn_reporte.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumen_df.to_excel(writer, sheet_name="resumen_metricas", index=False)
        cm_resumen_df.to_excel(writer, sheet_name="resumen_matrices_conf", index=False)
        report_df.to_excel(writer, sheet_name="detalle_por_fold", index=False)
        cm_df.to_excel(writer, sheet_name="matrices_confusion", index=False)
        for method in methods:
            sub = cm_df[cm_df["metodo"] == method].sort_values("fold")
            sheet_name = f"cm_{method}"[:31]
            sub.to_excel(writer, sheet_name=sheet_name, index=False)
    return report_df, cm_df, resumen_df, cm_resumen_df, excel_path


# ──────────────────────────────────────────────────────────────────────────
# Gráficos (idénticos a late_fusion_nested.py)
# ──────────────────────────────────────────────────────────────────────────
def build_metrics_table_por_metodo(report_df: pd.DataFrame, per_fold_df: pd.DataFrame, method: str) -> pd.DataFrame:
    folds = sorted(report_df.loc[report_df["metodo"] == method, "fold"].unique())
    rows = []
    for fold in folds:
        sub = report_df[(report_df["metodo"] == method) & (report_df["fold"] == fold)]
        macro = sub[sub["clase"] == "macro avg"].set_index("metrica")["valor"]
        acc = sub[sub["clase"] == "accuracy"]["valor"].values[0]
        auc_val = per_fold_df[(per_fold_df["metodo"] == method) & (per_fold_df["fold"] == fold)]["auc"].values[0]
        rows.append({
            "Fold": fold,
            "Accuracy": acc,
            "Precision (macro)": macro.get("precision", np.nan),
            "Recall (macro)": macro.get("recall", np.nan),
            "F1 (macro)": macro.get("f1-score", np.nan),
            "AUC": auc_val,
        })
    tabla = pd.DataFrame(rows).set_index("Fold")
    tabla.loc["Media"] = tabla.mean()
    tabla.loc["DE"] = tabla.drop(index="Media").std()
    return tabla.round(3)


def save_table_as_image(tabla: pd.DataFrame, title: str, out_path: Path):
    n_rows, n_cols = tabla.shape
    fig, ax = plt.subplots(figsize=(1.6 * (n_cols + 1) + 1, 0.45 * (n_rows + 2)))
    ax.axis("off")
    row_labels = [str(i) for i in tabla.index]
    col_labels = list(tabla.columns)
    cell_text = tabla.values.astype(str).tolist()
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif c == -1:
            cell.set_text_props(weight="bold")
            if row_labels[r - 1] in ("Media", "DE"):
                cell.set_facecolor("#D9E1F2")
        if row_labels[r - 1] in ("Media", "DE") and r != 0:
            cell.set_facecolor("#D9E1F2")
            cell.set_text_props(weight="bold")
    plt.title(title, fontsize=12, weight="bold", pad=14)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices_image(cm_df: pd.DataFrame, method: str, title: str, out_path: Path):
    sub = cm_df[cm_df["metodo"] == method].sort_values("fold")
    n_folds = len(sub)
    fig, axes = plt.subplots(1, n_folds, figsize=(3.0 * n_folds, 3.0))
    if n_folds == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sub.iterrows()):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        thresh = cm.max() / 2 if cm.max() > 0 else 0
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
        ax.set_title(f"Fold {row['fold']}", fontsize=9)
    fig.suptitle(title, fontsize=13, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_across_methods_image(report_df: pd.DataFrame, per_fold_df: pd.DataFrame,
                                          methods: list, out_path: Path):
    metric_specs = [
        ("Accuracy", "accuracy", "accuracy"),
        ("Precision (macro)", "macro avg", "precision"),
        ("Recall (macro)", "macro avg", "recall"),
        ("F1 (macro)", "macro avg", "f1-score"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (label, clase, metrica) in zip(axes[:4], metric_specs):
        medias, stds = [], []
        for method in methods:
            vals = report_df[(report_df["metodo"] == method) &
                              (report_df["clase"] == clase) &
                              (report_df["metrica"] == metrica)]["valor"]
            medias.append(vals.mean())
            stds.append(vals.std())
        ax.bar(methods, medias, yerr=stds, capsize=4, color="#4472C4")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=40)
    ax = axes[4]
    medias, stds = [], []
    for method in methods:
        vals = per_fold_df[per_fold_df["metodo"] == method]["auc"]
        medias.append(vals.mean())
        stds.append(vals.std())
    ax.bar(methods, medias, yerr=stds, capsize=4, color="#ED7D31")
    ax.set_title("AUC", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Comparación entre métodos de fusión (media \u00b1 DE entre folds)",
                 fontsize=13, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_visual_reports(report_df: pd.DataFrame, cm_df: pd.DataFrame,
                           per_fold_df: pd.DataFrame, methods: list, out_prefix: str):
    graficos_dir = Path(f"{out_prefix}_graficos")
    graficos_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        tabla = build_metrics_table_por_metodo(report_df, per_fold_df, method)
        save_table_as_image(
            tabla, f"Métricas por fold -- {method}",
            graficos_dir / f"tabla_metricas_{method}.png",
        )
        save_confusion_matrices_image(
            cm_df, method, f"Matrices de confusión por fold -- {method}",
            graficos_dir / f"matrices_confusion_{method}.png",
        )
    save_comparison_across_methods_image(
        report_df, per_fold_df, methods,
        graficos_dir / "comparacion_metodos.png",
    )
    print(f"  -> Gráficos (tablas + matrices de confusión + comparación) en: {graficos_dir}/")
    return graficos_dir


# ──────────────────────────────────────────────────────────────────────────
# Fusión con CV anidado sobre la MATRIZ FINAL (sin fold/repeat propios ->
# generamos el split externo nosotros mismos)
# ──────────────────────────────────────────────────────────────────────────
def run_nested_late_fusion_final(final_path: Path, feature_cols: list, out_prefix: str):
    det = pd.read_csv(final_path)
    for c in feature_cols + ["y_true", "subject_id"]:
        assert c in det.columns, f"{final_path}: falta la columna '{c}'."
    assert det["subject_id"].is_unique, (
        f"{final_path}: se esperaba una fila por sujeto (matriz final agregada), "
        f"pero hay subject_id repetidos -- ¿es en realidad el detalle por repeat?"
    )
    det = det.reset_index(drop=True).copy()

    # Generamos el split externo (no existe ya un "fold" en la matriz final).
    if N_OUTER_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=N_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=RANDOM_STATE,
        )
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    det["fold"] = -1
    for fold_id, (_, test_pos) in enumerate(splitter.split(det[feature_cols], det["y_true"])):
        det.loc[det.index[test_pos], "fold"] = fold_id
    assert (det["fold"] >= 0).all()

    n_outer_iters = det["fold"].nunique()
    print(f"  Split externo generado: {N_SPLITS} folds x {N_OUTER_REPEATS} repeticion(es) "
          f"= {n_outer_iters} iteraciones externas, {len(det)} sujetos totales.")

    for m in METHODS:
        det[f"fusion_{m}"] = np.nan
        det[f"fusion_{m}_pred"] = np.nan

    thresholds_rows = []
    outer_folds = sorted(det["fold"].unique())
    for outer_fold in outer_folds:
        test_mask = det["fold"] == outer_fold
        train_mask = det["fold"] != outer_fold
        train_idx = det.index[train_mask]

        X_train_full = det.loc[train_idx, feature_cols].values
        y_train_full = det.loc[train_idx, "y_true"].values
        X_test_raw = det.loc[test_mask, feature_cols].values

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=INNER_VAL_SIZE,
            random_state=RANDOM_STATE + outer_fold,
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
                "fold": outer_fold, "metodo": method,
                "youden_threshold": thr, "youden_j": j_stat,
            })

    thresholds_df = pd.DataFrame(thresholds_rows)

    baseline_rows = []
    for outer_fold, g_fold in det.groupby("fold"):
        y_true_fold = g_fold["y_true"].values
        for col in feature_cols:
            auc = roc_auc_score(y_true_fold, g_fold[col].values)
            baseline_rows.append({"fold": outer_fold, "metodo": col, "auc": auc})
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_summary_df = (
        baseline_df.groupby("metodo")["auc"].agg(auc_media="mean", auc_std="std").reset_index()
    )

    per_fold_rows = []
    for outer_fold, g_fold in det.groupby("fold"):
        y_true_fold = g_fold["y_true"].values
        for method in METHODS:
            probs = g_fold[f"fusion_{method}"].values
            preds = g_fold[f"fusion_{method}_pred"].values
            auc = roc_auc_score(y_true_fold, probs)
            tn, fp, fn, tp = confusion_matrix(y_true_fold, preds).ravel()
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            per_fold_rows.append({"fold": outer_fold, "metodo": method,
                                   "auc": auc, "sensibilidad": sens, "especificidad": spec})
    per_fold_df = pd.DataFrame(per_fold_rows)

    summary_rows = []
    for method in METHODS:
        sub = per_fold_df[per_fold_df["metodo"] == method]
        thr_sub = thresholds_df[thresholds_df["metodo"] == method]["youden_threshold"]
        summary_rows.append({
            "metodo": method,
            "auc_media": sub["auc"].mean(), "auc_std": sub["auc"].std(),
            "sens_media": sub["sensibilidad"].mean(), "sens_std": sub["sensibilidad"].std(),
            "spec_media": sub["especificidad"].mean(), "spec_std": sub["especificidad"].std(),
            "youden_thr_media": thr_sub.mean(), "youden_thr_std": thr_sub.std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    det.to_csv(f"{out_prefix}_oof_detail.csv", index=False)
    thresholds_df.to_csv(f"{out_prefix}_youden_por_fold.csv", index=False)
    per_fold_df.to_csv(f"{out_prefix}_metricas_por_fold.csv", index=False)
    summary_df.to_csv(f"{out_prefix}_resumen.csv", index=False)
    baseline_summary_df.to_csv(f"{out_prefix}_baseline_individual.csv", index=False)

    (report_df, cm_df, sklearn_resumen_df,
     cm_resumen_df, excel_path) = per_fold_sklearn_reports(det, METHODS, out_prefix)
    print(f"  -> Reporte sklearn por fold + matrices de confusión escritos en: {excel_path}")

    export_visual_reports(report_df, cm_df, per_fold_df, METHODS, out_prefix)
    return summary_df, baseline_summary_df, sklearn_resumen_df, cm_resumen_df


# ──────────────────────────────────────────────────────────────────────────
# Orquestador: corre con_kd y sin_kd (si existen) y compara
# ──────────────────────────────────────────────────────────────────────────
FEATURE_COLS_BASE = ["prob_extratrees_embeddings", "prob_bilstm_embeddings", "prob_bigru_video"]
FEATURE_COLS_CON_KD = FEATURE_COLS_BASE + ["prob_destilacion_estudiante"]


def main():
    results = {}

    if FINAL_CON_KD is not None:
        print(f"=== Fusión CON destilación ({FINAL_CON_KD}) ===")
        summary_con, baseline_con, sk_resumen_con, cm_resumen_con = run_nested_late_fusion_final(
            FINAL_CON_KD, FEATURE_COLS_CON_KD, str(OUT_DIR / "con_kd")
        )
        print(baseline_con.to_string(index=False))
        print(summary_con.to_string(index=False))
        print("\n-- Resumen de matrices de confusión (media +/- std entre folds) --")
        print(cm_resumen_con.to_string(index=False))
        results["con_kd"] = summary_con
    else:
        print("(no encontrado: oof_matrix_final_con_kd.csv, se omite)")

    if FINAL_SIN_KD is not None:
        print(f"\n=== Fusión SIN destilación ({FINAL_SIN_KD}) ===")
        summary_sin, baseline_sin, sk_resumen_sin, cm_resumen_sin = run_nested_late_fusion_final(
            FINAL_SIN_KD, FEATURE_COLS_BASE, str(OUT_DIR / "sin_kd")
        )
        print(baseline_sin.to_string(index=False))
        print(summary_sin.to_string(index=False))
        print("\n-- Resumen de matrices de confusión (media +/- std entre folds) --")
        print(cm_resumen_sin.to_string(index=False))
        results["sin_kd"] = summary_sin
    else:
        print("(no encontrado: oof_matrix_final_sin_kd.csv / oof_matrix_final.csv, se omite)")

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
    else:
        print(
            "\nPara la comparación automática hacen falta ambas matrices finales "
            "(con_kd y sin_kd)."
        )


if __name__ == "__main__":
    main()