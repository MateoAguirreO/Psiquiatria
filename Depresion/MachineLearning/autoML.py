import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    cross_val_predict,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.calibration import CalibratedClassifierCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
import shap

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_15.csv"
TARGET = "label"
GROUP_COLUMN = "audio_id"

N_SPLITS = 5
N_TRIALS = 40
THRESHOLD_METRIC = "f1"

OUTPUT_DIR = "outputs_automl"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CARGA DE DATOS
# ============================================================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")
print(df.head())


# ============================================================
# VALIDACIONES
# ============================================================

assert TARGET in df.columns, "No existe la columna target"
assert GROUP_COLUMN in df.columns, "No existe la columna de grupos"

print("\nDistribución clases:")
print(df[TARGET].value_counts())


# ============================================================
# ELIMINAR COLUMNAS NO ÚTILES
# ============================================================

remove_cols = [
    TARGET,
    GROUP_COLUMN,
    "seg_idx",
]

X = df.drop(columns=remove_cols, errors="ignore")
y = df[TARGET]
groups = df[GROUP_COLUMN]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nAplicando feature engineering...")


def create_interaction_features(dataframe):
    df_new = dataframe.copy()

    mfcc_cols = [c for c in df_new.columns if "mfcc" in c and "mean" in c]

    if len(mfcc_cols) > 3:
        df_new["mfcc_energy_mean"] = df_new[mfcc_cols].mean(axis=1)
        df_new["mfcc_energy_std"] = df_new[mfcc_cols].std(axis=1)

    shimmer_cols = [c for c in df_new.columns if "shimmer" in c]
    jitter_cols = [c for c in df_new.columns if "jitter" in c]

    if shimmer_cols:
        df_new["shimmer_global"] = df_new[shimmer_cols].mean(axis=1)

    if jitter_cols:
        df_new["jitter_global"] = df_new[jitter_cols].mean(axis=1)

    formant_cols = [c for c in df_new.columns if "f1_" in c or "f2_" in c or "f3_" in c]

    if formant_cols:
        df_new["formant_global_mean"] = df_new[formant_cols].mean(axis=1)

    return df_new


X = create_interaction_features(X)


# ============================================================
# PREPROCESAMIENTO
# ============================================================


def build_cv():
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    return GroupKFold(n_splits=N_SPLITS)


def min_class_count_in_folds(cv, X_data, y_data, group_data):
    y_array = np.asarray(y_data)
    min_count = None

    for train_idx, _ in cv.split(X_data, y_array, group_data):
        _, counts = np.unique(y_array[train_idx], return_counts=True)

        if len(counts) < 2:
            return 0

        fold_min = counts.min()
        min_count = fold_min if min_count is None else min(min_count, fold_min)

    return 0 if min_count is None else int(min_count)


def build_resampler(min_class_count):
    if min_class_count < 2:
        print("Aviso: folds con clase minoritaria insuficiente; usando RandomOverSampler.")
        return RandomOverSampler(random_state=SEED)

    k_neighbors = min(5, min_class_count - 1)
    return SMOTE(random_state=SEED, k_neighbors=k_neighbors)

numeric_features = X.columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("variance", VarianceThreshold(threshold=0.0001)),
    ("scaler", RobustScaler()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features)
])


cv = build_cv()
min_class_count = min_class_count_in_folds(cv, X, y, groups)
resampler = build_resampler(min_class_count)


# ============================================================
# FEATURE SELECTION
# ============================================================

feature_selector = SelectFromModel(
    ExtraTreesClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    ),
    threshold="median",
)


# ============================================================
# MÉTRICAS
# ============================================================


def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    return metrics


def find_best_threshold(y_true, y_prob, metric="f1"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if thresholds.size == 0:
        return 0.5

    if metric in {"f1", "f2"}:
        beta = 1.0 if metric == "f1" else 2.0
        beta_sq = beta ** 2
        f_scores = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f_scores[:-1]))
        return float(thresholds[best_idx])

    return 0.5


# ============================================================
# OPTUNA - XGBOOST
# ============================================================


def objective_xgb(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "auc",
        "tree_method": "hist",
    }

    model = XGBClassifier(**params)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("smote", resampler),
        ("classifier", model),
    ])

    probs = cross_val_predict(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    score = roc_auc_score(y, probs)

    return score


# ============================================================
# ENTRENAMIENTO XGBOOST
# ============================================================

print("\nOptimizando XGBoost...")

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)

print("Best XGB AUC:", study_xgb.best_value)
print("Best Params:")
print(study_xgb.best_params)


# ============================================================
# MODELOS FINALES
# ============================================================

best_xgb = XGBClassifier(
    **study_xgb.best_params,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="auc",
    tree_method="hist",
)

lgbm = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.8,
    force_col_wise=True,
    verbosity=-1,
    random_state=SEED,
)

catboost = CatBoostClassifier(
    iterations=700,
    learning_rate=0.03,
    depth=6,
    verbose=0,
    random_state=SEED,
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)


# ============================================================
# STACKING ENSEMBLE
# ============================================================

stack_model = StackingClassifier(
    estimators=[
        ("xgb", best_xgb),
        ("lgbm", lgbm),
        ("cat", catboost),
        ("rf", rf),
    ],
    final_estimator=LogisticRegression(max_iter=5000),
    stack_method="predict_proba",
    n_jobs=-1,
)


# ============================================================
# PIPELINE FINAL
# ============================================================

final_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", feature_selector),
    ("smote", resampler),
    ("classifier", stack_model),
])


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

print("\nEjecutando validación final...")

final_probs = cross_val_predict(
    final_pipeline,
    X,
    y,
    groups=groups,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

best_threshold = find_best_threshold(y, final_probs, metric=THRESHOLD_METRIC)
metrics = evaluate_model(y, final_probs, threshold=best_threshold)
metrics["threshold"] = best_threshold

print("\nRESULTADOS FINALES")
print("=" * 60)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

preds = (final_probs >= best_threshold).astype(int)

cm = confusion_matrix(y, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()


# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

report = classification_report(y, preds)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print(report)


# ============================================================
# ENTRENAMIENTO FINAL COMPLETO
# ============================================================

print("\nEntrenando modelo final con todo el dataset...")

final_pipeline.fit(X, y)


# ============================================================
# EXPORTAR MODELO
# ============================================================

joblib.dump(final_pipeline, f"{OUTPUT_DIR}/best_model.pkl")

print("Modelo guardado")


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

print("\nCalculando SHAP values...")

try:

    preprocessor_fitted = final_pipeline.named_steps["preprocessor"]
    transformed_X = preprocessor_fitted.transform(X)

    selector = final_pipeline.named_steps["feature_selection"]
    selected_mask = selector.get_support()

    try:
        feature_names = preprocessor_fitted.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(X.columns)

    selected_features = np.array(feature_names)[selected_mask]

    transformed_X = selector.transform(transformed_X)

    xgb_model = best_xgb
    xgb_model.fit(transformed_X, y)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transformed_X)

    plt.figure()

    shap.summary_plot(
        shap_values,
        transformed_X,
        feature_names=selected_features,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png")
    plt.close()

except Exception as e:
    print("Error SHAP:", e)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# ============================================================
# INFERENCIA NUEVOS DATOS
# ============================================================


def predict_new_data(csv_path):

    model = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")

    new_df = pd.read_csv(csv_path)

    drop_cols = [GROUP_COLUMN, TARGET, "seg_idx"]

    X_new = new_df.drop(columns=drop_cols, errors="ignore")

    X_new = create_interaction_features(X_new)

    probs = model.predict_proba(X_new)[:, 1]

    preds = (probs >= best_threshold).astype(int)

    results = pd.DataFrame({
        "probability_depression": probs,
        "prediction": preds,
    })

    return results


print("\nPipeline finalizado correctamente")