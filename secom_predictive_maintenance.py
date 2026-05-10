from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("uci-secom.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "secom_hgb_artifact.joblib"
METRICS_PATH = ARTIFACT_DIR / "secom_metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "secom_confusion_matrix.png"
MIN_VALIDATION_RECALL = 0.25


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Pass/Fail" not in df.columns:
        raise ValueError("La colonne cible 'Pass/Fail' est introuvable.")
    if "Time" not in df.columns:
        raise ValueError("La colonne 'Time' est introuvable.")

    time_index = pd.to_datetime(df["Time"], errors="coerce")
    features = df.drop(columns=["Time", "Pass/Fail"]).copy()

    hour = time_index.dt.hour.astype(float)
    dayofweek = time_index.dt.dayofweek.astype(float)
    month = time_index.dt.month.astype(float)

    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    features["dow_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    features["dow_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    features["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    target = (df["Pass/Fail"] == 1).astype(int)
    return features, target


def prepare_data(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SimpleImputer,
    StandardScaler,
]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.35,
        stratify=target,
        random_state=random_state,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=4.0 / 7.0,
        stratify=y_temp,
        random_state=random_state,
    )

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train)
    x_val_imputed = imputer.transform(x_val)
    x_test_imputed = imputer.transform(x_test)

    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_val_scaled = scaler.transform(x_val_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        imputer,
        scaler,
    )
def train_model(x_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_depth=5,
        learning_rate=0.05,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: HistGradientBoostingClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "support_positive": int((y_test == 1).sum()),
        "support_negative": int((y_test == 0).sum()),
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = recall[:-1] >= MIN_VALIDATION_RECALL
    if np.any(valid):
        best_precision = np.max(np.where(valid, precision[:-1], -1.0))
        precision_match = valid & (precision[:-1] == best_precision)
        if np.any(precision_match):
            best_index = int(np.argmax(np.where(precision_match, recall[:-1], -1.0)))
        else:
            best_index = int(np.argmax(np.where(valid, precision[:-1], -1.0)))
    else:
        f1_scores = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title("SECOM - Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    features, target = build_features(df)
    x_train, x_val, x_test, y_train, y_val, y_test, imputer, scaler = prepare_data(features, target)
    model = train_model(x_train, y_train)

    val_score = model.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, val_score)

    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = evaluate_model(model, x_test, y_test, threshold)
    metrics["threshold"] = float(threshold)

    print("=== SECOM Predictive Maintenance ===")
    print(f"Samples: {len(df)}")
    print(f"Features after engineering: {features.shape[1]}")
    print(f"Chosen threshold from validation: {threshold:.4f}")
    print(f"Minimum validation recall target: {MIN_VALIDATION_RECALL:.2f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    save_confusion_matrix(y_test, y_pred)
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "feature_columns": list(features.columns),
            "threshold": float(threshold),
        },
        MODEL_PATH,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved model artifact: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()