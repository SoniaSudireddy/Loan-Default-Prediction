import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import CreditFeatureEngineer

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Loan_default.csv"
MODEL_PATH = BASE_DIR / "loan_default_model.pkl"
CONFIG_PATH = BASE_DIR / "model_config.json"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

ENGINEERED_NUMERIC_FEATURES = [
    "LoanToIncome",
    "EstimatedMonthlyPayment",
    "PaymentToIncomeRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
    "CreditScoreTier",
]

REQUIRED_COLUMNS = (
    NUMERIC_FEATURES
    + [
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner",
        "Default",
    ]
)

def build_pipeline(classifier):

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent"),),
            ("encoder",OneHotEncoder(handle_unknown="ignore"),),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("features", CreditFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline

def evaluate_predictions(y_true, probabilities, threshold):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Actual labels and predicted probabilities "
            "must have the same length."
        )

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Predicted probabilities contain invalid values."
        )

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOAN DEFAULT PREDICTION - MODEL TRAINING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Loan_default.csv beside train_and_save.py."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape before cleaning: {df.shape}")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{missing_columns}"
        )

    df = df.drop(columns=["LoanID"], errors="ignore")

    if df["Default"].isna().any():
        raise ValueError(
            "The Default target contains missing values. "
            "Investigate and resolve them before training."
        )

    target_values = set(df["Default"].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Default must contain only binary values 0 and 1."
        )

    if len(target_values) != 2:
        raise ValueError(
            "The dataset must contain both Default=0 and Default=1."
        )

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        print(f"Removing {duplicate_count} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Default"])
    y = df["Default"].astype(int)

    print(f"Dataset shape after cleaning: {df.shape}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    assert X_train.index.equals(y_train.index)
    assert X_val.index.equals(y_val.index)
    assert X_test.index.equals(y_test.index)

    print("\nData split sizes:")
    print(f"Training:   {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Testing:    {len(X_test):,}")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    positive_weight = negative_count / max(positive_count, 1)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            scale_pos_weight=positive_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    comparison_results = []
    fitted_models = {}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(clone(classifier))
        pipeline.fit(X_train, y_train)

        validation_probabilities = (
            pipeline.predict_proba(X_val)[:, 1]
        )

        row = {
            "model": name,
            "validation_accuracy_at_0.50": float(
                accuracy_score(
                    y_val,
                    (
                        validation_probabilities >= 0.50
                    ).astype(int),
                )
            ),
            "validation_roc_auc": float(
                roc_auc_score(
                    y_val,
                    validation_probabilities,
                )
            ),
            "validation_average_precision": float(
                average_precision_score(
                    y_val,
                    validation_probabilities,
                )
            ),
            "validation_f1_at_0.50": float(
                f1_score(
                    y_val,
                    (
                        validation_probabilities >= 0.50
                    ).astype(int),
                    zero_division=0,
                )
            ),
        }

        comparison_results.append(row)
        fitted_models[name] = pipeline

        print(
            f"ROC-AUC: {row['validation_roc_auc']:.4f} | "
            f"Average Precision: "
            f"{row['validation_average_precision']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df.sort_values(
        "validation_average_precision",
        ascending=False,
    )

    comparison_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    best_model_name = str(comparison_df.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]

    print(f"\nSelected model: {best_model_name}")
    print(
        "Selection criterion: validation average precision"
    )

    validation_probabilities = (
        best_pipeline.predict_proba(X_val)[:, 1]
    )

    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    threshold_results = [
        evaluate_predictions(
            y_val,
            validation_probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    threshold_df = pd.DataFrame(threshold_results)

    threshold_df.to_csv(
        RESULTS_DIR / "threshold_comparison.csv",
        index=False,
    )

    best_threshold_index = threshold_df["f1_score"].idxmax()

    best_threshold_row = threshold_df.loc[
        best_threshold_index
    ]

    selected_threshold = float(
        best_threshold_row["threshold"]
    )

    validation_metrics = best_threshold_row.to_dict()

    save_json(
        RESULTS_DIR / "validation_metrics.json",
        validation_metrics,
    )

    print("\nSelected threshold:")
    print(f"{selected_threshold:.2f}")

    print("\nValidation metrics at selected threshold:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_probabilities = (
        best_pipeline.predict_proba(X_test)[:, 1]
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    save_json(
        RESULTS_DIR / "test_metrics.json",
        test_metrics,
    )

    joblib.dump(best_pipeline, MODEL_PATH)

    model_config = {
        "selected_model": best_model_name,
        "threshold": selected_threshold,
        "threshold_selection_metric": "validation_f1",
        "model_selection_metric": "validation_average_precision",
        "target_column": "Default",
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
    }

    save_json(CONFIG_PATH, model_config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved configuration: {CONFIG_PATH}")
    print(f"Saved reports: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
