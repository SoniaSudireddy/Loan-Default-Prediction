import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

# Required when loading a model containing this custom transformer.
from features import CreditFeatureEngineer  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_default_model.pkl"
CONFIG_PATH = BASE_DIR / "model_config.json"

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description=(
        "Predicts estimated default probability and risk category. "
        "It does not make a loan approval decision."
    ),
    version="1.1.0",
)

model = None
model_config = None

try:
    model = joblib.load(MODEL_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    threshold = float(model_config["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Saved threshold must be between 0 and 1.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must support predict_proba.")

except Exception as exc:
    model = None
    model_config = None
    print(f"Model loading failed: {exc}")


class LoanApplicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(..., ge=18, le=100, strict=True)
    Income: float = Field(
        ..., gt=0, le=10_000_000, allow_inf_nan=False
    )
    LoanAmount: float = Field(
        ..., gt=0, le=5_000_000, allow_inf_nan=False
    )
    CreditScore: int = Field(..., ge=300, le=850, strict=True)
    MonthsEmployed: int = Field(..., ge=0, le=720, strict=True)
    NumCreditLines: int = Field(..., ge=0, le=50, strict=True)
    InterestRate: float = Field(
        ..., ge=0.1, le=50, allow_inf_nan=False
    )
    LoanTerm: Literal[12, 24, 36, 48, 60]
    DTIRatio: float = Field(
        ..., ge=0, le=1.5, allow_inf_nan=False
    )

    Education: Literal[
        "High School", "Bachelor's", "Master's", "PhD"
    ]
    EmploymentType: Literal[
        "Full-time", "Part-time", "Self-employed", "Unemployed"
    ]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    HasMortgage: Literal["Yes", "No"]
    HasDependents: Literal["Yes", "No"]
    LoanPurpose: Literal[
        "Auto", "Business", "Education", "Home", "Other"
    ]
    HasCoSigner: Literal["Yes", "No"]


@app.get("/")
def root():
    return {
        "service": "Loan Default Risk Prediction API",
        "status": "ready" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    if model is None or model_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or configuration is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "selected_model": model_config.get(
            "selected_model", "unknown"
        ),
    }


@app.post("/predict")
def predict_default(applicant: LoanApplicant):
    if model is None or model_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is unavailable.",
        )

    input_df = pd.DataFrame([applicant.model_dump()])

    try:
        probabilities = model.predict_proba(input_df)

        classes = list(getattr(model, "classes_", []))

        if not classes and hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")
            classes = list(
                getattr(classifier, "classes_", [])
            )

        if probabilities.ndim != 2:
            raise ValueError("Unexpected probability output.")

        if 1 not in classes:
            raise ValueError(
                "Model does not contain the Default=1 class."
            )

        default_index = classes.index(1)

        probability = float(
            probabilities[0, default_index]
        )

        selected_threshold = float(
            model_config["threshold"]
        )

        risk_level = (
            "HIGH_DEFAULT_RISK"
            if probability >= selected_threshold
            else "LOW_DEFAULT_RISK"
        )

        return {
            "status": "success",
            "default_probability": round(probability, 4),
            "threshold_used": selected_threshold,
            "risk_level": risk_level,
            "selected_model": model_config.get(
                "selected_model", "unknown"
            ),
            "note": (
                "Risk classification only; "
                "not a loan approval decision."
            ),
        }

    except Exception as exc:
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs.",
        ) from exc
