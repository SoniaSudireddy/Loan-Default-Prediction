
"""API input validation and prediction tests."""

from fastapi.testclient import TestClient

from app import app, model

client = TestClient(app)

VALID_PAYLOAD = {
    "Age": 32,
    "Income": 60000.0,
    "LoanAmount": 15000.0,
    "CreditScore": 720,
    "MonthsEmployed": 36,
    "NumCreditLines": 4,
    "InterestRate": 10.5,
    "LoanTerm": 36,
    "DTIRatio": 0.35,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "No",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "No",
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    if model is None:
        assert response.status_code == 503
    else:
        assert response.status_code == 200


def test_valid_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    if model is None:
        assert response.status_code == 503
        return

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["default_probability"] <= 1
    assert 0 <= result["threshold_used"] <= 1
    assert result["risk_level"] in {
        "LOW_DEFAULT_RISK",
        "HIGH_DEFAULT_RISK",
    }


def test_credit_score_above_range():
    payload = {**VALID_PAYLOAD, "CreditScore": 950}
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_income():
    payload = {**VALID_PAYLOAD, "Income": -5000.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field():
    payload = dict(VALID_PAYLOAD)
    payload.pop("DTIRatio")

    assert client.post("/predict", json=payload).status_code == 422


def test_invalid_category():
    payload = {**VALID_PAYLOAD, "Education": "Doctorate"}
    assert client.post("/predict", json=payload).status_code == 422


def test_wrong_numeric_type():
    payload = {**VALID_PAYLOAD, "Age": "thirty-two"}
    assert client.post("/predict", json=payload).status_code == 422


def test_null_income():
    payload = {**VALID_PAYLOAD, "Income": None}
    assert client.post("/predict", json=payload).status_code == 422


def test_extra_field():
    payload = {**VALID_PAYLOAD, "UnexpectedFeature": 123}
    assert client.post("/predict", json=payload).status_code == 422


def test_non_finite_income():
    payload = {**VALID_PAYLOAD, "Income": "NaN"}
    assert client.post("/predict", json=payload).status_code == 422
