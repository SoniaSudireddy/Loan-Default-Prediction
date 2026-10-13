
"""Run an end-to-end API smoke test."""

from fastapi.testclient import TestClient
from app import app

PAYLOAD = {
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


def main():
    with TestClient(app) as client:
        health = client.get("/health")
        print("Health response:", health.status_code, health.json())
        health.raise_for_status()

        response = client.post("/predict", json=PAYLOAD)
        print("Prediction response:", response.status_code)
        print(response.json())
        response.raise_for_status()


if __name__ == "__main__":
    main()
