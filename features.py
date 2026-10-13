import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CreditFeatureEngineer expects a pandas DataFrame."
            )

        required = [
            "Income",
            "LoanAmount",
            "InterestRate",
            "LoanTerm",
            "CreditScore",
        ]

        missing = [col for col in required if col not in X.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        result = X.copy()

        for col in required:
            result[col] = pd.to_numeric(
                result[col], errors="coerce"
            )

        income = result["Income"].to_numpy(dtype=float)
        amount = result["LoanAmount"].to_numpy(dtype=float)
        rate = result["InterestRate"].to_numpy(dtype=float) / 1200
        term = result["LoanTerm"].to_numpy(dtype=float)

        # Feature 1: Loan amount divided by annual income.
        loan_to_income = np.divide(
            amount,
            income,
            out=np.full(len(result), np.nan),
            where=(
                np.isfinite(income)
                & (income > 0)
                & np.isfinite(amount)
            ),
        )

        # Feature 2: Estimated monthly loan payment.
        monthly_payment = np.full(len(result), np.nan)

        valid = (
            np.isfinite(amount)
            & (amount >= 0)
            & np.isfinite(rate)
            & (rate >= 0)
            & np.isfinite(term)
            & (term > 0)
        )

        zero_rate = valid & (rate == 0)
        monthly_payment[zero_rate] = (
            amount[zero_rate] / term[zero_rate]
        )

        positive_rate = valid & (rate > 0)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            monthly_payment[positive_rate] = (
                amount[positive_rate]
                * rate[positive_rate]
                / (
                    1
                    - (
                        1 + rate[positive_rate]
                    ) ** (-term[positive_rate])
                )
            )

        # Feature 3: Estimated monthly payment / monthly income.
        monthly_income = income / 12

        payment_to_income = np.divide(
            monthly_payment,
            monthly_income,
            out=np.full(len(result), np.nan),
            where=(
                np.isfinite(monthly_income)
                & (monthly_income > 0)
            ),
        )

        result["LoanToIncome"] = loan_to_income
        result["EstimatedMonthlyPayment"] = monthly_payment
        result["PaymentToIncomeRatio"] = payment_to_income

        # Feature 4: Credit score category.
        tiers = pd.cut(
            result["CreditScore"],
            bins=[-np.inf, 579, 669, 739, 799, np.inf],
            labels=[
                "Poor",
                "Fair",
                "Good",
                "Very Good",
                "Exceptional",
            ],
        )

        result["CreditScoreTier"] = (
            tiers.astype("object")
            .where(tiers.notna(), "Unknown")
        )

        # Replace invalid infinite feature values with missing values.
        engineered = [
            "LoanToIncome",
            "EstimatedMonthlyPayment",
            "PaymentToIncomeRatio",
        ]

        result[engineered] = result[engineered].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result
