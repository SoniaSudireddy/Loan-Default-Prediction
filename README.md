# Loan Default Prediction

An end-to-end machine-learning project that estimates the likelihood of loan default from applicant and loan attributes. The project covers data inspection, preprocessing, feature engineering, model comparison, validation-based threshold selection, final test-set evaluation, saving the trained model, and exposing predictions through a FastAPI REST API.

This README is written to explain the project to someone who has not seen the code before: what problem it addresses, what data it uses, how the pipeline works, how results should be interpreted, and how to run and test the application.

> **Responsible-use notice:** This is an educational project and a demonstration of a machine-learning workflow. It is not a production lending system and must not be used as the sole basis for approving or rejecting real loan applicants. Real-world use requires independent validation, fairness assessment, privacy safeguards, legal/regulatory review, human oversight, and monitoring.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Why This Project?](#why-this-project)
5. [How the Project Works](#how-the-project-works)
6. [Technology Stack](#technology-stack)
7. [Dataset Description](#dataset-description)
8. [Data Inspection and Preprocessing](#data-inspection-and-preprocessing)
9. [Feature Engineering](#feature-engineering)
10. [Models Implemented](#models-implemented)
11. [Model Evaluation and Results](#model-evaluation-and-results)
12. [Threshold Selection and Decision Logic](#threshold-selection-and-decision-logic)
13. [Saving and Using the Model](#saving-and-using-the-model)
14. [FastAPI Prediction API](#fastapi-prediction-api)
15. [How to Run the Project in VS Code](#how-to-run-the-project-in-vs-code)
16. [Automated Testing](#automated-testing)
17. [Expected Outputs and Repository Structure](#expected-outputs-and-repository-structure)
18. [Important Assumptions and Limitations](#important-assumptions-and-limitations)
19. [Future Improvements](#future-improvements)
20. [Learning Outcomes](#learning-outcomes)
21. [Author](#author)

---

## 1. Project Overview

Loan Default Prediction is a supervised machine-learning project. It uses historical loan records to learn patterns associated with loan default and then estimates the likelihood that a new record belongs to the default class.

The target column is `Default`:

- `0` — the record is labelled as no default.
- `1` — the record is labelled as default.

The trained model produces a score for an applicant record. A decision threshold converts that score into a binary prediction. The model does not know with certainty whether a borrower will repay a loan; it estimates patterns learned from the data.

The project has two main parts:

1. **Model-training workflow:** loads the dataset, validates and prepares the data, creates features, compares models, selects a threshold, evaluates the chosen model, and saves the resulting pipeline and reports.
2. **Prediction API:** loads the saved model and exposes a prediction endpoint through FastAPI. Pydantic validates incoming request fields before inference.

## 2. Problem Statement

Loan providers need to assess the risk that a borrower may fail to repay a loan according to its terms. Manual review can be time-consuming, and decisions based on limited information can be inconsistent. Machine learning can help analyse historical records and identify patterns associated with past defaults.

This project formulates loan default prediction as a **binary classification problem**. Applicant and loan attributes are used as input features, and `Default` is the target label. Several algorithms are trained and compared to identify a candidate model that performs well according to a defined validation metric.

The output is a risk estimate, not a guarantee or a complete credit assessment. The model's usefulness depends on the quality and representativeness of the dataset, the correctness of the feature definitions, and the metric used to select the model.

## 3. Objectives

The main objectives are to:

- Load and inspect a loan dataset using Python.
- Validate the expected columns and target values.
- Separate predictive features from the target column.
- Handle missing numerical and categorical values.
- Create additional loan-related features.
- Compare Logistic Regression, Decision Tree, Random Forest, and LightGBM.
- Use a validation set to select a model and a classification threshold.
- Evaluate the final model on a separate test set.
- Save the fitted pipeline and configuration so the model can be reused.
- Build a REST API for making predictions.
- Validate API input and test both valid and invalid requests.
- Document the workflow, results, assumptions, and limitations.

## 4. Why This Project?

This project demonstrates a practical machine-learning workflow using structured data. It is useful because real ML projects involve more than selecting an algorithm: data must be checked, transformations must be applied consistently, models must be compared fairly, and the final performance must be measured on data that was not used to tune the model.

The project provides practice in:
- Data analysis and validation
- Feature engineering and preprocessing pipelines
- Binary classification
- Class imbalance and evaluation metrics
- Model comparison and threshold selection
- Model serialization and reuse
- API development and request validation
- Automated testing and reproducible execution

## 5. How the Project Works

The overall workflow is:

1. **Load the CSV:** Read `Loan_default.csv` into a pandas DataFrame.
2. **Inspect and validate:** Check required columns and ensure the target values are valid.
3. **Prepare the target and features:** Separate `Default` from the input columns and exclude `LoanID` if it is present.
4. **Split the data:** Create training, validation, and test subsets using stratification.
5. **Engineer features:** Create derived features from existing loan attributes.
6. **Preprocess inputs:** Impute missing values, scale numerical values, and one-hot encode categorical values.
7. **Train candidate models:** Fit each candidate model on the training subset.
8. **Compare models:** Rank the candidates using validation average precision.
9. **Select a threshold:** Compare candidate thresholds using validation F1-score.
10. **Evaluate the selected model:** Apply the chosen model and threshold to the held-out test set.
11. **Save artifacts:** Store the fitted model, selected configuration, and evaluation reports.
12. **Serve predictions:** Load the saved model through FastAPI and accept validated input.
13. **Test the application:** Run automated tests for endpoints, valid predictions, and invalid input cases.

### Data split from the reported run

| Subset | Rows | Approximate share |
|---|---:|---:|
| Training | 178,742 | 70% |
| Validation | 38,302 | 15% |
| Test | 38,303 | 15% |
| **Total** | **255,347** | **100%** |

The training subset is used to fit the model. The validation subset is used to compare models and select the threshold. The test subset is reserved for final evaluation after those decisions are made.

## 6. Technology Stack

| Technology | How it is used |
|---|---|
| Python | Main programming language for the project |
| pandas | Loads, inspects, and manipulates the tabular dataset |
| NumPy | Supports numerical operations and threshold generation |
| scikit-learn | Provides data splitting, preprocessing, pipelines, classification models, and evaluation metrics |
| LightGBM | Provides a gradient-boosting model candidate |
| joblib | Saves the fitted model pipeline to disk |
| FastAPI | Exposes prediction functionality through HTTP endpoints |
| Pydantic | Validates the types and ranges of API input fields |
| Uvicorn | Runs the FastAPI application locally |
| pytest | Runs automated tests |
| VS Code | Local development and execution environment |

## 7. Dataset Description

The training script expects a CSV file named `Loan_default.csv` at the path configured in `train_and_save.py`. The target column is `Default`.

### Numerical features

- `Age`
- `Income`
- `LoanAmount`
- `CreditScore`
- `MonthsEmployed`
- `NumCreditLines`
- `InterestRate`
- `LoanTerm`
- `DTIRatio`

### Categorical features

- `Education`
- `EmploymentType`
- `MaritalStatus`
- `HasMortgage`
- `HasDependents`
- `LoanPurpose`
- `HasCoSigner`

`LoanID`, if present, is treated as an identifier rather than a predictive feature.

### Dataset summary from the reported training run

| Item | Value |
|---|---:|
| Rows before cleaning | 255,347 |
| Rows after cleaning | 255,347 |
| Columns before cleaning | 18 |
| Columns after cleaning | 17 |
| `Default = 0` | 225,694 |
| `Default = 1` | 29,653 |

The number of rows stayed the same in this run, while the column count decreased by one. Confirm from the training code which column was removed and document the reason.

The dataset's original source, licence, feature definitions, and permitted use must be verified and documented before publishing or redistributing it. The README does not assume a particular source or licence without verification.

## 8. Data Inspection and Preprocessing

Before training, the script checks that required columns exist and that the target contains valid binary values. The target is separated from the input features, and the identifier is excluded when present.

The pipeline applies separate preprocessing to numerical and categorical columns:

- **Numerical data:** missing values are imputed using the median, then values are standardised.
- **Categorical data:** missing values are imputed using the most frequent category, then categories are one-hot encoded.
- **Unknown categories:** the encoder is configured to ignore categories not observed during fitting. The API may still reject values that do not match its declared request schema.

The transformations are part of the fitted pipeline. This helps ensure that training and inference use the same transformations and prevents the preprocessing stage from learning imputation or scaling values from the validation or test subsets.

The target class distribution shows that the no-default class is more common than the default class. Therefore, accuracy alone is not enough to assess model quality; precision, recall, F1-score, average precision, and confusion-matrix counts are also important.

## 9. Feature Engineering

Feature engineering creates additional variables from the original inputs to represent potentially useful relationships.

The `CreditFeatureEngineer` transformer creates the following features:

| Derived feature | Purpose |
|---|---|
| `LoanToIncome` | Relates the loan amount to income |
| `EstimatedMonthlyPayment` | Estimates a payment amount using loan amount, interest rate, and term |
| `PaymentToIncomeRatio` | Relates estimated payment to monthly income |
| `CreditScoreTier` | Groups credit scores into categorical bands |

These transformations are included in the machine-learning pipeline so they are applied consistently when the model is trained and when it receives new records through the API.

**Implementation detail to verify:** the payment calculation is only as reliable as its formula and units. A standard amortising-loan payment calculation depends on the interest rate per payment period and the number of payment periods. Confirm the implementation in `features.py`, including the units of `Income`, `InterestRate`, and `LoanTerm`, before treating this derived feature as a financially exact payment.

## 10. Models Implemented

The training script compares four classification algorithms.

### Logistic Regression
A linear classification algorithm that estimates the relationship between input features and the target. It is a useful baseline and was selected in the reported run.

### Decision Tree
A tree-based algorithm that learns decision rules by splitting records according to feature values. It can model non-linear relationships but may overfit if not constrained.

### Random Forest
An ensemble of decision trees whose predictions are combined. It can model complex relationships and reduce some of the instability of a single tree.

### LightGBM
A gradient-boosting tree algorithm that builds trees sequentially to improve prediction performance. It is commonly used for structured/tabular data.

The winning model is selected by **validation average precision**, not by the model name or by test-set performance.

## 11. Model Evaluation and Results

Multiple metrics are reported because the default class is less common than the no-default class.

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted defaults that are actually labelled default.
- **Recall:** proportion of actual defaults identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures the model's ability to rank positive cases above negative cases across thresholds.
- **Average precision:** summarises precision-recall performance and is useful when classes are imbalanced.
- **Confusion matrix:** counts true negatives, false positives, false negatives, and true positives.

### Validation model comparison

The following values are from the reported training run:

| Model | Validation ROC-AUC | Validation average precision |
|---|---:|---:|
| Logistic Regression | 0.7590 | **0.3343** |
| Decision Tree | 0.6982 | 0.2669 |
| Random Forest | 0.7526 | 0.3245 |
| LightGBM | 0.7556 | 0.3298 |

**Selected model: Logistic Regression.** It achieved the highest validation average precision among the four models in this run.

### Final test-set evaluation

The selected model was evaluated on the held-out test set using threshold `0.62`.

| Metric | Test result |
|---|---:|
| Accuracy | 80.61% |
| Precision | 30.44% |
| Recall | 52.11% |
| F1-score | 0.3843 |
| ROC-AUC | 0.7659 |
| Average precision | 0.3392 |

### Confusion matrix counts

| Outcome | Count | Meaning |
|---|---:|---|
| True negatives (TN) | 28,557 | Correctly predicted no default |
| False positives (FP) | 5,298 | Predicted default, but actual class was no default |
| False negatives (FN) | 2,130 | Actual default cases not identified |
| True positives (TP) | 2,318 | Correctly predicted default |

### What the results mean

The model achieved 80.61% accuracy on the test set, but that value should not be interpreted alone. The test recall of 52.11% means the model identified a little over half of the actual default records in this test sample. Precision of 30.44% means about three in ten records predicted as default were labelled default in this sample.

The confusion matrix also shows that false positives and false negatives remain. These errors can have different consequences depending on the intended use. The reported metrics describe this dataset and split; they are not a guarantee of future performance.

## 12. Threshold Selection and Decision Logic

A model's predicted score is converted into a class by applying a decision threshold. The threshold determines how high the score must be before the record is classified as default.

For the reported run:

- **Selected threshold:** `0.62`
- Score greater than or equal to `0.62` → predicted default (`1`)
- Score below `0.62` → predicted no default (`0`)

The threshold was selected by maximising **F1-score on the validation set**. This balances precision and recall according to the F1 formula. It does not necessarily minimise financial loss or represent an optimal real-world lending policy.

The test set is used only after model and threshold selection. If the project later adopts a cost-based decision rule, the costs of false positives and false negatives should be explicitly justified, and the threshold should still be selected using validation data rather than the final test set.

## 13. Saving and Using the Model

After training, the script saves the fitted pipeline and its configuration.

| File | Purpose |
|---|---|
| `loan_default_model.pkl` | Fitted feature-engineering, preprocessing, and classifier pipeline |
| `model_config.json` | Selected model and threshold configuration |
| `results/model_comparison.csv` | Validation metrics for candidate models |
| `results/threshold_comparison.csv` | Metrics for candidate thresholds on validation data |
| `results/test_metrics.json` | Final metrics from the held-out test set |

The saved pipeline should be used for inference so that feature engineering and preprocessing match the training process. Only load model files from trusted sources because joblib/pickle files are not safe to load from untrusted origins.

## 14. FastAPI Prediction API

The prediction service is implemented in `app.py`. FastAPI exposes endpoints for checking the service and requesting predictions. The main prediction route is `POST /predict`.

Pydantic validates request fields before inference. The schema can enforce required fields, numeric types, allowed ranges, and accepted categories. This helps reject malformed requests before they reach the model.

The API should load the model saved by the training script and use the intended threshold configuration. The interactive documentation shows the exact request schema accepted by the current code.

### API endpoints

- `GET /` — returns basic service status.
- Health endpoint — reports the service/model status if configured in `app.py`.
- `POST /predict` — validates an applicant payload and returns a prediction response.

Endpoint details depend on the current implementation in `app.py`; use the live API documentation as the definitive schema.

## 15. How to Run the Project in VS Code

These instructions assume Windows PowerShell and a project directory containing `train_and_save.py`, `app.py`, `features.py`, and `requirements.txt`.

### Step 1: Open the project folder

Open the project folder in VS Code and select **Terminal → New Terminal**.

Example path from the reported setup:

```powershell
cd "C:\Users\SoniaSudireddy\Documents\Loan_Prediction_Checkpoint2\Loan Default Prediction"
```

If your project is stored elsewhere, use that actual path.

### Step 2: Create and activate a virtual environment

If your existing `venv` already works, activate it rather than creating a new one.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project does not yet have a complete `requirements.txt`, ensure it includes the packages actually imported by the project, including pandas, NumPy, scikit-learn, LightGBM, joblib, FastAPI, Pydantic, Uvicorn, and pytest.

### Step 4: Confirm the dataset path

Make sure `Loan_default.csv` exists where `train_and_save.py` expects it. If it is stored in a different directory, update the configured path in the script.

### Step 5: Train and evaluate the model

Run:

```powershell
python train_and_save.py
```

The script should print the dataset summary, model comparison, selected model, selected threshold, validation metrics, and final test metrics. It should save the model and report files. If the script reports an error, resolve that error before proceeding.

### Step 6: Start the API

After training completes and `loan_default_model.pkl` exists, run:

```powershell
python -m uvicorn app:app --reload
```

Keep this terminal open while the server is running.

Open the following URLs in your browser:

- API root: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs

### Step 7: Request a prediction

1. Open `http://127.0.0.1:8000/docs`.
2. Expand `POST /predict`.
3. Click **Try it out**.
4. Enter JSON containing the fields required by the schema displayed on the page.
5. Click **Execute**.
6. Review the response and HTTP status.
7. If there is an error, check the terminal running Uvicorn.

### Step 8: Stop the server

Return to the terminal running Uvicorn and press `Ctrl+C`.

## 16. Automated Testing

Activate the project environment, then run:

```powershell
python -m pytest -v
```

The reported test run passed **11 tests**, covering:
- Root endpoint
- Health endpoint
- Valid prediction
- Credit score above the permitted range
- Negative income
- Missing required field
- Invalid category
- Wrong numeric type
- Null income
- Extra field
- Non-finite income

A Starlette/httpx deprecation warning appeared in the reported run but did not cause a test failure. Dependency compatibility should be reviewed during maintenance, and tests should be rerun after changes to the code, model, or environment.

Passing API tests confirms the behaviours covered by those tests. It does not prove that the model is accurate, fair, calibrated, or ready for production.

## 17. Expected Outputs and Repository Structure

### Generated outputs

A successful training run should create or update:

- `loan_default_model.pkl`
- `model_config.json`
- `results/model_comparison.csv`
- `results/threshold_comparison.csv`
- `results/test_metrics.json`

A successful API startup should show a local Uvicorn URL. The `/docs` page should load while the server is running, and a valid request should return a prediction response.

### Suggested project structure

```text
Loan Default Prediction/
├── app.py
├── train_and_save.py
├── features.py
├── test_api.py
├── test_run.py
├── requirements.txt
├── Loan_default.csv
├── loan_default_model.pkl
├── model_config.json
├── README.md
└── results/
    ├── model_comparison.csv
    ├── threshold_comparison.csv
    └── test_metrics.json
```

This is a suggested structure. Check the actual project folder to confirm which files exist. Do not commit private applicant data, credentials, virtual environments, or datasets whose licence does not permit redistribution.

## 18. Important Assumptions and Limitations

- The training script expects the named columns and a binary `Default` target.
- The dataset contains more no-default examples than default examples, so accuracy alone can be misleading.
- The selected threshold maximises validation F1-score; it does not automatically minimise business costs.
- Precision in the reported test results is relatively low, and both false-positive and false-negative errors remain.
- Results may change when the dataset, random seed, code, or package versions change.
- The dataset's source, licence, units, and feature definitions must be verified.
- Feature engineering formulas should be checked for correct units and financial meaning.
- Fairness, calibration, and performance on future applicant populations have not been established by the reported metrics.
- The API's input validation checks request structure and ranges; it cannot guarantee that data is truthful or that the model's decision is appropriate.

Before any real-world use, conduct independent validation, privacy and security review, appropriate fairness testing, legal/regulatory review, and ongoing monitoring. Do not use this model alone to make consequential lending decisions.

## 19. Future Improvements

Possible next steps include:

1. Document and verify the dataset source, licence, and feature definitions.
2. Review data-quality rules and feature ranges.
3. Evaluate threshold choices against a documented cost matrix if reliable business costs are available.
4. Assess probability calibration.
5. Evaluate performance across appropriate demographic or operational groups where legally and ethically appropriate.
6. Add explainability for the selected model.
7. Record package versions and training metadata for reproducibility.
8. Add monitoring and drift detection before considering deployment.
9. Strengthen API security, logging, and deployment configuration if the project is extended.

## 20. Learning Outcomes

This project demonstrates:
- Data loading, inspection, and validation
- Numerical and categorical preprocessing
- Feature engineering within a pipeline
- Binary classification with four algorithms
- Stratified train/validation/test splitting
- Model comparison using validation metrics
- Threshold selection without using the test set for tuning
- Evaluation with precision, recall, F1-score, ROC-AUC, average precision, and confusion-matrix counts
- Saving and reusing a trained model
- Building and testing a FastAPI service
- Documenting assumptions, limitations, and reproducible steps

## 21. Author

**Sonia Sudireddy**

Project repository: [Loan Default Prediction](https://github.com/SoniaSudireddy/Loan-Default-Prediction)

Before publishing or submitting, confirm the dataset citation and licence, verify that the paths and filenames match the actual project, and regenerate the reported metrics if the training pipeline changes.
