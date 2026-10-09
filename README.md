# Loan Default Prediction — End-to-End Machine Learning Project

> **Project type:** Binary classification / supervised machine learning  
> **Primary notebook:** `loan-default-prediction-end-to-end-ml.ipynb`  
> **Dataset:** `Loan_default.csv` (loaded from a Kaggle dataset)  
> **Main model used for the saved pipeline:** LightGBM (`LGBMClassifier`)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Why This Project?](#why-this-project)
5. [How the Project Works](#how-the-project-works)
6. [Technology Stack](#technology-stack)
7. [Dataset Description](#dataset-description)
8. [Project Workflow](#project-workflow)
9. [Data Inspection and Preprocessing](#data-inspection-and-preprocessing)
10. [Feature Engineering](#feature-engineering)
11. [Models Implemented](#models-implemented)
12. [Model Evaluation](#model-evaluation)
13. [Business Cost and Decision Threshold](#business-cost-and-decision-threshold)
14. [Feature Importance](#feature-importance)
15. [Saving and Using the Model](#saving-and-using-the-model)
16. [How to Run the Project](#how-to-run-the-project)
17. [Running Locally in VS Code](#running-locally-in-vs-code)
18. [Expected Outputs](#expected-outputs)
19. [Repository Structure](#repository-structure)
20. [Important Implementation Notes](#important-implementation-notes)
21. [Advantages](#advantages)
22. [Limitations and Future Improvements](#limitations-and-future-improvements)
23. [Responsible Use](#responsible-use)
24. [Learning Outcomes](#learning-outcomes)
25. [Author](#author)

---

## Project Overview

The **Loan Default Prediction** project uses historical loan-application data to estimate whether a borrower may default on a loan. It treats the task as a **binary classification problem**:

- `0` — the record is labelled as **no default**
- `1` — the record is labelled as **default**

The notebook explores the data, prepares numerical and categorical features, trains machine-learning models, evaluates their predictions, engineers additional features, compares several algorithms, explores a cost-based decision threshold, and saves a fitted preprocessing-and-model pipeline as `loan_default_model.pkl`.

The notebook is an experimental machine-learning workflow. It is **not yet a deployed production application or a complete lending API**.

## Problem Statement

Loan providers need to assess the risk that a borrower will not repay a loan as agreed. Reviewing applications manually can be time-consuming, and risk decisions based on limited information can be inconsistent.

This project investigates whether borrower and loan attributes can be used by a supervised-learning model to estimate default risk. The output is a probability estimate and a threshold-based classification. Such predictions can support further review, but they should not be treated as an automatic or definitive lending decision.

## Objectives

- Load and inspect the loan dataset.
- Check column types, missing values, and the distribution of the target variable.
- Separate input features (`X`) from the target (`y`).
- Preprocess numerical and categorical data with scikit-learn pipelines.
- Split data into training and testing sets while preserving the target distribution.
- Train a LightGBM classifier.
- Evaluate predictions using a classification report, ROC-AUC, and a confusion matrix.
- Create additional loan-related features.
- Compare Logistic Regression, Random Forest, and LightGBM.
- Explore a decision threshold based on assumed false-negative and false-positive costs.
- Inspect LightGBM feature importance.
- Save a fitted pipeline and demonstrate prediction for a sample application.

## Why This Project?

Loan default prediction is a practical example of classification in financial risk analysis. It demonstrates how raw tabular data can be transformed into model-ready features and how evaluation should consider more than accuracy alone.

The project is useful for learning data preparation, feature engineering, classification, model comparison, evaluation metrics, class-imbalance handling, model persistence, and the difference between a probability score and a business decision.

## How the Project Works

At a high level, the workflow is:

1. **Load data:** Read `Loan_default.csv` into a pandas DataFrame.
2. **Inspect data:** Review data types, missing values, and the target distribution.
3. **Prepare inputs:** Remove `LoanID` if it exists; separate the `Default` target from predictors.
4. **Preprocess features:** Impute and scale numerical columns; impute and one-hot encode categorical columns.
5. **Split data:** Keep 80% for training and 20% for testing, using stratification and `random_state=42`.
6. **Train baseline model:** Fit a LightGBM classifier with class weighting based on the training target distribution.
7. **Evaluate:** Generate predicted probabilities and classifications, then calculate classification metrics and plot a confusion matrix.
8. **Engineer features:** Create loan-to-income, estimated monthly payment, payment-to-income ratio, and credit-score tier.
9. **Compare models:** Train Logistic Regression, Random Forest, and LightGBM on the engineered feature set.
10. **Explore threshold costs:** Compare assumed financial costs across thresholds from 0.10 to 0.90.
11. **Save a pipeline:** Fit the engineered-feature LightGBM pipeline and save it to `loan_default_model.pkl`.
12. **Demonstrate prediction:** Pass a sample applicant dictionary to `predict_loan_application()`.

---

## Technology Stack

| Technology | How it is used |
|---|---|
| Python | Main programming language |
| Kaggle Notebooks | Notebook environment used by the current workflow |
| pandas | Loading and manipulating tabular data |
| NumPy | Numerical operations and threshold generation |
| Matplotlib | Plotting evaluation and feature-importance charts |
| Seaborn | Confusion-matrix heatmap and feature-importance visualisation |
| scikit-learn | Data splitting, preprocessing, pipelines, metrics, and Logistic Regression / Random Forest |
| LightGBM | Gradient-boosted tree classification |
| joblib | Saving the fitted model pipeline to disk |
| Jupyter Notebook | Interactive, cell-by-cell execution |

### Installation requirements

The notebook imports these core libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm joblib jupyter
```

If running inside Kaggle, several libraries may already be installed. If a package is missing, install it in the notebook environment and restart the kernel if required.

## Dataset Description

The notebook reads a file named `Loan_default.csv` using this Kaggle input path:

```python
file_path = '/kaggle/input/datasets/sonia27chowdary/loan-default-data/Loan_default.csv'
df = pd.read_csv(file_path)
```

The notebook expects the dataset to contain the following columns.

### Target column

| Column | Meaning in this project |
|---|---|
| `Default` | Target label: `0` for no default and `1` for default |

### Numerical input features

| Column | Description |
|---|---|
| `Age` | Borrower's age |
| `Income` | Reported income |
| `LoanAmount` | Amount requested or recorded for the loan |
| `CreditScore` | Credit score |
| `MonthsEmployed` | Number of months employed |
| `NumCreditLines` | Number of credit lines |
| `InterestRate` | Loan interest rate |
| `LoanTerm` | Loan term |
| `DTIRatio` | Debt-to-income ratio |

### Categorical input features

| Column | Description |
|---|---|
| `Education` | Education category |
| `EmploymentType` | Employment category |
| `MaritalStatus` | Marital-status category |
| `HasMortgage` | Whether a mortgage is recorded |
| `HasDependents` | Whether dependents are recorded |
| `LoanPurpose` | Purpose of the loan |
| `HasCoSigner` | Whether a co-signer is recorded |

### Identifier

| Column | How it is handled |
|---|---|
| `LoanID` | Dropped if present because it is an identifier, not intended as a predictive feature |

**Important:** These descriptions explain how the notebook uses the column names; confirm their exact definitions, units, and coding against the original dataset documentation before using the model for real decisions. The notebook prints the actual dataset shape and column information when executed; this README does not claim a row count or model score because those values are not embedded as fixed results in the notebook source.

## Project Workflow

### 1. Load the data

`pandas.read_csv()` loads the CSV file into `df`. The notebook prints the dataset shape and displays the first rows so that the data can be checked.

### 2. Inspect data quality

The notebook uses:

- `df.info()` to inspect columns, data types, and non-null counts.
- `df.isnull().sum()` to count missing values in each column.
- `df['Default'].value_counts(normalize=True)` to inspect the proportion of each target class.

These checks are initial diagnostics. They do not replace a full audit of duplicates, invalid ranges, inconsistent categories, leakage, or unusual values.

### 3. Separate features and target

The target is `Default`. All other retained columns are inputs:

```python
target_col = 'Default'
X = df.drop(columns=[target_col])
y = df[target_col]
```

If `LoanID` is present, it is dropped first.

### 4. Preprocess numerical columns

The numerical pipeline applies:

- **Median imputation:** Replaces missing numerical values with the median of the relevant training feature.
- **Standard scaling:** Transforms numerical values to a common scale by centring and scaling them.

The preprocessing steps are inside a scikit-learn `Pipeline` and `ColumnTransformer`, so they are fitted on the training data rather than manually applied to the complete dataset before splitting.

### 5. Preprocess categorical columns

The categorical pipeline applies:

- **Most-frequent imputation:** Fills missing categorical values with the most common training value.
- **One-hot encoding:** Converts categories into numerical indicator columns.
- `handle_unknown='ignore'`: Allows the encoder to handle categories not seen during fitting without raising an error.

### 6. Split the data

The notebook uses an 80/20 train-test split:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- `test_size=0.2` reserves 20% for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` helps preserve the target-class proportions in both sets.

### 7. Handle class imbalance

The baseline LightGBM model calculates:

```python
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

This ratio is supplied as `scale_pos_weight`. It gives additional training weight to the positive/default class when the negative/no-default class is more common. The actual class distribution should still be checked; weighting is not a substitute for evaluating precision, recall, and calibration.

### 8. Train the baseline LightGBM model

The baseline model is wrapped with its preprocessing steps in a single pipeline. Its configured parameters include:

- `n_estimators=150`
- `learning_rate=0.05`
- `scale_pos_weight=pos_weight`
- `random_state=42`
- `n_jobs=-1`

The pipeline is fitted on `X_train` and `y_train`.

### 9. Evaluate the baseline model

The notebook calculates default-class probabilities with `predict_proba()` and initially classifies a record as default when its probability is at least `0.50`.

It then prints a classification report and ROC-AUC and displays a confusion matrix. See [Model Evaluation](#model-evaluation) for how to interpret these outputs.

### 10. Engineer features and compare models

The notebook creates four additional features and compares three algorithms using the engineered dataset. See [Feature Engineering](#feature-engineering) and [Models Implemented](#models-implemented).

---

## Feature Engineering

Feature engineering adds derived variables that may represent financial relationships more directly than the original columns.

| Engineered feature | Formula in the notebook | Intended interpretation |
|---|---|---|
| `LoanToIncome` | `LoanAmount / (Income + 1)` | Loan amount relative to reported income |
| `EstimatedMonthlyPayment` | `LoanAmount / (LoanTerm + 1)` | A simplified loan-amount-per-term estimate |
| `PaymentToIncomeRatio` | `EstimatedMonthlyPayment / ((Income / 12) + 1)` | Estimated payment relative to monthly income |
| `CreditScoreTier` | Bins `CreditScore` into named categories | A categorical representation of credit-score bands |

The credit-score bins are defined as `[300, 580, 670, 740, 800, 850]`, with labels `Poor`, `Fair`, `Good`, `Very_Good`, and `Exceptional`.

### Important interpretation notes

- The `+1` terms avoid division by zero but change the exact meaning of each ratio.
- `EstimatedMonthlyPayment` is a simplified approximation; it does **not** calculate a real amortised payment using interest rate and loan term.
- Credit scores outside the defined bin boundaries can become missing in `CreditScoreTier`; the categorical imputer may fill those values during model preprocessing. Check score ranges and bin coverage before relying on this feature.
- The feature engineering is performed before the engineered data is split. The current formulas use columns from each row and do not learn global statistics, but a stricter production workflow should build and validate all transformations inside a reproducible training pipeline.

## Models Implemented

The notebook benchmarks three classifiers using the engineered feature set.

| Model | Role | Key configuration |
|---|---|---|
| Logistic Regression | Linear classification baseline | `max_iter=1000`, `class_weight='balanced'`, `random_state=42` |
| Random Forest | Ensemble of decision trees | `n_estimators=100`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` |
| LightGBM | Gradient-boosted decision trees | `n_estimators=150`, `learning_rate=0.05`, `scale_pos_weight=pos_weight`, `random_state=42`, `n_jobs=-1` |

Each model is placed after the same engineered-feature preprocessor. The notebook records accuracy, default-class recall, default-class precision, F1-score, and ROC-AUC in a model-comparison DataFrame.

The notebook subsequently saves a **LightGBM** pipeline. It does not automatically save whichever model happens to rank first in the comparison table; the code explicitly selects `models_dict['LightGBM']`.

## Model Evaluation

The project uses several complementary metrics.

| Metric | What it tells you |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision for default | Of applications predicted as default, the fraction actually labelled default |
| Recall for default | Of actual defaults, the fraction the model identifies |
| F1-score | Harmonic mean of precision and recall for the selected positive class |
| ROC-AUC | How well probability scores rank positive examples above negative examples across thresholds |
| Confusion matrix | Counts true negatives, false positives, false negatives, and true positives |

### Confusion matrix

The notebook plots a confusion matrix with `No Default (0)` and `Default (1)` labels. In a binary confusion matrix:

- **True Negative (TN):** Correctly predicts no default.
- **False Positive (FP):** Predicts default for a record labelled no default.
- **False Negative (FN):** Predicts no default for a record labelled default.
- **True Positive (TP):** Correctly predicts default.

In lending-risk analysis, false negatives can be costly because risky applications may be missed. False positives can also be costly because potentially creditworthy applicants may be flagged. The appropriate trade-off depends on verified business costs, policy, fairness requirements, and human review.

### Why accuracy alone is not enough

If defaults are less common than non-defaults, a model can achieve high accuracy by mostly predicting no default. Recall, precision, F1-score, ROC-AUC, the confusion matrix, and probability calibration provide additional information. The actual printed results should be reviewed after running the notebook; no performance score is guaranteed by this README.

## Business Cost and Decision Threshold

The notebook assigns illustrative costs:

```python
COST_FN = 10000  # cost assigned to a false negative
COST_FP = 1000   # cost assigned to a false positive
```

It calculates:

```text
Total cost = (number of false negatives × COST_FN)
           + (number of false positives × COST_FP)
```

It checks 17 thresholds between `0.10` and `0.90` and selects the threshold with the lowest calculated cost. Lowering the threshold generally flags more applications as default; raising it generally flags fewer. The exact impact must be measured from the test predictions.

**These cost values are assumptions for demonstration, not validated real-world financial estimates.** They must be justified with relevant business data before they are used operationally.

The notebook currently reuses `y_pred_proba` from the earlier baseline model when evaluating thresholds against `y_test_fe`, the engineered-feature test target. Those probabilities were produced by a model trained on a different feature pipeline/split representation. For a valid threshold comparison, generate probabilities from a model trained on `X_train_fe` and evaluate them against the matching `X_test_fe` / `y_test_fe`. See [Important Implementation Notes](#important-implementation-notes).

## Feature Importance

The notebook extracts `feature_importances_` from the baseline LightGBM classifier and plots the ten highest-ranked features.

Feature importance can help identify which input columns the fitted tree model used most for its splits. It does **not** prove that a feature causes default, show the direction of its effect, or establish that the feature is fair or appropriate for lending. One-hot encoding also expands categorical features into multiple indicator columns.

For a stronger explanation, consider permutation importance, SHAP analysis, stability checks, and fairness analysis on held-out data.

## Saving and Using the Model

The notebook constructs an engineered-feature pipeline using the engineered preprocessor and LightGBM classifier, fits it, and saves it:

```python
import joblib

joblib.dump(best_pipeline, 'loan_default_model.pkl')
```

The prediction helper loads that file and expects a dictionary containing **all input fields required by the engineered pipeline**. That includes the original input features and the engineered fields:

- `LoanToIncome`
- `EstimatedMonthlyPayment`
- `PaymentToIncomeRatio`
- `CreditScoreTier`

The helper returns a dictionary with:

- `status`
- `default_probability`
- `risk_decision`
- `threshold_used`

The decision labels in the notebook are `REJECT / HIGH RISK` when the predicted probability meets or exceeds the threshold, and `APPROVE / LOW RISK` otherwise. These are demo labels in the notebook, not a recommendation to make real lending decisions automatically.

### Sample prediction

The notebook includes one sample applicant and calls `predict_loan_application(sample_applicant)`. The probability and decision depend on the trained model, dataset, library versions, and selected threshold. Run the notebook to obtain the current output.

**Important:** The helper currently requires callers to calculate engineered fields themselves. A safer design is to save a pipeline that receives only the original application fields and calculates derived features within the pipeline, reducing the risk of inconsistent or incorrect input calculations.

## How to Run the Project

### Option A — Run in Kaggle (closest to the notebook as written)

1. Open Kaggle and navigate to your notebook.
2. Attach the dataset that contains `Loan_default.csv`.
3. Confirm the dataset input path matches the `file_path` in the first cell. Kaggle input paths can differ between notebooks and datasets.
4. Run the notebook cells **from top to bottom**.
5. Review the printed dataset information, missing-value counts, target distribution, classification report, ROC-AUC, confusion matrix, feature-importance plot, model comparison, and threshold output.
6. The final cell saves `loan_default_model.pkl` in the notebook's current working directory.
7. Download the `.pkl` file from the notebook output/files area if you need to use it elsewhere.

### Option B — Run in VS Code / locally

1. Install Python (use a version supported by the installed libraries).
2. Create a project folder and place the notebook and CSV dataset in it.
3. Open the folder in VS Code.
4. Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt activation instead:

```bat
.venv\Scripts\activate.bat
```

5. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm joblib jupyter ipykernel
```

6. Update the dataset path in the notebook. For a CSV stored beside the notebook, use:

```python
file_path = 'Loan_default.csv'
df = pd.read_csv(file_path)
```

7. Open the `.ipynb` file in VS Code and select the `.venv` Python kernel.
8. Run each cell from top to bottom.
9. Check that the notebook completes and creates `loan_default_model.pkl`.

**Note:** The current notebook uses Kaggle's `/kaggle/input/...` path, so it will not run locally unchanged. The dataset file must be downloaded or otherwise made available to your local environment, and the path must be updated.

## Expected Outputs

When run successfully, the notebook should produce:

- Dataset shape and first rows.
- Data types and non-null counts.
- Missing-value counts by column.
- Target-class proportions.
- Training progress messages.
- A classification report at threshold `0.50`.
- A ROC-AUC score.
- A confusion-matrix heatmap.
- A chart of the top ten baseline LightGBM feature importances.
- A comparison table for Logistic Regression, Random Forest, and LightGBM.
- A selected threshold under the notebook's assumed cost values.
- A saved file named `loan_default_model.pkl`.
- A sample prediction response.

Actual metrics, plots, threshold, and prediction values depend on the dataset and execution environment.

## Repository Structure

A recommended repository layout is:

```text
loan-default-prediction/
├── README.md
├── loan-default-prediction-end-to-end-ml.ipynb
├── Loan_default.csv              # Optional; avoid committing data without checking its licence/privacy
├── loan_default_model.pkl        # Generated model artifact; usually do not commit large binaries
├── requirements.txt
└── .gitignore
```

The notebook currently contains the workflow in one file; the folders above are a suggested organisation, not a claim that all of these files already exist in the repository.

### Suggested `.gitignore`

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
*.py[cod]
.env
loan_default_model.pkl
```

Add the dataset filename to `.gitignore` if its licence or size means it should not be committed. Do not commit confidential applicant data, credentials, or private records.

### Suggested `requirements.txt`

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
joblib
jupyter
ipykernel
```

For reproducible results, record the tested Python and package versions after confirming the notebook runs successfully.

## Important Implementation Notes

The following points should be fixed or verified before describing the notebook as production-ready:

1. **Threshold evaluation uses mismatched predictions and labels.** The notebook's `y_pred_proba` comes from the baseline model's `X_test`, but the cost calculation compares those probabilities with `y_test_fe`. Recompute probabilities using the engineered-feature model and its matching engineered test split before selecting a threshold.
2. **The saved model's input contract is fragile.** `best_pipeline` uses the engineered preprocessor, which expects the engineered columns. The sample helper calculates none of them automatically; it relies on the caller to supply them. Move feature creation into a reusable transformer/pipeline or validate and calculate these fields in the prediction function.
3. **The threshold is selected on the test set.** Choosing a threshold that minimises cost on the same set used for final evaluation can make reported results optimistic. Use a validation set or cross-validation to select the threshold, then evaluate once on a separate untouched test set.
4. **The comparison table does not explicitly select the winner.** It prints metrics for three models, but the saved pipeline explicitly uses LightGBM. Decide the model-selection criterion (for example, default recall subject to an acceptable precision or cost) and document the evidence.
5. **Feature importance is from the baseline model.** The feature-importance plot is generated from the initial LightGBM model, not necessarily the final engineered-feature pipeline.
6. **Credit-score bin boundaries need validation.** Verify that all legitimate credit scores fall within the configured bins and decide how out-of-range values should be handled.
7. **The payment feature is simplified.** `LoanAmount / (LoanTerm + 1)` is not a standard amortisation formula. If the intended feature is a monthly payment estimate, define the term units and incorporate the interest rate appropriately.
8. **Data quality checks can be expanded.** Add checks for duplicate records, target values outside `{0, 1}`, invalid ranges, impossible values, inconsistent categories, and data leakage.
9. **Use a separate final test set.** Avoid repeatedly tuning features, models, and thresholds based on the same test data.
10. **Evaluate probability quality and fairness.** Review calibration, performance by relevant groups, data representativeness, and potential disparate impact before any real-world use.

These are code-review observations based on the notebook's current implementation, not evidence that the model has failed. They are practical next steps to make the analysis more reliable.

## Advantages

- Demonstrates a complete tabular classification workflow in one notebook.
- Uses scikit-learn pipelines to keep preprocessing organised.
- Handles missing values through imputation.
- Supports numerical and categorical features.
- Accounts for possible class imbalance in model training.
- Evaluates models with several metrics instead of accuracy alone.
- Compares linear, bagging-based, and gradient-boosting approaches.
- Adds domain-inspired features.
- Includes feature-importance visualisation.
- Saves the fitted pipeline with joblib for reuse.

## Limitations and Future Improvements

- The notebook currently depends on a Kaggle-specific data path.
- The exact dataset provenance, licence, and feature definitions should be documented.
- Threshold selection currently uses mismatched baseline probabilities and engineered test labels and should be corrected.
- Threshold tuning should use validation data rather than the final test set.
- The prediction helper requires manually supplied engineered features.
- The estimated monthly payment formula is simplified.
- The notebook does not include a web interface, REST API, database, monitoring, or deployment configuration.
- No fixed performance results are documented here; metrics should be recorded from a reproducible run.
- More complete validation, model calibration, explainability, fairness testing, and drift monitoring are needed for real-world use.

Possible future enhancements:

1. Correct threshold evaluation and add validation-based threshold selection.
2. Move feature engineering into the fitted pipeline.
3. Add cross-validation and hyperparameter tuning.
4. Add calibration curves and probability calibration.
5. Add unit tests for preprocessing and prediction inputs.
6. Add explainability with permutation importance or SHAP.
7. Assess fairness and model performance across appropriate groups.
8. Build a FastAPI or Flask prediction API with input validation.
9. Add experiment tracking and model-version metadata.
10. Deploy only after privacy, security, compliance, fairness, and business validation.

## Responsible Use

This project is educational and experimental. A prediction is a statistical estimate learned from historical data, not certainty about an individual borrower. Historical labels may reflect biases or outdated lending practices. Do not use this notebook alone to approve or reject real applicants. Any real deployment requires legal and regulatory review, privacy safeguards, fairness assessment, clear explanations, human oversight, monitoring, and validated business assumptions.

## Learning Outcomes

By completing and reviewing this project, you can practise:

- Loading and inspecting a real-world-style CSV dataset.
- Understanding a binary classification target.
- Separating features from labels.
- Handling missing numerical and categorical values.
- Using `Pipeline` and `ColumnTransformer`.
- Splitting data with stratification.
- Training LightGBM, Logistic Regression, and Random Forest.
- Evaluating classification using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Creating domain-inspired features.
- Interpreting tree-based feature importance.
- Exploring cost-sensitive classification thresholds.
- Serialising a model with joblib and using it for a sample prediction.
- Reviewing code for evaluation consistency and deployment readiness.

## Author

**Sonia Sudireddy**

This project was developed as a machine-learning learning project using a Kaggle notebook and a loan-default dataset. Update this section with your repository link, dataset citation, and any verified experiment results before publishing.
