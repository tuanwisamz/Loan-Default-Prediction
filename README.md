# Loan Default Prediction

A learning project that practises the full scikit-learn workflow (exploration, preprocessing, training, evaluation) by predicting whether a loan applicant is likely to default.

## Status

Planning is complete. The notebook and models have not been built yet, so there are no results to report.

## Dataset

[`dataset/credit.csv`](dataset/credit.csv) holds German credit data.

- **Size:** 1,000 rows and 17 columns (16 attributes and the target).
- **Target:** `default`, with 700 "no" and 300 "yes" (imbalanced).
- **Cleaning:** rename `Attribute1` to `checking_balance` (assumed from its values) and merge `car0` into `car` (assumed to be a typo).
- **Excluded feature:** `age`, because it can be a sensitive attribute in lending.

## Approach

This is a binary classification problem. Four models will be compared:

1. Dummy classifier (baseline)
2. Logistic regression
3. Decision tree
4. Random forest

Preprocessing runs inside a scikit-learn `Pipeline` to avoid data leakage. The workflow is:

load CSV → explore → clean → split → preprocess and train → compare models → evaluate on the test set → conclusions

## Evaluation

- **Primary metric:** recall on `default = yes`.
- **Also reported:** precision, F1, ROC-AUC and the confusion matrix.
- **Validation:** stratified 80/20 train/test split, with stratified 5-fold cross-validation on the training set.
- **Success targets:** recall ≥ 0.70, precision ≥ 0.45 and ROC-AUC ≥ 0.75. These are targets to test, not achieved results.

## Getting started

You need Python and either Jupyter or VS Code. Everything runs locally on a CPU.

```bash
git clone https://github.com/tuanwisamz/Scikit-learn-.git
cd Scikit-learn-
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the libraries:

```bash
pip install -r requirements.txt
```

## Project structure

```text
.
├── dataset/     # Data used in the project
├── docs/        # Project brief and other documentation
├── models/      # One folder per model
├── AGENTS.md    # Project instructions for coding agents
├── README.md
└── requirements.txt
```

## Documentation

- [Project brief](docs/project_brief.md): problem, data, model approach and evaluation plan
