# Loan Default Prediction

A learning project that practises the full scikit-learn workflow (exploration, preprocessing, training, evaluation) by predicting whether a loan applicant is likely to default.

## Status

In progress. Data loading and the train/test split work, and a dummy classifier runs as the baseline. Cleaning, preprocessing, the other models and evaluation are not built yet, so there are no results to report.

## Dataset

[`dataset/raw/credit.csv`](dataset/raw/credit.csv) holds German credit data.

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

## Running

Run the pipeline from the project folder:

```bash
python src/train.py
```

To explore the data, open [`notebooks/01_explore.ipynb`](notebooks/01_explore.ipynb) in VS Code or Jupyter and select the `.venv` environment.

## Project structure

```text
.
├── dataset/
│   └── raw/credit.csv   # Original data, never edited
├── docs/                # Project brief and other documentation
├── notebooks/           # Jupyter notebooks for exploring the data
├── results/             # Charts and the model comparison table
├── src/
│   ├── config.py        # Shared settings (paths, random_state, test size)
│   ├── data.py          # Load, clean and split the data
│   ├── features.py      # One-hot encoding and scaling (Pipeline)
│   ├── models.py        # Create the models
│   ├── train.py         # Main script: train and compare the models
│   └── evaluate.py      # Metrics and charts
├── AGENTS.md            # Project instructions for coding agents
├── README.md
└── requirements.txt
```

## Documentation

- [Project brief](docs/project_brief.md): problem, data, model approach and evaluation plan
