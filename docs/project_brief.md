1. Problem and goal

Problem: a lender needs to know which loan applicants are likely to default.
Goal: a learning project that practises the full scikit-learn workflow (exploration, preprocessing, training, evaluation) by predicting loan default.

2. Problem type

Predict a number → regression (linear regression, etc.)
Predict a category → classification (decision tree, etc.)
Find groups → clustering
Images, text, audio → usually neural networks

This project: **binary classification**. The target is `default` (yes / no).

3. Data

Import from dataset: `dataset/credit.csv` or `https://raw.githubusercontent.com/tuanwisamz/Scikit-learn-/main/dataset/credit.csv`

- Size: 1,000 rows, 17 columns (16 attributes and the target).
- Target: `default`, 700 "no" and 300 "yes" (imbalanced).
- Numeric columns: `months_loan_duration`, `amount`, `percent_of_income`, `years_at_residence`, `age`, `existing_loans_count`, `dependents`.
- Categorical columns: `Attribute1`, `credit_history`, `purpose`, `savings_balance`, `employment_duration`, `other_credit`, `housing`, `job`, `phone`.
- Blank values: none. Some columns use "unknown" as a value; keep it as its own category.
- Cleaning: rename `Attribute1` to `checking_balance` (assumed from its values), and merge `car0` (12 rows) into `car` (assumed to be a typo).
- Excluded feature: `age`, because it can be a sensitive attribute in lending. This leaves 15 input features. Other columns may still correlate with age.

4. Model approach

Compare several models and pick the best:

1. Dummy classifier (baseline)
2. Logistic regression
3. Decision tree
4. Random forest

- Preprocessing: one-hot encode categorical columns and scale numeric columns for logistic regression, all inside a scikit-learn `Pipeline` to avoid data leakage.
- Imbalance: try `class_weight="balanced"` and adjust the decision threshold.
- Tuning: small grid search with cross-validation.

5. Tech stack

Language: Python.
Environment: Jupyter/VS Code locally.
Libraries: pandas and NumPy for data, scikit-learn for machine learning, Matplotlib for charts.

6. Compute and constraints

- Compute: local laptop, CPU only.
- Cost: free, open-source libraries only; no cloud services.
- Deadline: none.
- Reproducibility: set a fixed `random_state`.

7. Delivery and architecture

Delivery: one Jupyter notebook.

Flow: load CSV → explore → clean → split → preprocess and train (pipeline) → compare models → evaluate on the test set → conclusions.

8. Evaluation plan

Which metrics will be used (RMSE, accuracy, F1)?

- Primary: recall on `default = yes`, to catch as many defaulters as possible.
- Also report: precision, F1, ROC-AUC and the confusion matrix. Accuracy alone is misleading, since always predicting "no" scores 70%.
- Split: stratified 80/20 train/test, with stratified 5-fold cross-validation on the training set.
- Selection: choose the model and decision threshold using cross-validation only; use the test set once, at the end.
- Success targets: recall ≥ 0.70, precision ≥ 0.45 and ROC-AUC ≥ 0.75. These are targets to test, not guaranteed results.
