# Overview
This project intended to create learning project that practises the full scikit-learn workflow (exploration, preprocessing, training, evaluation) by predicting loan default.

# .github
Folder for github workflow.

# dataset
Data set used in this project.

## raw
The original data (`credit.csv`). Never edit it; all cleaning happens in code (`src/data.py`).

# docs
This folder include documentations, Implementation Plan, fix-plan and any related documentation intended by user.

If the code function changes, the docs must up to date with the code. Same goes for vice versa.

## changelog.md
State in bullet point the changes that you have made (Maximum 2/3/4 sentences).

## project_brief.md
Contain goal, problem, tech stack etc template intended to narrow the project path.

## manual_command.md
PowerShell commands for setting up the environment, installing packages and running the code.

# notebooks
Jupyter notebooks for exploring the data. Number them in order, e.g. `01_explore.ipynb`.

# src
Python code for the pipeline, one file per step. Only `train.py` is run (`python src/train.py`); the other files hold functions that it imports.

- `config.py`: shared settings (data path, `random_state`, test size).
- `data.py`: load, clean and split the data.
- `features.py`: one-hot encoding and scaling inside a scikit-learn `Pipeline`.
- `models.py`: create the models: dummy classifier, logistic regression, decision tree and random forest.
- `train.py`: train and compare the models.
- `evaluate.py`: metrics and charts.

# results
Outputs: charts (e.g. confusion matrices) and the model comparison table.
