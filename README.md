# ML Project: Regression and Rule Discovery

Two machine learning tasks on the same dataset (10,000 samples, 273 numeric features).

## Part 1: Regression (`target01`)
`src/Part1_solution.py`

- Feature selection: top 10 features by CatBoost importance (computed inside each fold)
- Degree-3 polynomial expansion of those 10 features (285 terms)
- CatBoostRegressor (800 trees, learning rate 0.05, depth 10)
- 5-fold cross-validation, with feature selection and polynomial fitting done inside each fold

Result: CV R² ≈ 0.95 (RMSE ≈ 0.05).

## Part 2: Rule discovery (`target02`)
`src/rulesextraction.py` and `src/framework_50.py`

The target is reconstructed as a set of if/else rules:
- `feat_259` acts as a switcher with thresholds 0.2, 0.5, 0.7
- In each of the four segments, `target02` is a linear combination of `feat_60`, `feat_86`, `feat_88`

`framework_50.py` implements the rules with only standard-library-style operations (no ML inference), as required for the edge-device constraint.

## Running
```
pip install -r requirements.txt
cd src
python Part1_solution.py          # expects dataset_50.csv, target_50.csv, EVAL_50.csv in the working directory
python rulesextraction.py
python framework_50.py --eval_file_path EVAL_50.csv
```

The datasets are not included in this repository.
