# Housing Price Prediction

Kaggle competition — predicting sale prices for residential homes in Ames, Iowa using 79 features.

Started with a quick script (`housing_price_v1.py`) using `pd.get_dummies` and a basic XGBoost with no log transform — got ~$16k MAE on validation. This repo documents the full iteration from that baseline to a proper sklearn Pipeline that brought it down to ~$13k MAE.

## What changed between v1 and v2

The main issues with v1:

- Ordinal columns like `KitchenQual` (Po < Fa < TA < Gd < Ex) were being treated as unordered categories, so the model had no idea "Ex" was better than "Po"
- The imputer was fit before the train/val split — small leakage on `LotFrontage` etc.
- No log transform on the target, so the model was being pulled around by expensive outliers

v2 fixes all of that: proper `OrdinalEncoder` with the actual orderings from the data docs, a full sklearn `Pipeline` so nothing leaks, and `log1p` on the target.

## Feature engineering

Beyond the raw features, I added:

- `TotalSF` — total square footage across basement + floors
- `HouseAge` — years between build and sale
- `TotalBath` — full baths + 0.5 × half baths
- `QualXSF` — `OverallQual × GrLivArea` (r=0.84 with log price — a 2000sqft house rated 9/10 is very different from one rated 4/10)
- `GarageRatio` — garage area relative to lot size

Also tried `QualXAge` but it only correlated at -0.41 and wasn't adding anything the model didn't already have from `QualXSF`. Dropped it.

## Results

| | MAE |
|---|---|
| Baseline (mean predictor) | ~$57k |
| v1 (get_dummies, no log transform) | ~$16k |
| v2 / notebook (Pipeline + OrdinalEncoder + log target) | ~$13k |

## Files

| File | Description |
|------|-------------|
| `housing_price_notebook.ipynb` | Full walkthrough — EDA, feature engineering, v1 vs v2 comparison, evaluation |
| `housing_price_v2.py` | Production-style pipeline script |
| `housing_price_v1.py` | Original script (kept for comparison) |
| `housing_price_pipeline.pkl` | Saved trained pipeline |
| `train.csv` / `test.csv` | Kaggle competition data |
| `data_description.txt` | Feature documentation |
| `submission_*.csv` | Kaggle submission files |

## How to run

```bash
pip install pandas numpy scikit-learn xgboost joblib
python housing_price_v2.py
```

Or open `housing_price_notebook.ipynb` in Jupyter for the full walkthrough.
