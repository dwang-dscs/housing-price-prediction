# Housing Price Prediction

Predicting house sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Overview

End-to-end regression project covering data cleaning, feature engineering, model training, and submission generation.

## Files

| File | Description |
|------|-------------|
| `housing_price_notebook.ipynb` | Full walkthrough — EDA, preprocessing, modeling |
| `housing_price_v1.py` | Initial pipeline script |
| `housing_price_v2.py` | Improved pipeline with tuning |
| `housing_price_pipeline.pkl` | Saved trained model pipeline |
| `train.csv` | Training data |
| `test.csv` | Test data |
| `data_description.txt` | Feature descriptions |
| `submission_*.csv` | Kaggle submission files |

## How to Run

```bash
pip install pandas scikit-learn numpy
python housing_price_v2.py
```

Or open `housing_price_notebook.ipynb` in Jupyter.

## Results

See submission CSVs for predicted values on the test set.
