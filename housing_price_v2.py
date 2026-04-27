#production style with pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("train.csv")
X = df.drop(['SalePrice'], axis = 1)
y = np.log1p(df.SalePrice)

X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['HouseAge'] = X['YrSold'] - X['YearBuilt']
X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
X['QualXSF'] = X['OverallQual'] * X['GrLivArea']
X['GarageRatio'] = X['GarageArea'] / X['LotArea'].replace(0, np.nan)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

numerical_cols = X_train.select_dtypes(exclude='object').columns

quality_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
bsmt_fin_order = ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
bsmt_exp_order = ['No', 'Mn', 'Av', 'Gd']
garage_fin_order = ['Unf', 'RFn', 'Fin']
land_slope_order = ['Gtl', 'Mod', 'Sev']
functional_order = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
paved_order = ['N', 'P', 'Y']

ordinal_categories = (
    [quality_order] * 10 +
    [bsmt_fin_order] * 2 +
    [bsmt_exp_order] +
    [garage_fin_order] +
    [land_slope_order] +
    [functional_order] +
    [paved_order]
)

ordinal_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond', 'PoolQC',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
    'GarageFinish', 'LandSlope', 'Functional', 'PavedDrive'
]

categorical_cols = [col for col in X_train.select_dtypes(include='object').columns 
                    if col not in ordinal_cols]

numerical_transformer = SimpleImputer(strategy='median')
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=ordinal_categories,
                               handle_unknown='use_encoded_value',
                               unknown_value=-1))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('ord', ordinal_transformer, ordinal_cols),
    ('cat', categorical_transformer, categorical_cols)
])

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=2500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
    ))
])

my_pipeline.fit(X_train, y_train)
predictions = my_pipeline.predict(X_valid)
print("MAE:", mean_absolute_error(np.expm1(y_valid), np.expm1(predictions)))


#--------------#

test_data = pd.read_csv("test.csv")

test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
test_data['HouseAge'] = test_data['YrSold'] - test_data['YearBuilt']
test_data['TotalBath'] = test_data['FullBath'] + 0.5 * test_data['HalfBath']
test_data['QualXSF'] = test_data['OverallQual'] * test_data['GrLivArea']
test_data['GarageRatio'] = test_data['GarageArea'] / test_data['LotArea'].replace(0, np.nan)

test_predictions = np.expm1(my_pipeline.predict(test_data))

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
output.to_csv('submission_v2.csv', index=False)