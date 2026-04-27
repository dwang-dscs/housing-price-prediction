#Basic model for learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train.csv")
X = df.drop(['SalePrice'], axis = 1)
y = df.SalePrice

X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['HouseAge'] = X['YrSold'] - X['YearBuilt']
X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

missing_cols = X_train.columns[X_train.isnull().any()].tolist()
for_imputation = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'Electrical']
for_encoding = list(set(missing_cols) - set(for_imputation))

for col in for_encoding:
    X_train[col] = X_train[col].apply(lambda x: 0 if pd.isna(x) else 1)
    X_valid[col] = X_valid[col].apply(lambda x: 0 if pd.isna(x) else 1)

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train), columns=X_train.columns)   
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid), columns=X_valid.columns)    

my_model = XGBRegressor(n_estimators = 500, early_stopping_rounds=50)
my_model.fit(imputed_X_train, y_train,
             eval_set=[(imputed_X_valid, y_valid)])
predictions = my_model.predict(imputed_X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))


#--------------#

test_data = pd.read_csv("test.csv")

test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
test_data['HouseAge'] = test_data['YrSold'] - test_data['YearBuilt']
test_data['TotalBath'] = test_data['FullBath'] + 0.5 * test_data['HalfBath']

for col in for_encoding:
    test_data[col] = test_data[col].apply(lambda x: 0 if pd.isna(x) else 1)

test_data = pd.get_dummies(test_data)
test_data = X_train.align(test_data, join='left', axis=1, fill_value=0)[1]  # align to train columns

imputed_test = pd.DataFrame(my_imputer.transform(test_data), columns=test_data.columns)
test_predictions = my_model.predict(imputed_test)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
output.to_csv('submission_v1.csv', index=False)