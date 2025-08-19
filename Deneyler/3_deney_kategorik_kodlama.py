import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Veri Kümesini Yükle
housing = pd.read_csv('../data/housing.csv')

# Kategorik Kolonları Bul ve Kodla (Label Encoding)
le = LabelEncoder()

# Kategorik kolonları otomatik olarak buluyoruz
categorical_columns = housing.select_dtypes(include=['object']).columns

# Tüm kategorik kolonları etiket kodlaması yapıyoruz
for col in categorical_columns:
    housing[col] = le.fit_transform(housing[col])

# One-Hot Encoding
housing_one_hot_encoded = pd.get_dummies(housing, drop_first=True)

# Model 1: Label Encoding ile Model
X_le = housing.drop('price', axis=1)
y_le = housing['price']
X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(X_le, y_le, test_size=0.3, random_state=42)
model_le = LinearRegression()
model_le.fit(X_train_le, y_train_le)
y_pred_le = model_le.predict(X_test_le)

# Model 2: One-Hot Encoding ile Model
X_ohe = housing_one_hot_encoded.drop('price', axis=1)
y_ohe = housing_one_hot_encoded['price']
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y_ohe, test_size=0.3, random_state=42)
model_ohe = LinearRegression()
model_ohe.fit(X_train_ohe, y_train_ohe)
y_pred_ohe = model_ohe.predict(X_test_ohe)

# Performans Değerlendirmesi
print("Label Encoding Model Results:")
print("MAE:", mean_absolute_error(y_test_le, y_pred_le))
print("RMSE:", mean_squared_error(y_test_le, y_pred_le))
print("R2:", r2_score(y_test_le, y_pred_le))

print("\nOne-Hot Encoding Model Results:")
print("MAE:", mean_absolute_error(y_test_ohe, y_pred_ohe))
print("RMSE:", mean_squared_error(y_test_ohe, y_pred_ohe))
print("R2:", r2_score(y_test_ohe, y_pred_ohe))
