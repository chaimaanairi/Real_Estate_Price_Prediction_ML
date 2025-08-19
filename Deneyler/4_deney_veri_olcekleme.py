import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Veri Kümesini Yükle
housing = pd.read_csv('../data/housing.csv')

# Categorical encoding (if needed)
# Let's automatically detect categorical columns
categorical_columns = housing.select_dtypes(include=['object']).columns

# Apply LabelEncoder to these categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    housing[col] = label_encoder.fit_transform(housing[col])

# Özellik ve hedef değişkeni ayır
X = housing.drop('price', axis=1)
y = housing['price']

# Min-Max Normalizasyonu
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)

# Standardizasyon (Z-Score Normalizasyonu)
scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X)

# Eğitim ve test verisini ayır
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(X_minmax_scaled, y, test_size=0.3, random_state=42)
X_train_standard, X_test_standard, y_train_standard, y_test_standard = train_test_split(X_standard_scaled, y, test_size=0.3, random_state=42)

# Modeli oluştur
model = LinearRegression()

# Min-Max Normalizasyonu ile model
model.fit(X_train_minmax, y_train_minmax)
y_pred_minmax = model.predict(X_test_minmax)

# Standartlaştırma ile model
model.fit(X_train_standard, y_train_standard)
y_pred_standard = model.predict(X_test_standard)

# Performans Değerlendirmesi
print("Min-Max Normalization Model Results:")
print("MAE:", mean_absolute_error(y_test_minmax, y_pred_minmax))
print("RMSE:", mean_squared_error(y_test_minmax, y_pred_minmax))
print("R2:", r2_score(y_test_minmax, y_pred_minmax))

print("\nStandardization Model Results:")
print("MAE:", mean_absolute_error(y_test_standard, y_pred_standard))
print("RMSE:", mean_squared_error(y_test_standard, y_pred_standard))
print("R2:", r2_score(y_test_standard, y_pred_standard))
