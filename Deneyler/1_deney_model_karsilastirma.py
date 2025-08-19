import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Veri Kümesini Yükle
housing = pd.read_csv('../data/Housing.csv')

# Kategorik Verileri Kodla (Label Encoding)
le = LabelEncoder()

# Identify categorical columns and encode them
categorical_cols = housing.select_dtypes(include=['object']).columns
for col in categorical_cols:
    housing[col] = le.fit_transform(housing[col])

# Özellikler ve hedef değişkeni ayır
X = housing.drop('price', axis=1)
y = housing['price']

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Lineer Regresyon
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Model 2: Karar Ağaçları
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Model 3: Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Model 4: Yapay Sinir Ağı
model_nn = Sequential()
model_nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))
model_nn.compile(optimizer='adam', loss='mean_squared_error')
model_nn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
y_pred_nn = model_nn.predict(X_test)


# Performans Değerlendirmesi ve Sonuçları Kaydet
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Performance for {model_name}:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 50)

    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


# Evaluate all models
results = []
results.append(evaluate_model(y_test, y_pred_lr, 'Linear Regression'))
results.append(evaluate_model(y_test, y_pred_dt, 'Decision Tree'))
results.append(evaluate_model(y_test, y_pred_rf, 'Random Forest'))
results.append(evaluate_model(y_test, y_pred_nn, 'Neural Network'))

# Sonuçları bir DataFrame'e kaydet
df_results = pd.DataFrame(results)

# CSV'ye kaydet
df_results.to_csv('../results/model_results.csv', index=False)

# Performans sonuçlarını yazdır
print("\nAll Model Performance Results:")
print(df_results)

print("model_results.csv kaydedildi.")

