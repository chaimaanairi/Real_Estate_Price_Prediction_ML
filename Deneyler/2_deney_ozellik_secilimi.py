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

for col in categorical_columns:
    housing[col] = le.fit_transform(housing[col])

# Korelasyon Analizi
correlation_matrix = housing.corr()
important_features = correlation_matrix['price'].sort_values(ascending=False)

# Yüksek korelasyona sahip özellikleri seç (0.5 üzeri)
selected_features = important_features[abs(important_features) > 0.5].index.tolist()
selected_features.remove('price')  # price hedef değişken olduğu için çıkarıyoruz

# Özellikleri seçerek yeni veri kümesi oluştur
housing_selected = housing[selected_features + ['price']]

# Eğitim ve test verisini ayır
X = housing_selected.drop('price', axis=1)
y = housing_selected['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Performans Değerlendirmesi
print("Feature Selection Model Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
