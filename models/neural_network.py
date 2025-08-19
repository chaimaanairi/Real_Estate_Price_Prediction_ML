import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def main():
    # Veri kümesini yükle
    housing = pd.read_csv('../data/housing.csv')

    # Özellik ve hedef değişkenleri ayır
    X = housing.drop('price', axis=1)
    y = housing['price']

    # Kategorik verileri dönüştür
    X = pd.get_dummies(X, drop_first=True)

    # Eksik değerleri doldur
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)

    # Eğitim ve test verisi ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Veriyi NumPy array'lerine dönüştür
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Neural Network modelini oluştur
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Modeli eğit
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Tahmin yap
    y_pred = model.predict(X_test).flatten()

    # Performans metriklerini yazdır
    print("Neural Network Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Eğitim kayıplarını görselleştir
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../results/learning_curve_nn.png')
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.title('Yapay Sinir Ağı: Gerçek ve Tahmin Edilen Değerler')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # X=Y line
    plt.savefig('../results/nn_true_vs_predicted.png')
    plt.show()


if __name__ == "__main__":
    main()

