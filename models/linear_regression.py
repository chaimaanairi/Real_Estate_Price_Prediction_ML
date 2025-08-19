import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def main():
    # Veri kümesini yükle
    housing = pd.read_csv('../data/housing.csv')

    # Özellik ve hedef değişkenleri ayır
    X = housing.drop('price', axis=1)
    y = housing['price']

    # Kategorik verileri dönüştür
    X = pd.get_dummies(X, drop_first=True)

    # Eğitim ve test verisi ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Performans metriklerini yazdır
    print("Linear Regression Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    plt.title('Doğrusal Regresyon: Gerçek ve Tahmin Edilen Değerler')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # X=Y line
    plt.savefig('../results/LR_true_vs_predicted.png')
    plt.show()


if __name__ == "__main__":
    main()


