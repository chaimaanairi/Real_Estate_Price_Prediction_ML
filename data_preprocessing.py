import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    # Veri kümesini yükle
    housing = pd.read_csv('data/housing.csv')

    # Eksik verileri kontrol et
    print("Eksik Veri Durumu:")
    print(housing.isnull().sum())

    # Min-max normalizasyonu yap
    scaler = MinMaxScaler()
    housing_scaled = housing.copy()
    housing_scaled['area'] = scaler.fit_transform(housing[['area']])

    print(housing_scaled.head())

if __name__ == "__main__":
    main()

