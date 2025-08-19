import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# CSV dosyasından model sonuçlarını oku
df = pd.read_csv('../results/model_results.csv')

# Model abbreviations (Dictionary for abbreviations)
model_abbreviations = {
    'Linear Regression': 'LR',
    'Decision Tree': 'DT',
    'Random Forest': 'RF',
    'Neural Network': 'NN'
}

# Replace model names with abbreviations
df['Models'] = df['Model'].map(model_abbreviations)

# Performans metriklerini görselleştirme
plt.figure(figsize=(12, 6))

# MAE (Mean Absolute Error) bar grafiği
plt.subplot(1, 3, 1)
sns.barplot(x='Models', y='MAE', data=df)
plt.title('MAE Comparison')

# RMSE (Root Mean Squared Error) bar grafiği
plt.subplot(1, 3, 2)
sns.barplot(x='Models', y='RMSE', data=df)
plt.title('RMSE Comparison')

# R2 (R-Kare) bar grafiği
plt.subplot(1, 3, 3)
sns.barplot(x='Models', y='R2', data=df)
plt.title('R2 Comparison')

# Ekstra: Model açıklamalarını ekle
plt.figtext(0.5, 0.01,
            'Model Abbreviations: LR = Linear Regression, DT = Decision Tree, RF = Random Forest, NN = Neural Network',
            wrap=True, horizontalalignment='center', fontsize=12)

# Grafikleri sıkıştır
plt.tight_layout()

# Sonucu kaydet
plt.savefig('../results/model_comparison.png')  # Save the plot to the 'results' directory

# Grafikleri göster
plt.show()
