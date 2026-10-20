import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# SIMULASI FITUR INDEKS SPEKTRAL SATELIT (NDVI, NDBI, EVI) & GEOSPASIAL
np.random.seed(42)
n_samples = 1000
drivers = ['Plantation_OilPalm', 'Smallholder_Agri', 'Logging_Timber', 'Wildfire']

data, target = [], []
for _ in range(n_samples):
    d = np.random.choice(drivers, p=[0.40, 0.30, 0.15, 0.15])
    if d == 'Plantation_OilPalm':
        ndvi, ndbi, evi, dist_road = np.random.normal(0.65, 0.05), np.random.normal(-0.20, 0.05), np.random.normal(0.50, 0.04), np.random.normal(500, 150)
    elif d == 'Smallholder_Agri':
        ndvi, ndbi, evi, dist_road = np.random.normal(0.45, 0.08), np.random.normal(-0.05, 0.06), np.random.normal(0.35, 0.05), np.random.normal(1200, 300)
    elif d == 'Logging_Timber':
        ndvi, ndbi, evi, dist_road = np.random.normal(0.55, 0.07), np.random.normal(-0.15, 0.05), np.random.normal(0.42, 0.06), np.random.normal(3500, 800)
    else:  # Wildfire
        ndvi, ndbi, evi, dist_road = np.random.normal(0.20, 0.08), np.random.normal(0.15, 0.07), np.random.normal(0.15, 0.05), np.random.normal(5000, 1500)
    
    data.append([ndvi, ndbi, evi, dist_road])
    target.append(d)

df = pd.DataFrame(data, columns=['NDVI', 'NDBI', 'EVI', 'Dist_Road_m'])
X = df
y = pd.Series(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

if __name__ == "__main__":
    print("=== TOPIK 4: GEOAI KLASIFIKASI ATRIBUSI PEMICU DEFORESTASI ===")
    print(f"Akurasi GeoAI Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))