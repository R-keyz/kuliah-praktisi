import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. SIMULASI EKTRAKSI FITUR SATELIT & GEOSPASIAL
# ==========================================
np.random.seed(42)
n_samples = 1200

drivers = ['Plantation_OilPalm', 'Smallholder_Agri', 'Logging_Timber', 'Wildfire']

data = []
target = []

for _ in range(n_samples):
    driver = np.random.choice(drivers, p=[0.40, 0.30, 0.15, 0.15])
    
    if driver == 'Plantation_OilPalm':
        ndvi = np.random.normal(0.65, 0.05)
        ndbi = np.random.normal(-0.20, 0.05)
        evi = np.random.normal(0.50, 0.04)
        dist_road = np.random.normal(500, 150)
        slope = np.random.normal(5, 2)
    elif driver == 'Smallholder_Agri':
        ndvi = np.random.normal(0.45, 0.08)
        ndbi = np.random.normal(-0.05, 0.06)
        evi = np.random.normal(0.35, 0.05)
        dist_road = np.random.normal(1200, 300)
        slope = np.random.normal(12, 4)
    elif driver == 'Logging_Timber':
        ndvi = np.random.normal(0.55, 0.07)
        ndbi = np.random.normal(-0.15, 0.05)
        evi = np.random.normal(0.42, 0.06)
        dist_road = np.random.normal(3500, 800)
        slope = np.random.normal(20, 6)
    else:  # Wildfire
        ndvi = np.random.normal(0.20, 0.08)
        ndbi = np.random.normal(0.15, 0.07)
        evi = np.random.normal(0.15, 0.05)
        dist_road = np.random.normal(5000, 1500)
        slope = np.random.normal(15, 5)
        
    data.append([ndvi, ndbi, evi, dist_road, slope])
    target.append(driver)

feature_names = ['NDVI', 'NDBI', 'EVI', 'Dist_to_Road_m', 'Slope_deg']
df = pd.DataFrame(data, columns=feature_names)
df['driver_label'] = target

# ==========================================
# 2. DATA PREPROCESSING & SPLITTING
# ==========================================
X = df[feature_names]
y = df['driver_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING & EVALUATION (GeoAI Baseline)
# ==========================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("=== TOPIK 4: GEOAI KLASIFIKASI ATRIBUSI PEMICU DEFORESTASI ===")
print(f"Total Sampel Kejadian Alert Deforestasi : {len(df)}")
print(f"Akurasi Model Baseline Random Forest    : {acc * 100:.2f}%\n")

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 4. ANALISIS FEATURE IMPORTANCE
# ==========================================
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("--- TINGKAT KONTRIBUSI FITUR (FEATURE IMPORTANCE) ---")
print(feature_imp_df.to_string(index=False))

# ==========================================
# 5. SIMULASI ATRIBUSI ALERT BARU (INFERENCE)
# ==========================================
new_alert = np.array([[0.62, -0.18, 0.48, 400, 4]])  # Profil area perkebunan
new_alert_scaled = scaler.transform(new_alert)
predicted_driver = rf_model.predict(new_alert_scaled)[0]
confidence = np.max(rf_model.predict_proba(new_alert_scaled)) * 100

print("\n--- DETEKSI & ATRIBUSI ALERT BARU ---")
print(f"Pemicu Deforestasi Terprediksi : {predicted_driver}")
print(f"Tingkat Kepercayaan (Confidence): {confidence:.2f}%")