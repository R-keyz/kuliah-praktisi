import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. BUSINESS & DATA UNDERSTANDING
# ==========================================
np.random.seed(42)
n_invoices = 1000

# Simulasi Data Mentah (Real-world Dirty Data)
data = {
    'invoice_id': [f"INV-{i:04d}" for i in range(1, n_invoices + 1)],
    'client_credit_score': np.random.choice([np.nan, 300, 450, 600, 750, 800], size=n_invoices, p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10]),
    'invoice_amount_usd': np.random.exponential(scale=5000, size=n_invoices) + 500,
    'historical_late_count': np.random.poisson(lam=1.8, size=n_invoices),
    'payment_term_days': np.random.choice([15, 30, 45, 60], size=n_invoices),
    'actual_days_to_pay': []
}

# Membuat target keterlambatan pembayaran berdasarkan faktor risiko
for i in range(n_invoices):
    credit = data['client_credit_score'][i]
    amt = data['invoice_amount_usd'][i]
    hist = data['historical_late_count'][i]
    term = data['payment_term_days'][i]
    
    # Logika keterlambatan
    base_delay = (hist * 5) + (amt / 2000)
    if pd.isna(credit) or credit < 550:
        base_delay += np.random.uniform(10, 30)
    
    actual_pay = term + base_delay + np.random.normal(0, 5)
    data['actual_days_to_pay'].append(max(1, int(actual_pay)))

df_raw = pd.DataFrame(data)

# ==========================================
# 2. DATA PREPARATION (Cleaning & Feature Engineering)
# ==========================================
df = df_raw.copy()

# Handing Missing Values (Imputasi Median)
credit_median = df['client_credit_score'].median()
df['client_credit_score'] = df['client_credit_score'].fillna(credit_median)

# Feature Engineering
df['delay_days'] = df['actual_days_to_pay'] - df['payment_term_days']
df['is_late'] = (df['delay_days'] > 7).astype(int)  # Terlambat jika > 7 hari dari jatuh tempo

# ==========================================
# 3. IMPLEMENTASI 4 TINGKATAN ANALISIS DATA
# ==========================================

print("=== 1. DESCRIPTIVE ANALYTICS (Apa yang terjadi?) ===")
total_invoices = len(df)
late_invoices = df['is_late'].sum()
late_ratio = (late_invoices / total_invoices) * 100
avg_delay = df[df['is_late'] == 1]['delay_days'].mean()

print(f"Total Faktur Diproses        : {total_invoices}")
print(f"Jumlah Faktur Terlambat     : {late_invoices} ({late_ratio:.1f}%)")
print(f"Rata-rata Keterlambatan     : {avg_delay:.1f} hari di atas jatuh tempo\n")


print("=== 2. DIAGNOSTIC ANALYTICS (Mengapa itu terjadi?) ===")
correlation = df[['client_credit_score', 'invoice_amount_usd', 'historical_late_count', 'is_late']].corr()['is_late']
print("Korelasi Faktor terhadap Keterlambatan Pembayaran:")
print(correlation.drop('is_late').round(3).to_string())
print("Diagnosis: Skor kredit rendah dan riwayat keterlambatan tinggi menjadi pendorong utama late payment.\n")


print("=== 3. PREDICTIVE ANALYTICS (Apa yang akan terjadi?) ===")
X = df[['client_credit_score', 'invoice_amount_usd', 'historical_late_count', 'payment_term_days']]
y = df['is_late']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Akurasi Model Prediksi Late Payment: {acc * 100:.2f}%")
print("\nLaporan Klasifikasi Model Prediksi:")
print(classification_report(y_test, y_pred, target_names=['Tepat Waktu', 'Terlambat']))


print("=== 4. PRESCRIPTIVE ANALYTICS (Apa yang harus kita lakukan?) ===")
# Simulasi Keputusan Bisnis Otomatis pada Faktur Baru
new_invoice = pd.DataFrame([{
    'client_credit_score': 480,
    'invoice_amount_usd': 12500.0,
    'historical_late_count': 4,
    'payment_term_days': 30
}])

risk_prob = model.predict_proba(new_invoice)[0][1] * 100
prediction = model.predict(new_invoice)[0]

print(f"Analisis Faktur Baru ({new_invoice['invoice_amount_usd'].iloc[0]} USD):")
print(f"Probabilitas Risiko Terlambat: {risk_prob:.1f}%")

# Logika Preskriptif (Human-in-the-loop Rules)
if risk_prob > 70:
    action = "TINDAKAN KETAT: Minta Pembayaran Uang Muka (DP 50%) & Tawarkan Diskon Early Payment 3%."
elif risk_prob > 40:
    action = "TINDAKAN MODERAT: Kirimkan Pengingat Otomatis 5 Hari Sebelum Jatuh Tempo."
else:
    action = "TINDAKAN STANDAR: Proses Faktur Sesuai Syarat Pembayaran Normal."

print(f"Rekomendasi Keputusan Bisnis : {action}")