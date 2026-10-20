import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# SINTESIS DATASET NASABAH BANK
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'annual_income': np.random.uniform(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'dti_ratio': np.random.uniform(0.1, 0.6, n),
    'loan_amount': np.random.uniform(5000, 500000, n)
})

# Penetapan Target 'loan_approved' (1/0)
logit = 0.00003 * df['annual_income'] + 0.008 * (df['credit_score'] - 600) - 4.0 * df['dti_ratio']
prob = 1 / (1 + np.exp(-logit))
df['loan_approved'] = (np.random.binomial(1, prob) == 1).astype(int)

# PIPELINE ML
X = df.drop(columns=['loan_approved'])
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

if __name__ == "__main__":
    print("=== TOPIK 2: PIPELINE MACHINE LEARNING LOAN APPROVAL ===")
    print(f"Akurasi Model ML: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("--- Laporan Klasifikasi ---")
    print(classification_report(y_test, y_pred, target_names=['Ditolak', 'Disetujui']))