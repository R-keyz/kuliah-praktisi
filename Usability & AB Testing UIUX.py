import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SIMULASI DATA EKSPERIMEN USABILITY PENGGUNA
np.random.seed(42)
n_users = 200

versions = np.random.choice(['UI_V1_Baseline', 'UI_V2_Redesign'], size=n_users, p=[0.5, 0.5])
time_sec, errors, sus = [], [], []

for v in versions:
    if v == 'UI_V1_Baseline':
        time_sec.append(np.random.normal(140, 25))
        errors.append(np.random.poisson(3.2))
        sus.append(np.random.normal(62, 12))
    else:
        time_sec.append(np.random.normal(95, 18))
        errors.append(np.random.poisson(1.1))
        sus.append(np.random.normal(82, 9))

df = pd.DataFrame({
    'interface_version': versions,
    'task_time_sec': np.clip(time_sec, 30, 300),
    'error_count': errors,
    'sus_score': np.clip(sus, 0, 100)
})

if __name__ == "__main__":
    print("=== TOPIK 3: EVALUASI KEBOLIHGUNAAN UI/UX (HCI DATA VISUALIZATION) ===")
    print(df.groupby('interface_version').mean().round(2))
    
    # Visualisasi
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sns.boxplot(ax=axes[0], data=df, x='interface_version', y='task_time_sec', palette='Set2')
    axes[0].set_title("Waktu Penyelesaian Tugas (Detik)")
    
    sns.barplot(ax=axes[1], data=df, x='interface_version', y='error_count', palette='Set2')
    axes[1].set_title("Rata-Rata Jumlah Error")
    
    sns.kdeplot(ax=axes[2], data=df, x='sus_score', hue='interface_version', fill=True, palette='Set2')
    axes[2].set_title("Distribusi Skor System Usability Scale (SUS)")
    
    plt.tight_layout()
    plt.savefig("hci_usability_dashboard.png")
    print("\nGrafik berhasil disimpan sebagai 'hci_usability_dashboard.png'")