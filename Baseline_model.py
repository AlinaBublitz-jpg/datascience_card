import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

# Pfad zur Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Historische Daten laden und kombinieren
data_path = './historical_baseline_data.csv'
if os.path.exists(data_path):
    historical_data = pd.read_csv(data_path)
    data = pd.concat([historical_data, data], ignore_index=True)
    data.to_csv(data_path, index=False)
else:
    data.to_csv(data_path, index=False)

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# PSP-spezifische Erfolgsraten berechnen
psp_success_rate = data.groupby('PSP')['success'].mean().reset_index()
psp_success_rate.rename(columns={'success': 'psp_success_rate'}, inplace=True)
data = data.merge(psp_success_rate, on='PSP', how='left')

# Features und Zielvariable definieren
X = data[['psp_success_rate']]  # Baseline-Modell nutzt nur die Erfolgsrate
y = data['success']

# **Daten in Trainings- und Testdaten teilen**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Pipeline für Skalierung und Training des Modells erstellen
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Features skalieren
    ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic Regression Modell
])

# Modell mit Trainingsdaten trainieren
pipeline.fit(X_train, y_train)

# Vorhersagen auf Testdaten durchführen
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Gesamtmetriken berechnen
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Gesamtmetriken ausgeben
print("\nGesamtmetriken des Modells:")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Metriken pro PSP berechnen
print("\nMetriken pro PSP:")
for psp in data['PSP'].unique():
    psp_data = data[data['PSP'] == psp]  # Daten für den aktuellen PSP filtern
    if len(psp_data['success'].unique()) > 1:  # Sicherstellen, dass beide Klassen vorhanden sind
        psp_X = psp_data[['psp_success_rate']]
        psp_y = psp_data['success']

        # **Vorhersagen durchführen nur mit dem Modell auf Testdaten**
        psp_pred = pipeline.predict(psp_X)
        psp_pred_proba = pipeline.predict_proba(psp_X)[:, 1]

        # Metriken berechnen
        accuracy_psp = accuracy_score(psp_y, psp_pred)
        auc_psp = roc_auc_score(psp_y, psp_pred_proba)
        precision_psp = precision_score(psp_y, psp_pred, zero_division=0)
        recall_psp = recall_score(psp_y, psp_pred, zero_division=0)
        f1_psp = f1_score(psp_y, psp_pred)
        conf_matrix_psp = confusion_matrix(psp_y, psp_pred)

        print(f"\n{psp}:")
        print(f"  Accuracy: {accuracy_psp:.2f}")
        print(f"  AUC: {auc_psp:.2f}")
        print(f"  Precision: {precision_psp:.2f}")
        print(f"  Recall: {recall_psp:.2f}")
        print(f"  F1-Score: {f1_psp:.2f}")
        print(f"  Confusion Matrix:\n{conf_matrix_psp}")
    else:
        print(f"\n{psp}: Nicht genug Klassen, um Metriken zu berechnen (nur eine Klasse vorhanden).")

# Precision-Recall-Kurve berechnen und plotten
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Logistic Regression)')
plt.legend()
plt.grid()
plt.show()

