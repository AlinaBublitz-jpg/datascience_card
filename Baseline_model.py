import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

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
# Daten in Trainings- und Testdaten teilen, dabei auch die Originalindices behalten
indices = data.index
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y)

# Reines Baseline-Modell: Vorhersage der Mehrheitsklasse aus den Trainingsdaten
majority_class = y_train.mode()[0]
baseline_probability = y_train.mean()  # Anteil von Success=1 in den Trainingsdaten

# Vorhersagen auf Testdaten (immer die Mehrheitsklasse)
y_pred = [majority_class] * len(y_test)
y_pred_proba = [baseline_probability] * len(y_test)  # konstante Wahrscheinlichkeit

# Gesamtmetriken berechnen
accuracy = accuracy_score(y_test, y_pred)
# Da alle Wahrscheinlichkeiten konstant sind, ist der AUC-Wert nicht sinnvoll berechenbar;
# im Baseline-Fall wird häufig ein AUC-Wert von 0.5 angenommen.
try:
    auc = roc_auc_score(y_test, y_pred_proba)
except Exception as e:
    auc = 0.5
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

# PSP-spezifische Metriken ausschließlich auf den Testdaten berechnen
print("\nMetriken pro PSP (auf Testdaten):")
for psp in data['PSP'].unique():
    # Filtere Testindices für den aktuellen PSP
    psp_indices = [i for i in idx_test if data.loc[i, 'PSP'] == psp]
    if not psp_indices:
        print(f"\n{psp}: Keine Testdaten vorhanden.")
        continue

    psp_y_test = data.loc[psp_indices, 'success']
    # Baseline-Vorhersagen für den aktuellen PSP (immer die Mehrheitsklasse)
    psp_pred = [majority_class] * len(psp_y_test)
    psp_pred_proba = [baseline_probability] * len(psp_y_test)

    # Überprüfen, ob beide Klassen in den Testdaten vorhanden sind
    if len(psp_y_test.unique()) > 1:
        accuracy_psp = accuracy_score(psp_y_test, psp_pred)
        try:
            auc_psp = roc_auc_score(psp_y_test, psp_pred_proba)
        except Exception as e:
            auc_psp = 0.5
        precision_psp = precision_score(psp_y_test, psp_pred, zero_division=0)
        recall_psp = recall_score(psp_y_test, psp_pred, zero_division=0)
        f1_psp = f1_score(psp_y_test, psp_pred)
        conf_matrix_psp = confusion_matrix(psp_y_test, psp_pred)


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
plt.title('Precision-Recall Curve (Baseline Model)')
plt.legend()
plt.grid()
plt.show()



# Confusion Matrix als Heatmap plotten
def plot_confusion_matrix_with_labels(y_true, y_pred, title='Confusion Matrix (Baseline Model)'):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

# Erstelle ein 2D-Array für die Heatmap
    cm_array = np.array([[tn, fp],
                         [fn, tp]])

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False,
                     xticklabels=['Predicted: 0', 'Predicted: 1'],
                     yticklabels=['Actual: 0', 'Actual: 1'])

    plt.title("Confusion Matrix - Heatmap")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Wahre Klasse")

    # Füge zusätzliche Labels in die Heatmap ein
    ax.text(0.3, 0.7, "TN", color="black", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.7, 0.7, "FP", color="black", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.3, 0.25, "FN", color="black", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.7, 0.25, "TP", color="black", ha="center", va="center", transform=ax.transAxes, fontsize=12)

    plt.show()






plt.figure(figsize=(5, 4))
sns.countplot(x=y_test)
plt.title("Verteilung der Zielvariable im Testset")
plt.xlabel("Success (0 = Fail, 1 = Success)")
plt.ylabel("Anzahl Transaktionen")
plt.show()



# PSP-spezifische Erfolgsraten im Testset
test_data = data.loc[idx_test].copy()

psp_success_test = test_data.groupby('PSP')['success'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=psp_success_test, x='PSP', y='success')
plt.title('PSP-spezifische Erfolgsraten (Testset)')
plt.xlabel('Payment Service Provider')
plt.ylabel('Durchschnittliche Erfolgsrate (0-1)')
plt.ylim(0, 1)
plt.show()