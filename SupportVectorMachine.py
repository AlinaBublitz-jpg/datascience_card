import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Laden der Daten aus einer Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Laden und Kombinieren historischer Daten mit Prüfung auf neue Datenpunkte
data_path = './historical_svm_data.csv'
success_prob_path = './psp_success_probabilities.csv'
new_data_threshold = 100  # Mindestanzahl neuer Datenpunkte für Neutrainierung

# Falls bereits historische Daten vorhanden sind, kombiniere sie mit den neuen Daten
if os.path.exists(data_path):
    historical_data = pd.read_csv(data_path)
    num_new_data = len(data)
    data = pd.concat([historical_data, data], ignore_index=True)
    data.to_csv(data_path, index=False)
else:
    data.to_csv(data_path, index=False)
    num_new_data = len(data)  # Falls keine historischen Daten existieren, alle als neu zählen




# Sicherstellen, dass die Zielvariable 'success' als Integer gespeichert wird (0 = Misserfolg, 1 = Erfolg)
data['success'] = data['success'].astype(int)

# Hinzufügen der PSP-spezifischen Gebührenstrukturen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Konvertiere Zeitstempel in Stundenwerte
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# One-Hot-Encoding für die Spalte 'country' (Länderinformationen)
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Definiere die Features für das Modell
final_features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour', 'country_Germany']

# Erfolgswahrscheinlichkeit aus vorherigem Lauf laden
update_model = num_new_data >= new_data_threshold  # Update falls genügend neue Datenpunkte

if os.path.exists(success_prob_path):
    prev_psp_success = pd.read_csv(success_prob_path).set_index('PSP')['Success_Probability'].to_dict()
else:
    prev_psp_success = {}

psp_success_probabilities = {}


# Initialisieren der Speicherstrukturen für Modelle und Erfolgswahrscheinlichkeiten
results = {}


# Training SVM-Modells für jeden PSP
for psp in data['PSP'].unique():
    print(f"\nModel for PSP: {psp}")

    # Filtere die Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']


    # Erfolgswahrscheinlichkeit des PSP berechnen
    if psp in prev_psp_success:
        previous_prob = prev_psp_success[psp]
    else:
        previous_prob = 0



    # Falls kein Modell-Update notwendig ist, trainiere mit allen Daten
    if not update_model:
        print("ℹ️ Modell bleibt unverändert, aber Performance wird berechnet.")
        X_train, X_test, y_train, y_test = X_psp, X_psp, y_psp, y_psp  # Trainiere mit allen Daten
    else:
        # Daten in Trainings- und Testdaten aufteilen
        X_train, X_test, y_train, y_test = train_test_split(
            X_psp, y_psp, test_size=0.3, random_state=42, stratify=y_psp
        )


    # Erstellen einer Pipeline: Standardisieren der Daten + Trainieren des SVM-Modells
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardisierung der Daten (wichtig für SVM!)
        ('svm', SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42))
    ])

    # Training des Modells
    pipeline.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Evaluierung des Modells
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Durchschnittliche Erfolgswahrscheinlichkeit berechnen
    success_probability = pipeline.predict_proba(X_psp)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    # Falls sich die Erfolgswahrscheinlichkeit signifikant ändert, erneutes Training
    if abs(success_probability - previous_prob) > 0.05:
        update_model = True
    else:
        print("ℹ️ Kein Modell-Update erforderlich, aber aktuelle Metriken werden ausgegeben.")

    # Speichern der Modellbewertung
    results[psp] = {
        'Model': pipeline,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
    }

    # Ausgabe der Metriken
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Success Probability: {success_probability:.4f}")



# Erfolgswahrscheinlichkeiten für den nächsten Lauf speichern
pd.DataFrame(psp_success_probabilities.items(), columns=['PSP', 'Success_Probability']).to_csv(success_prob_path, index=False)



print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: {prob:.4f}")

# Regelbasierte Auswahl des PSP für eine bestimmte Transaktion
selected_row_index = 80
selected_row = data.iloc[selected_row_index]

# Erfolgswahrscheinlichkeit für jede PSP für die ausgewählte Zeile berechnen
selected_features = selected_row[final_features].values.reshape(1, -1)
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():
    # Filtere Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Sicherstellen, dass mindestens zwei Klassen vorhanden sind
    if len(y_psp.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Standardisieren der Daten
    scaler = StandardScaler()
    X_psp_scaled = scaler.fit_transform(X_psp)
    selected_features_scaled = scaler.transform(selected_features)

    # Trainiere ein SVM-Modell
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_psp_scaled, y_psp)

    # Vorhersage der Erfolgswahrscheinlichkeit für die gewählte Zeile
    success_probability = svm_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Erfolgswahrscheinlichkeiten für die ausgewählte Transaktion ausgeben
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Extrahiere die Wahrscheinlichkeiten in eine Liste
all_probs = list(psp_success_probabilities_row.values())

# Initialisiere die Auswahl des besten PSP mit None
chosen_psp = None

# Regel 1: Falls alle Erfolgswahrscheinlichkeiten 0 sind, wähle 'Simplecard'
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Bestimme die höchste Erfolgswahrscheinlichkeit
    max_prob = max(all_probs)

    # Regel 2: Wähle 'Simplecard', falls es die höchste Wahrscheinlichkeit hat
    # oder wenn der Unterschied zur höchsten Wahrscheinlichkeit weniger als 0.1 beträgt
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Falls noch kein PSP gewählt wurde, wähle 'UK_Card', wenn es die höchste Wahrscheinlichkeit hat
    # oder wenn der Unterschied zur höchsten Wahrscheinlichkeit weniger als 0.1 beträgt
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Regel 4: Falls noch kein PSP gewählt wurde, wähle 'Moneycard', wenn es die höchste Wahrscheinlichkeit hat
    # oder wenn der Unterschied zur höchsten Wahrscheinlichkeit weniger als 0.1 beträgt
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Regel 5: Falls noch kein PSP gewählt wurde, wähle 'Goldcard', wenn es die höchste Wahrscheinlichkeit hat
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'


# Falls keine der Regeln zutrifft, Standardauswahl auf 'Simplecard'
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Endgültige Entscheidung ausgeben
print(f"\nDecision: Use {chosen_psp} as the PSP.")
