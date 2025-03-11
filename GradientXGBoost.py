import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Laden der Daten aus einer Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')


# Prüfen, ob alte Daten gespeichert sind und kombinieren
data_path = './historical_data.csv'
if os.path.exists(data_path):
    historical_data = pd.read_csv(data_path)
    data = pd.concat([historical_data, data])
    data.to_csv(data_path, index=False)
else:
    data.to_csv(data_path, index=False)

# Sicherstellen, dass die Zielvariable 'success' als Integer formatiert ist
data['success'] = data['success'].astype(int)

# Zeitstempel auf die nächste Minute runden und eine Gruppierungs-ID generieren
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Berechnung der Anzahl der Versuche für jede Transaktionsgruppe
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Gebühren für die verschiedenen PSP hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Berechnung der Erfolgsrate für jeden PSP und Zuordnung zu den Daten
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Extraktion der Stunde aus dem Zeitstempel als neues Feature
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Kategorische Variable 'country' in numerische Werte umwandeln (Label-Encoding)
label_encoder = LabelEncoder()
data['country_encoded'] = label_encoder.fit_transform(data['country'])

# Aggregation der Daten nach 'group_key' (Behalten des ersten Auftretens jeder Eigenschaft)
aggregation = {
    'tmsp': 'first',
    'amount': 'first',
    'success': 'max',
    'PSP': 'first',
    '3D_secured': 'first',
    'card': 'first',
    'attempt_count': 'first',
    'success_fee': 'first',
    'failure_fee': 'first',
    'psp_success_rate': 'first',
    'hour': 'first',
    'country_encoded': 'first'
}
data = data.groupby('group_key').agg(aggregation).reset_index()

# Entfernen nicht mehr benötigter Spalten
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Definition der finalen Features für das Modelltraining und die Vorhersage
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_encoded']

# Initialisierung eines Dictionarys zur Speicherung der Ergebnisse für jeden PSP
results = {}

# Überprüfen, ob genügend neue Daten vorhanden sind
if len(data) < 100:
    print("ℹ️ Nicht genug neue Daten für ein Modell-Update.")
    exit()

# Prüfen, ob sich Erfolgsraten signifikant geändert haben
if 'historical_data' in locals() and not historical_data.empty:
    old_success_rates = historical_data.groupby('PSP')['success'].mean()
else:
    old_success_rates = pd.Series(0, index=new_success_rates.index)  # Sicherstellen, dass Struktur übereinstimmt

new_success_rates = data.groupby('PSP')['success'].mean()

# Angleichung der Indizes, um Subtraktion zu ermöglichen
old_success_rates = old_success_rates.reindex(new_success_rates.index, fill_value=0)

rate_change = abs(new_success_rates - old_success_rates)


if (rate_change > 0.05).any():
    print("⚙️ Signifikante Änderungen erkannt – Modell wird aktualisiert.")
else:
    print("ℹ️ Keine signifikanten Änderungen – Modell-Update übersprungen.")
    exit()


# Training und Bewertung von Modellen für jeden PSP
for psp in data['PSP'].unique():
    print(f"\nModel for PSP: {psp}")

    # Daten für den aktuellen PSP filtern
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Aufteilung der Daten in Trainings- und Testsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Definieren XGBoost-Modells mit spezifischen Hyperparametern
    xgb_model = xgb.XGBClassifier(
        max_depth=3,
        min_child_weight=5,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Vorhersage der Wahrscheinlichkeiten für die Testdaten
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Umwandlung der Wahrscheinlichkeiten in Klassenvorhersagen
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Modellbewertung mit verschiedenen Metriken
    auc = roc_auc_score(y_test, y_pred_prob)  # Area Under the ROC Curve
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix

    # Speicherung der Ergebnisse für den aktuellen PSP
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Model': xgb_model
    }

    # Ausgabe der Bewertungsergebnisse
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Ausgabe der Erfolgswahrscheinlichkeiten für alle PSPs
print("\nSuccess probabilities for each PSP:")
for psp, result in results.items():
    print(f"{psp}: Success Probability = {result['AUC']:.2f}")


# Regelbasierte Auswahl des besten PSP für eine spezifische Transaktion
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Erfolgswahrscheinlichkeiten für den ausgewählten Datensatz berechnen
psp_success_probabilities_row = {}

# Iteration über alle einzigartigen PSPs im Datensatz
for psp in data['PSP'].unique():
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Aufteilung der Daten in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Training des Models mit Hyperparametern
    xgb_model = xgb.XGBClassifier(
        max_depth=3,
        min_child_weight=5,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Vorhersage der Erfolgswahrscheinlichkeit für die ausgewählte Transaktion
    success_probability = xgb_model.predict_proba(selected_features)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Ausgabe der Erfolgswahrscheinlichkeiten für die gewählte Transaktion
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Regelbasierte Auswahl des besten PSP
all_probs = list(psp_success_probabilities_row.values())
chosen_psp = None

# Regel 1: Falls alle Erfolgswahrscheinlichkeiten 0 sind, wähle 'Simplecard'
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'
else:
    max_prob = max(all_probs)
    # Regel 2: Wähle 'Simplecard', falls es die höchste Wahrscheinlichkeit hat
    # oder wenn der Unterschied zur höchsten Wahrscheinlichkeit weniger als 0.1 beträgt.
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Falls noch kein PSP gewählt wurde, wähle 'UK_Card', wenn es die höchste Wahrscheinlichkeit hat
    # oder wenn der Unterschied zur höchsten Wahrscheinlichkeit weniger als 0.1 beträgt.
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
    # Regel 5: Falls noch kein PSP gewählt wurde, wähle 'Goldcard', wenn es die höchste Wahrscheinlichkeit hat.
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Falls keine der Regeln greift, setze 'Simplecard' als Standardauswahl
if chosen_psp is None:
    chosen_psp = 'Simplecard'

print(f"\nDecision: Use {chosen_psp} as the PSP.")
