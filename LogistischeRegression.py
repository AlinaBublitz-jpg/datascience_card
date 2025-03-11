import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Laden der Daten aus einer Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')


# Laden und Kombinieren historischer Daten mit neuen Daten
data_path = './historical_data.csv'

if os.path.exists(data_path):
    historical_data = pd.read_csv(data_path)
    data = pd.concat([historical_data, data], ignore_index=True)
    data.to_csv(data_path, index=False)
else:
    # Falls keine historischen Daten existieren, speichere nur die aktuellen Daten
    data.to_csv(data_path, index=False)

print(f"📂 Daten gespeichert: {len(data)} Zeilen insgesamt.")



# Sicherstellen, dass die Zielvariable 'success' als Integer gespeichert wird (0 = Misserfolg, 1 = Erfolg)
data['success'] = data['success'].astype(int)

# Zeitstempel auf Minutenebene runden (für Gruppierungen von Transaktionen)
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')

# Erstellen eines Gruppenschlüssels zur Identifikation ähnlicher Transaktionen
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Anzahl der Versuche für jede Transaktionsgruppe berechnen
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Hinzufügen der PSP-spezifischen Gebührenstrukturen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Berechnung der durchschnittlichen Erfolgsraten pro PSP
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)



# Berechnung der Erfolgsraten pro PSP
new_success_rates = data.groupby('PSP')['success'].mean()

# Falls historische Daten existieren, Erfolgsraten vergleichen
if 'historical_data' in locals() and not historical_data.empty:
    old_success_rates = historical_data.groupby('PSP')['success'].mean()
else:
    old_success_rates = pd.Series(0, index=new_success_rates.index)  # Struktur anpassen

# Angleichung der Indizes, um Subtraktion zu ermöglichen
old_success_rates = old_success_rates.reindex(new_success_rates.index, fill_value=0)

# Berechnung der absoluten Änderungen
rate_change = abs(new_success_rates - old_success_rates)

# Falls sich die Erfolgsraten um >5% geändert haben, trainiere das Modell neu
if not (rate_change > 0.05).any():
    print("ℹ️ Keine signifikanten Änderungen – Modell-Update übersprungen.")
    update_model = False  # Kein Modell-Update nötig, aber Performance testen
else:
    print("⚙️ Signifikante Änderungen erkannt – Modell wird aktualisiert.")
    update_model = True




# Extrahieren der Stunde aus dem Zeitstempel als Feature
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# One-Hot-Encoding für die Spalte 'country'
data['original_country'] = data['country']
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Aggregieren der Daten basierend auf dem Gruppenschlüssel
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
    'original_country': 'first',
}

# Alle One-Hot-Encoded-Spalten in die Aggregation einfügen
for col in data.columns:
    if col.startswith('country_'):
        aggregation[col] = 'first'

data = data.groupby('group_key').agg(aggregation).reset_index()

# Entfernen von temporären Spalten, die nicht mehr benötigt werden
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Anzeige der bereinigten und aggregierten Daten
print("\nCleaned and aggregated data with relevant fields:")
print(data[['tmsp', 'original_country', 'attempt_count']].head(6))

# Definition der finalen Features für das Modell
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Extraktion aller einzigartigen PSPs zur Modellierung
psps = data['PSP'].unique()

# Initialisierung von Speichern für Modell-Ergebnisse und Erfolgswahrscheinlichkeiten
results = {}
psp_success_probabilities = {}

# Training und Evaluierung eines Modells für jede PSP
for psp in psps:
    # Filtern der Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Falls kein Modell-Update erforderlich ist, verwende bestehende Testdaten
    if not update_model:
        print("ℹ️ Modell bleibt unverändert, aber Performance wird mit bisherigen Testdaten berechnet.")
        # Prüfen, ob vorherige Trainings- und Testdaten existieren
        if 'X_train' in locals() and 'X_test' in locals():
            print("🔄 Verwende bestehende Train-Test-Splits für Evaluation.")
        else:
            # Falls keine gespeicherten Testdaten existieren, erstelle einen neuen Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    else:
        print("⚙️ Signifikante Änderungen erkannt – Modell wird neu trainiert.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



    # Erstellen einer Pipeline für Skalierung und logistische Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardisierung der Features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistische Regression
    ])

    # Hyperparameter-Tuning mit GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Auswahl des besten Modells aus der GridSearch
    best_model = grid_search.best_estimator_

    # Vorhersage auf Testdaten
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Speichern der durchschnittlichen Erfolgswahrscheinlichkeit für den jeweiligen PSP
    psp_success_probabilities[psp] = y_pred_proba.mean()

    # Bewertung des Modells
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    # Speichern der Ergebnisse für den aktuellen PSP
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': matrix,
        'Model': best_model
    }

    # Ausgabe der Evaluationsmetriken
    print(f"\nModel for PSP: {psp}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Ausgabe der Erfolgswahrscheinlichkeiten für alle PSPs
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: Success Probability = {prob:.2f}")

# Auswahl einer bestimmten Transaktion (z.B. Zeilenindex 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

psp_success_probabilities_row = {}

for psp in psps:
    # Filtern der Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]

    # FDefinieren der Features (X) und der Zielvariable (y)
    X = psp_data[final_features]
    y = psp_data['success']

    # Sicherstellen, dass der PSP mindestens zwei verschiedene Klassen hat (erfolgreiche und nicht erfolgreiche Transaktionen)
    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Aufteilen der Daten in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Erstellen und Trainieren eines Pipelines mit Standardisierung und logistischer Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Hyperparameter-Tuning mit GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Auswahl des besten Modells aus der Grid-Search
    best_model = grid_search.best_estimator_

    # Skalieren der ausgewählten Merkmale mit dem trainierten StandardScaler
    selected_features_scaled = best_model.named_steps['scaler'].transform(selected_features)

    # Vorhersage der Erfolgswahrscheinlichkeit für die ausgewählte Zeile
    success_probability = best_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Ausgabe der Erfolgswahrscheinlichkeiten für die ausgewählte Transaktion
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Extrahieren aller Erfolgswahrscheinlichkeiten in eine Liste
all_probs = list(psp_success_probabilities_row.values())

# Initialisierung der Wahl des besten PSPs
chosen_psp = None

# Regel 1: Falls alle Erfolgswahrscheinlichkeiten 0 sind, wähle 'Simplecard'
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Bestimmung der höchsten Erfolgswahrscheinlichkeit
    max_prob = max(all_probs)

    # Regel 2: Falls 'Simplecard' die höchste Wahrscheinlichkeit hat oder innerhalb von 0.1 der höchsten Wahrscheinlichkeit liegt, wähle 'Simplecard'
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Falls 'UK_Card' die höchste Wahrscheinlichkeit hat oder innerhalb von 0.1 der höchsten Wahrscheinlichkeit liegt, wähle 'UK_Card'
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Regel 4: Falls 'Moneycard' die höchste Wahrscheinlichkeit hat oder innerhalb von 0.1 der höchsten Wahrscheinlichkeit liegt, wähle 'Moneycard'
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Regel 5: Falls 'Goldcard' die höchste Wahrscheinlichkeit hat, wähle 'Goldcard'
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback-Regel: Falls keine der vorherigen Regeln zutrifft, wähle 'Simplecard'
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Ausgabe der endgültigen Entscheidung
print(f"\nDecision: Use {chosen_psp} as the PSP.")





