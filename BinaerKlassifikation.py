import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Datensatz aus einer Excel-Datei laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' als Integer (0 und 1) formatiert ist
data['success'] = data['success'].astype(int)

# Zeitstempel auf Minuten runden, um eine einheitliche Gruppierung zu ermöglichen
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')

# Gruppierungsschlüssel erstellen, der auf Zeitstempel, Land und Betrag basiert
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Anzahl der Versuche für jede Transaktionsgruppe berechnen
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Gebühren für jede PSP (Payment Service Provider) hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Erfolgsrate für jede PSP berechnen und dem Datensatz zuordnen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Stunde aus dem Zeitstempel extrahieren
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# One-Hot-Encoding für die Variable 'country' durchführen
data['original_country'] = data['country']  # Preserve the original column
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# aten nach Gruppen zusammenfassen und nur die erste relevante Information pro Gruppe behalten
aggregation = {
    'tmsp': 'first',
    'amount': 'first',
    'success': 'max',
    'PSP': 'first',
    '3D_secured': 'first',  #
    'card': 'first',
    'attempt_count': 'first',
    'success_fee': 'first',
    'failure_fee': 'first',
    'psp_success_rate': 'first',
    'hour': 'first',
    'original_country': 'first'
}

for col in data.columns:
    if col.startswith('country_'):
        aggregation[col] = 'first'

data = data.groupby('group_key').agg(aggregation).reset_index()

# Temporäre Spalten entfernen
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Bereinigte und aggregierte Daten ausgeben
print("\nCleaned and aggregated data with relevant fields, fees, and attempt counts:")
print(data[['tmsp', 'original_country', 'attempt_count']].head(6))

# Finale Feature-Auswahl (ohne die Zielvariable 'success')
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Einzigartige PSP-Werte extrahieren
psps = data['PSP'].unique()

# Dictionaries zur Speicherung der Ergebnisse und Erfolgswahrscheinlichkeiten initialisieren
results = {}
psp_success_probabilities = {}

# Modelle für jede PSP trainieren und evaluieren
for psp in psps:
    # Daten für die aktuelle PSP filtern
    psp_data = data[data['PSP'] == psp]

    # Features (X) und Zielvariable (y) definieren
    X = psp_data[final_features]
    y = psp_data['success']

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline für Skalierung und logistische Regression erstellen
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale the features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic regression
    ])

    #  Hyperparameter-Tuning mit GridSearchCV durchführen
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularization parameter
        'model__penalty': ['l1', 'l2'],  # Regularization type
        'model__solver': ['liblinear', 'saga']  # Optimization methods
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Bestes Modell aus GridSearch verwenden
    best_model = grid_search.best_estimator_

    # Vorhersagen auf Testdaten durchführen
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (1)

    # Erfolgswahrscheinlichkeit für aktuelle PSP speichern
    psp_success_probabilities[psp] = y_pred_proba.mean()

    # Modellbewertung
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    # Ergebnisse speichern
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': matrix,
        'Model': best_model
    }

    # Ergebnisse für aktuelle PSP ausgeben
    print(f"\nModel for PSP: {psp}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Erfolgswahrscheinlichkeiten für jede PSP ausgeben
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: Success Probability = {prob:.2f}")

# Zeile für Vorhersage auswählen (z. B. Zeile 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Erfolgswahrscheinlichkeiten für die gewählte Zeile berechnen
psp_success_probabilities_row = {}

for psp in psps:
    # Daten für die aktuelle PSP filtern
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Prüfen, ob genügend Klassen für Training vorhanden sind
    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Aufteilen in Trainings- und Testdatensatz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Erstellen einer Pipeline für die Skalierung und logistische Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Durchführung der Hyperparameter-Optimierung mit GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularization strength
        'model__penalty': ['l1', 'l2'],  # L1 or L2 regularization
        'model__solver': ['liblinear', 'saga']  # Optimization solvers
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Das beste Modell aus GridSearch verwenden
    best_model = grid_search.best_estimator_

    # Erfolgswahrscheinlichkeit für die ausgewählte Zeile berechnen
    selected_features_scaled = best_model.named_steps['scaler'].transform(selected_features)
    success_probability = best_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Erfolgswahrscheinlichkeiten für die ausgewählte Zeile ausgeben
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Liste aller Wahrscheinlichkeiten extrahieren
all_probs = list(psp_success_probabilities_row.values())

chosen_psp = None

# Regel 1: Falls alle Erfolgswahrscheinlichkeiten 0 sind, wähle 'Simplecard'
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Höchste Erfolgswahrscheinlichkeit bestimmen
    max_prob = max(all_probs)

    # Regel 2: Falls 'Simplecard' die höchste Wahrscheinlichkeit hat oder maximal 0.1 schlechter ist als die höchste, wähle 'Simplecard'
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Falls 'UK_Card' die höchste Wahrscheinlichkeit hat oder maximal 0.1 schlechter ist als die höchste, wähle 'UK_Card'
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Regel 4: Falls 'Moneycard' die höchste Wahrscheinlichkeit hat oder maximal 0.1 schlechter ist als die höchste, wähle 'Moneycard'
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Regel 5: Falls 'Goldcard' die höchste Wahrscheinlichkeit hat, wähle 'Goldcard'
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Falls keine der Regeln greift, Standardauswahl auf 'Simplecard'
if chosen_psp is None:
    chosen_psp = 'Simplecard'

print(f"\nDecision: Use {chosen_psp} as the PSP.")

