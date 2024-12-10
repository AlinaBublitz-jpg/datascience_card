import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# Zeitstempel auf Minuten reduzieren
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')  # Zeitstempel auf Minuten runden

# Gruppierungsschlüssel erstellen
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Anzahl der Versuche berechnen
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Gebühren hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# PSP-spezifische Erfolgsrate berechnen und hinzufügen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Zeitmerkmale hinzufügen
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# One-Hot-Encoding für Länder
data['original_country'] = data['country']  # Originalspalte sichern
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Aggregation inklusive Dummy-Spalten
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
    'original_country': 'first'
}
# Füge alle Dummy-Spalten zur Aggregation hinzu
for col in data.columns:
    if col.startswith('country_'):
        aggregation[col] = 'first'

data = data.groupby('group_key').agg(aggregation).reset_index()

# Temporäre Spalten entfernen
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Bereinigte und aggregierte Daten ausgeben
print("\nBereinigte und aggregierte Daten mit den relevanten Feldern, Gebühren und Anzahl der Versuche:")
print(data[['tmsp', 'original_country', 'attempt_count']].head(6))

# Finale Features (ohne 'success')
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Einzigartige PSP-Werte extrahieren
psps = data['PSP'].unique()

# Initialisiere ein Dictionary für Ergebnisse
results = {}
psp_success_probabilities = {}

# Modelle für jede PSP trainieren und bewerten
for psp in psps:
    # Filter für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]

    # Features (X) und Zielvariable (y) definieren
    X = psp_data[final_features]  # Nur Features, 'success' nicht enthalten
    y = psp_data['success']      # Zielvariable

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline erstellen: Skalierung + Logistische Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Hyperparameter-Tuning mit GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularisierungsparameter
        'model__penalty': ['l1', 'l2'],  # L1 oder L2 Regularisierung
        'model__solver': ['liblinear', 'saga']  # Optimierungsmethoden
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Bestes Modell aus GridSearch verwenden
    best_model = grid_search.best_estimator_

    # Vorhersagen auf Testdaten
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für die positive Klasse (1)

    # Erfolgswahrscheinlichkeit speichern
    psp_success_probabilities[psp] = y_pred_proba.mean()

    # Evaluation
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

    # Ergebnisse ausgeben
    print(f"\nModell für PSP: {psp}")
    print(f"Beste Hyperparameter: {grid_search.best_params_}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Erfolgswahrscheinlichkeiten für jedes PSP ausgeben
print("\nErfolgswahrscheinlichkeiten für jedes PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: Erfolgswahrscheinlichkeit = {prob:.2f}")





# Zeile auswählen (z. B. Zeile mit Index 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Erfolgswahrscheinlichkeit für jeden PSP berechnen
psp_success_probabilities_row = {}

for psp in psps:
    # Filter für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]

    # Features (X) und Zielvariable (y) definieren
    X = psp_data[final_features]
    y = psp_data['success']

    # Prüfen, ob mindestens zwei Klassen existieren
    if len(y.unique()) < 2:
        print(f"Nicht genug Klassen für PSP: {psp}")
        continue

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline erstellen
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Hyperparameter-Tuning
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Bestes Modell
    best_model = grid_search.best_estimator_

    # Erfolgswahrscheinlichkeit für die ausgewählte Zeile berechnen
    selected_features_scaled = best_model.named_steps['scaler'].transform(selected_features)
    success_probability = best_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability




# Erfolgswahrscheinlichkeiten ausgeben
print("\nErfolgswahrscheinlichkeiten für die ausgewählte Zeile:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Liste der Wahrscheinlichkeiten extrahieren
all_probs = list(psp_success_probabilities_row.values())

# Initialisiere die Entscheidung als None
chosen_psp = None

# Regel 1: Wenn alle Wahrscheinlichkeiten exakt 0 sind, wähle Simplecard
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Maximale Erfolgswahrscheinlichkeit
    max_prob = max(all_probs)

    # Regel 2: Wenn Simplecard die höchste Erfolgswahrscheinlichkeit hat oder um weniger als 0,1 schlechter ist als der Rest, wähle Simplecard
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Wenn UK_Card die höchste Erfolgswahrscheinlichkeit hat oder um weniger als 0,1 schlechter ist als der Rest, wähle UK_Card
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Regel 4: Wenn Moneycard die höchste Erfolgswahrscheinlichkeit hat oder um weniger als 0,1 schlechter ist als der Rest, wähle Moneycard
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Regel 5: Wenn Goldcard die höchste Erfolgswahrscheinlichkeit hat, wähle Goldcard
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: Wenn keine Regel zutrifft, wähle Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Entscheidung ausgeben
print(f"\nEntscheidung: Verwenden Sie {chosen_psp} als PSP.")

