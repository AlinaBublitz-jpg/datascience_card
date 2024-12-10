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


def assign_psp(transaction_data):
    """
    Wählt einen PSP basierend auf Erfolgswahrscheinlichkeit und Kosten.
    :param transaction_data: DataFrame mit Transaktionsdetails für jeden PSP
    :return: Optimaler PSP und Erklärung
    """
    psp_candidates = transaction_data['PSP'].unique()
    success_probs = {}
    costs = {
        'Moneycard': {'success_fee': 5, 'failure_fee': 2},
        'Goldcard': {'success_fee': 10, 'failure_fee': 5},
        'UK_Card': {'success_fee': 3, 'failure_fee': 1},
        'Simplecard': {'success_fee': 1, 'failure_fee': 0.5}
    }
    best_psp = None
    explanation = ""

    # Berechne Erfolgswahrscheinlichkeiten spezifisch für die Transaktion
    for psp in psp_candidates:
        psp_data = transaction_data[transaction_data['PSP'] == psp]
        model = results[psp]['Model']
        success_probs[psp] = model.predict_proba(psp_data[final_features])[:, 1][0]  # Spezifische Wahrscheinlichkeit

    # Maximale Erfolgswahrscheinlichkeit bestimmen
    max_prob = max(success_probs.values())
    sorted_probs = sorted(success_probs.items(), key=lambda x: x[1], reverse=True)

    # Regel A: Höchste Erfolgswahrscheinlichkeit über 0.8
    if max_prob > 0.8:
        best_psp = max(success_probs, key=success_probs.get)
        explanation = f"Der PSP {best_psp} hat die höchste Erfolgswahrscheinlichkeit von {max_prob:.2f}, wähle diesen PSP."

    # Regel B: Ähnliche Erfolgswahrscheinlichkeiten (Differenz < 0.1), wähle den günstigsten
    elif max_prob - sorted_probs[1][1] < 0.1:
        cheapest_psp = min(
            sorted_probs[:2],  # Nur die PSPs mit den höchsten Erfolgswahrscheinlichkeiten betrachten
            key=lambda x: costs[x[0]]['success_fee']
        )[0]
        best_psp = cheapest_psp
        explanation = f"Die PSPs haben ähnliche Erfolgswahrscheinlichkeiten. {best_psp} hat die geringsten Kosten."

    # Regel C: Alle Erfolgswahrscheinlichkeiten unter 0.8
    else:
        weighted_costs = {
            psp: costs[psp]['success_fee'] / success_probs[psp]
            for psp in success_probs.keys()
        }
        best_psp = min(weighted_costs, key=weighted_costs.get)
        explanation = f"Alle Erfolgswahrscheinlichkeiten liegen unter 0.8. {best_psp} bietet das beste Verhältnis aus Kosten und Erfolgswahrscheinlichkeit."

    return best_psp, explanation

# Beispieltransaktion bewerten
example_transaction = data.sample(1)
optimal_psp, explanation = assign_psp(example_transaction)
print("\nSpezifische Erfolgswahrscheinlichkeit für die Beispieltransaktion:")
for psp, prob in assign_psp(example_transaction)[0].items():
    print(f"{psp}: {prob:.4f}")

print("\nOptimaler PSP für die Beispieltransaktion:")
print(optimal_psp)
print("Erklärung:", explanation)

