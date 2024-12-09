import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


# Finale Features
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Modelltraining und Bewertung für jede PSP
psps = data['PSP'].unique()
results = {}

# Speichern der Erfolgswahrscheinlichkeiten für jedes PSP
psp_success_probabilities = {}

for psp in psps:
    # Filter für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modell trainieren
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für die positive Klasse (1)

    # Speichern der Erfolgswahrscheinlichkeit
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
        'Model': model
    }

    # Ergebnisse ausgeben
    print(f"\nModell für PSP: {psp}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Ausgabe der Erfolgswahrscheinlichkeiten für jedes PSP
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
    best_psp = None
    explanation = ""

    # Erfolgswahrscheinlichkeiten berechnen
    for psp in psp_candidates:
        psp_data = transaction_data[transaction_data['PSP'] == psp]
        model = results[psp]['Model']
        success_probs[psp] = model.predict_proba(psp_data[final_features])[:, 1].mean()

    # Maximale Erfolgswahrscheinlichkeit bestimmen
    max_prob = max(success_probs.values())
    sorted_probs = sorted(success_probs.items(), key=lambda x: x[1], reverse=True)

    # Regel 1: Wenn alle PSPs eine Erfolgswahrscheinlichkeit > 0.8 haben, wähle Simplecard
    if all(prob > 0.8 for prob in success_probs.values()):
        best_psp = 'Simplecard'
        explanation = "Alle PSPs haben eine Erfolgswahrscheinlichkeit über 0,8, wähle Simplecard."

    # Regel 2: Wenn alle PSPs eine Erfolgswahrscheinlichkeit < 0.2 haben, wähle den PSP mit der höchsten Erfolgswahrscheinlichkeit
    elif all(prob < 0.2 for prob in success_probs.values()):
        best_psp = sorted_probs[0][0]
        explanation = f"Alle PSPs haben eine Erfolgswahrscheinlichkeit unter 0,2, wähle den PSP mit der höchsten Erfolgswahrscheinlichkeit ({best_psp})."

    # Regel 3: Wenn Simplecard die höchste Erfolgswahrscheinlichkeit hat
    elif success_probs.get('Simplecard', 0) == max_prob:
        best_psp = 'Simplecard'
        explanation = "Simplecard hat die höchste Erfolgswahrscheinlichkeit, wähle Simplecard."

    # Regel 4: Wenn die Erfolgswahrscheinlichkeit von Simplecard um weniger als 0,1 schlechter ist als die höchste,
    # und die höchste Wahrscheinlichkeit nicht von einem anderen PSP mit größerem Abstand dominiert wird.
    elif max_prob - success_probs.get('Simplecard', 0) < 0.1 and max_prob != success_probs.get('Simplecard', 0):
        best_psp = 'Simplecard'
        explanation = "Die Erfolgswahrscheinlichkeit von Simplecard ist um weniger als 0,1 schlechter als die höchste, wähle Simplecard."

    # Regel 5: Wenn UK_Card die höchste Erfolgswahrscheinlichkeit hat
    elif success_probs.get('UK_Card', 0) == max_prob:
        best_psp = 'UK_Card'
        explanation = "UK_Card hat die höchste Erfolgswahrscheinlichkeit, wähle UK_Card."

    # Regel 6: Wenn die Erfolgswahrscheinlichkeit von UK_Card um weniger als 0,1 schlechter ist als die höchste
    elif max_prob - success_probs.get('UK_Card', 0) < 0.1 and max_prob != success_probs.get('UK_Card', 0):
        best_psp = 'UK_Card'
        explanation = "Die Erfolgswahrscheinlichkeit von UK_Card ist um weniger als 0,1 schlechter als die höchste, wähle UK_Card."

    # Regel 7: Wenn Moneycard die höchste Erfolgswahrscheinlichkeit hat
    elif success_probs.get('Moneycard', 0) == max_prob:
        best_psp = 'Moneycard'
        explanation = "Moneycard hat die höchste Erfolgswahrscheinlichkeit, wähle Moneycard."

    # Regel 8: Wenn die Erfolgswahrscheinlichkeit von Moneycard um weniger als 0,1 schlechter ist als die höchste
    elif max_prob - success_probs.get('Moneycard', 0) < 0.1 and max_prob != success_probs.get('Moneycard', 0):
        best_psp = 'Moneycard'
        explanation = "Die Erfolgswahrscheinlichkeit von Moneycard ist um weniger als 0,1 schlechter als die höchste, wähle Moneycard."

    # Regel 9: Wenn Goldcard die höchste Erfolgswahrscheinlichkeit hat
    elif success_probs.get('Goldcard', 0) == max_prob:
        best_psp = 'Goldcard'
        explanation = "Goldcard hat die höchste Erfolgswahrscheinlichkeit, wähle Goldcard."

    # Standardauswahl, falls keine Regel greift (sollte nicht vorkommen)
    else:
        best_psp = sorted_probs[0][0]
        explanation = f"Standardregel angewandt, wähle den PSP mit der höchsten Erfolgswahrscheinlichkeit ({best_psp})."

    return best_psp, explanation




# Beispielausgabe für eine Transaktion
example_transaction = data.sample(10)  # Beispiel: 10 zufällige Zeilen
optimal_psp, explanation = assign_psp(example_transaction)
print("\nBeispielausgabe:")
print(f"Optimaler PSP: {optimal_psp}")
print(f"Erklärung: {explanation}")
