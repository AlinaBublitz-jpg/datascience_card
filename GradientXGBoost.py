import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Zielvariable formatieren
data['success'] = data['success'].astype(int)

# Zeitstempel in Minuten runden und Features generieren
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Anzahl der Versuche berechnen
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Gebührenstruktur des PSPs hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# PSP-spezifische Erfolgsrate berechnen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Zeitmerkmale hinzufügen
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Label Encoding für kategorische Variablen (z. B. country)
label_encoder = LabelEncoder()
data['country_encoded'] = label_encoder.fit_transform(data['country'])

# Aggregation (ohne One-Hot-Encoding)
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

# Temporäre Spalten entfernen
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Finale Features
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_encoded']

# Ergebnisse speichern
results = {}

# Modelltraining und Bewertung für jede PSP
for psp in data['PSP'].unique():
    print(f"\nModell für PSP: {psp}")

    # Filter für aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # GradientBoosting-Modell (XGBoost)
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

    # Vorhersagen der Wahrscheinlichkeiten
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Schwelle anwenden, um Klassen vorherzusagen
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Modellbewertung
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Ergebnisse speichern
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Model': xgb_model
    }

    # Ergebnisse ausgeben
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Erfolgswahrscheinlichkeiten für alle PSPs ausgeben
print("\nErfolgswahrscheinlichkeiten für jedes PSP:")
for psp, result in results.items():
    print(f"{psp}: Erfolgswahrscheinlichkeit = {result['AUC']:.2f}")



# Regelbasierter Ansatz zur Auswahl des PSP mit verbesserter Erklärung

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

    print("\nTransaktionsdetails:")
    print(transaction_data)

    # Erfolgswahrscheinlichkeiten berechnen
    for psp in psp_candidates:
        psp_data = transaction_data[transaction_data['PSP'] == psp]

        if psp_data.empty:
            print(f"Keine Daten für PSP: {psp}")
            continue

        model = results.get(psp, {}).get('Model')
        if model is None:
            print(f"Kein Modell für PSP: {psp}")
            continue

        try:
            success_probs[psp] = model.predict_proba(psp_data[final_features])[:, 1].mean()
        except Exception as e:
            print(f"Fehler bei der Vorhersage für PSP {psp}: {e}")
            success_probs[psp] = 0

    # Debugging-Ausgabe
    print("\nBerechnete Erfolgswahrscheinlichkeiten:")
    for psp, prob in success_probs.items():
        print(f"{psp}: {prob:.2f}")

    # Maximale Erfolgswahrscheinlichkeit bestimmen
    max_prob = max(success_probs.values(), default=0)
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

    # Regel 4: Wenn die Erfolgswahrscheinlichkeit von Simplecard um weniger als 0,1 schlechter ist als die höchste
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
        best_psp = sorted_probs[0][0] if sorted_probs else None
        explanation = f"Standardregel angewandt, wähle den PSP mit der höchsten Erfolgswahrscheinlichkeit ({best_psp})."

    return best_psp, explanation



# Beispieltransaktion
example_transaction = data.sample(1)
optimal_psp, explanation = assign_psp(example_transaction)
print("\nOptimaler PSP für die Beispieltransaktion:")
print(optimal_psp)
print("Erklärung:", explanation)
