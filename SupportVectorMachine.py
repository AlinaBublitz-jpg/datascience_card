import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Zielvariable formatieren
data['success'] = data['success'].astype(int)

# Gebührenstruktur des PSPs hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Zeitstempel in Stunden umwandeln
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# One-Hot-Encoding für Länder
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Relevante finale Features
final_features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour', 'country_Germany']

# Ergebnisse speichern
results = {}
psp_success_probabilities = {}

for psp in data['PSP'].unique():
    print(f"\nModell für PSP: {psp}")

    # Daten für PSP filtern
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_psp, y_psp, test_size=0.3, random_state=42, stratify=y_psp
    )

    # Pipeline erstellen
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42))
    ])

    # Modell trainieren
    pipeline.fit(X_train, y_train)

    # Vorhersagen der Wahrscheinlichkeiten
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Erfolgswahrscheinlichkeit berechnen
    success_probability = pipeline.predict_proba(X_psp)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    results[psp] = {
        'Model': pipeline,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
    }

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Erfolgswahrscheinlichkeit: {success_probability:.4f}")

# Erfolgswahrscheinlichkeiten für alle PSPs ausgeben
print("\nErfolgswahrscheinlichkeiten für jeden PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: {prob:.4f}")

# Regelbasierte Entscheidung

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
