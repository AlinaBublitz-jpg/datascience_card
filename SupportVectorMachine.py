import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
scaler = StandardScaler()
psp_success_probabilities = {}

for psp in data['PSP'].unique():
    print(f"\nModell für PSP: {psp}")

    # Daten für PSP filtern
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Skalieren der Features
    X_psp_scaled = pd.DataFrame(scaler.fit_transform(X_psp), columns=final_features, index=X_psp.index)

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X_psp_scaled, y_psp, test_size=0.3, random_state=42, stratify=y_psp)

    # Support Vector Machine mit RBF Kernel
    model = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)
    model.fit(X_train, y_train)

    # Vorhersagen der Wahrscheinlichkeiten
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Erfolgswahrscheinlichkeit berechnen
    X_psp_transformed = pd.DataFrame(scaler.transform(X_psp), columns=final_features, index=X_psp.index)
    success_probability = model.predict_proba(X_psp_transformed)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    results[psp] = {
        'Model': model,
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

# Beispieltransaktion bewerten
example_transaction = data.sample(1)
optimal_psp, explanation = assign_psp(example_transaction)
print("\nOptimaler PSP für die Beispieltransaktion:")
print(optimal_psp)
print("Erklärung:", explanation)
