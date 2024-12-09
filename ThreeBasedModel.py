import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# 'country' als numerischen Wert zuordnen (statt One-Hot-Encoding)
country_mapping = {country: idx for idx, country in enumerate(data['country'].unique())}
data['country_encoded'] = data['country'].map(country_mapping)

# Finale Features ohne Hot-Encoding
final_features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour', 'country_encoded']

# Ergebnisse speichern
results = {}
psp_success_probabilities = {}

for psp in data['PSP'].unique():
    print(f"\nModell für PSP: {psp}")

    # Daten für PSP filtern
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Prüfen, ob mindestens zwei Klassen existieren
    if len(y_psp.unique()) < 2:
        print(f"Nicht genug Klassen für PSP: {psp}")
        continue

    # Datenaufteilung
    X_train, X_test, y_train, y_test = train_test_split(X_psp, y_psp, test_size=0.3, random_state=42, stratify=y_psp)

    # Decision Tree Classifier
    tree_model = DecisionTreeClassifier(max_depth=50, random_state=42)
    tree_model.fit(X_train, y_train)

    # Vorhersagen und Evaluation
    y_pred_prob = tree_model.predict_proba(X_test)[:, 1]
    y_pred = tree_model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Erfolgswahrscheinlichkeit berechnen
    success_probability = tree_model.predict_proba(X_psp)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    results[psp] = {
        'Model': tree_model,
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



# Regelbasierte Auswahl
# Berechne transaktionsspezifische Erfolgswahrscheinlichkeiten
def calculate_specific_probabilities(transaction_data):
    specific_success_probs = {}
    print("\nTransaktionsspezifische Erfolgswahrscheinlichkeiten für die Beispieltransaktion:")

    # Iteriere über die PSPs
    for psp in transaction_data['PSP'].unique():
        psp_data = transaction_data[transaction_data['PSP'] == psp]
        model = results.get(psp, {}).get('Model')

        if model and not psp_data.empty:
            # Extrahiere die Merkmale aus der Transaktionszeile
            input_features = psp_data[final_features].values.reshape(1, -1)

            # Berechne die Wahrscheinlichkeit
            specific_success_probs[psp] = model.predict_proba(input_features)[:, 1][0]
        else:
            specific_success_probs[psp] = 0.0  # Standardwert für fehlende Daten oder Modelle

    # Ausgabe der spezifischen Wahrscheinlichkeiten
    for psp, prob in specific_success_probs.items():
        print(f"{psp}: {prob:.4f}")

    return specific_success_probs


# Beispieltransaktion auswählen
example_transaction = data.sample(1)

# Globale Erfolgswahrscheinlichkeiten
print("\nGlobale Erfolgswahrscheinlichkeiten für jeden PSP:")
for psp, prob in global_success_probs.items():
    print(f"{psp}: {prob:.4f}")

# Transaktionsspezifische Wahrscheinlichkeiten berechnen
specific_probs = calculate_specific_probabilities(example_transaction)

