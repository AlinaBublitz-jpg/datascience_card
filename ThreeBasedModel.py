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
# Zeile auswählen (z. B. Zeile mit Index 5)
selected_row_index = 14
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Erfolgswahrscheinlichkeit für jeden PSP berechnen
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():
    # Daten für PSP filtern
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Prüfen, ob mindestens zwei Klassen existieren
    if len(y_psp.unique()) < 2:
        print(f"Nicht genug Klassen für PSP: {psp}")
        continue

    # Modell trainieren
    tree_model = DecisionTreeClassifier(max_depth=50, random_state=42)
    tree_model.fit(X_psp, y_psp)

    # Erfolgswahrscheinlichkeit für die ausgewählte Zeile berechnen
    success_probability = tree_model.predict_proba(selected_features)[:, 1][0]
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
