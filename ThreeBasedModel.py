import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Pfad für historische Daten
historical_data_path = './historical_data.csv'

# 🔄 Laden der neuen Daten aus der Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# 🔄 Laden und Kombinieren mit historischen Daten
if os.path.exists(historical_data_path):
    historical_data = pd.read_csv(historical_data_path)
    data = pd.concat([historical_data, data], ignore_index=True)
    data.to_csv(historical_data_path, index=False)  # Speichern der kombinierten Daten
else:
    data.to_csv(historical_data_path, index=False)

# Sicherstellen, dass die Zielvariable 'success' als Integer formatiert ist
data['success'] = data['success'].astype(int)


# Hinzufügen der PSP-spezifischen Gebührenstruktur (Erfolgs- und Misserfolgsgebühren)
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Konvertiere Zeitstempel in Stundenwerte
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Mappe 'country' auf numerische Werte (nützlich für baumbasierte Modelle)
country_mapping = {country: idx for idx, country in enumerate(data['country'].unique())}
data['country_encoded'] = data['country'].map(country_mapping)

# Definiere die finalen Features für das Modell (keine One-Hot-Encoding, stattdessen numerische Mappings)
final_features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour', 'country_encoded']

# Initialisiere Dictionaries zur Speicherung der Ergebnisse und Erfolgswahrscheinlichkeiten
results = {}
psp_success_probabilities = {}

# Schleife durch jeden PSP (Payment Service Provider), um individuelle Modelle zu trainieren
for psp in data['PSP'].unique():
    print(f"\nModel for PSP: {psp}")

    # Filtere Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]  # Features
    y_psp = psp_data['success']       # Zielvariable

    # Sicherstellen, dass es mindestens zwei Klassen für den PSP gibt
    if len(y_psp.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Teile Daten in Trainings- und Testdaten auf
    X_train, X_test, y_train, y_test = train_test_split(
        X_psp, y_psp, test_size=0.3, random_state=42, stratify=y_psp
    )

    # 🌟 Grid Search zur Optimierung von max_depth
    from sklearn.model_selection import GridSearchCV

    param_grid = {'max_depth': range(3, 20)}
    tree_model = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Bestes max_depth aus Grid Search wählen
    best_max_depth = grid_search.best_params_['max_depth']
    print(f"Optimaler max_depth für {psp}: {best_max_depth}")


    # Trainiere ein Decision Tree Classifier Modell
    tree_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    tree_model.fit(X_train, y_train)



    # Mache Vorhersagen und evaluiere das Modell
    y_pred_prob = tree_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeitsvorhersage für Klasse 1
    y_pred = tree_model.predict(X_test)  # Klassenvorhersage

    # Berechne Evaluationsmetriken
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Berechne die durchschnittliche Erfolgswahrscheinlichkeit für den gesamten PSP-Datensatz
    success_probability = tree_model.predict_proba(X_psp)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    # Speichere Ergebnisse für den aktuellen PSP
    results[psp] = {
        'Model': tree_model,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
    }

    # Gib Evaluationsmetriken aus
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Success Probability: {success_probability:.4f}")

# Gib die durchschnittlichen Erfolgswahrscheinlichkeiten für alle PSPs aus
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: {prob:.4f}")

# Regelbasierte Auswahl des PSP für eine bestimmte Transaktion
# Wähle eine bestimmte Zeile aus (z.B. Index 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Berechne die Erfolgswahrscheinlichkeit für jeden PSP für die ausgewählte Zeile
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():
    # Filtere Daten für den aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Sicherstellen, dass es mindestens zwei Klassen für den PSP gibt
    if len(y_psp.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Trainiere ein Modell für den PSP
    tree_model = DecisionTreeClassifier(max_depth=50, random_state=42)
    tree_model.fit(X_psp, y_psp)

    # Vorhersage der Erfolgswahrscheinlichkeit für die ausgewählte Zeile
    success_probability = tree_model.predict_proba(selected_features)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Gib die Erfolgswahrscheinlichkeiten für die ausgewählte Zeile aus
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Extrahiere die Liste der Erfolgswahrscheinlichkeiten
all_probs = list(psp_success_probabilities_row.values())

# Initialisiere den gewählten PSP als None
chosen_psp = None

# Regelbasierte Entscheidungsfindung
if all(prob == 0 for prob in all_probs):   # Regel 1: Wenn alle Wahrscheinlichkeiten 0 sind, wähle Simplecard
    chosen_psp = 'Simplecard'
else:
    max_prob = max(all_probs)

    # Regel 2: Wähle Simplecard, wenn es die höchste oder innerhalb von 0.1 der maximalen Wahrscheinlichkeit liegt
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Regel 3: Wähle UK_Card, wenn es die höchste oder innerhalb von 0.1 der maximalen Wahrscheinlichkeit liegt
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Regel 4: Wähle Moneycard, wenn es die höchste oder innerhalb von 0.1 der maximalen Wahrscheinlichkeit liegt
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Regel 5: Wähle Goldcard, wenn es die höchste Wahrscheinlichkeit hat
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: Wenn keine Regel zutrifft, wähle Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Gib die endgültige Entscheidung aus
print(f"\nDecision: Use {chosen_psp} as the PSP.")
