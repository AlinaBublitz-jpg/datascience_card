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


# Zeile auswählen (z. B. Zeile mit Index 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Erfolgswahrscheinlichkeit für jeden PSP berechnen
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():

    # Filter für aktuellen PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Prüfen, ob mindestens zwei Klassen existieren
    if len(y.unique()) < 2:
        print(f"Nicht genug Klassen für PSP: {psp}")
        continue

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

    # Erfolgswahrscheinlichkeit für die ausgewählte Zeile berechnen
    success_probability = xgb_model.predict_proba(selected_features)[:, 1][0]
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

