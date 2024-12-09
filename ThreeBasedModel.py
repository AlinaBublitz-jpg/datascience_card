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

# PSP-Kategorien behalten
unique_psps = data['PSP'].unique()

# Relevante Features
features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour']
results = {}

for psp in unique_psps:
    # Filter für den PSP
    psp_data = data[data['PSP'] == psp]

    # Prüfen, ob Daten für den PSP vorhanden sind
    if psp_data.empty:
        print(f"Keine Daten für PSP: {psp}")
        continue

    X_psp = psp_data[features]
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

    results[psp] = {
        'Model': tree_model,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
    }

    print(f"\nModell für PSP: {psp}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Regelbasierte Auswahl
def assign_psp(transaction_data):
    """
    Wählt den besten PSP basierend auf Erfolgswahrscheinlichkeit und Kosten.
    """
    best_psp = None
    best_score = float('-inf')
    explanation = ""

    for psp in unique_psps:
        psp_data = transaction_data[transaction_data['PSP'] == psp]
        if psp_data.empty:
            continue

        model = results.get(psp, {}).get('Model')
        if not model:
            continue

        success_prob = model.predict_proba(psp_data[features])[:, 1]
        avg_success_prob = success_prob.mean()

        cost = np.where(success_prob >= 0.5, psp_data['success_fee'], psp_data['failure_fee']).mean()
        score = avg_success_prob - 0.1 * cost

        if score > best_score:
            best_score = score
            best_psp = psp
            explanation = (
                f"PSP {psp} gewählt: Erfolgswahrscheinlichkeit = {avg_success_prob:.2f}, "
                f"Kosten = {cost:.2f}, Score = {best_score:.2f}."
            )

    return best_psp, explanation

# Beispieltransaktion bewerten
example_transaction = data.sample(1)
optimal_psp, explanation = assign_psp(example_transaction)
print("\nOptimaler PSP für die Beispieltransaktion:")
print(optimal_psp)
print("Erklärung:", explanation)
