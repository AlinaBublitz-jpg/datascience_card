import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# PSP-spezifische Merkmale hinzufügen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Gebührenstruktur des PSPs hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Transaktionskosten berechnen
data['transaction_cost'] = np.where(data['success'] == 1, data['success_fee'], data['failure_fee'])

# Zeitstempel in Stunden umwandeln
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Relevante Features auswählen
selected_features = ['psp_success_rate', 'failure_fee', 'success_fee', 'transaction_cost', '3D_secured', 'amount', 'hour']
X = data[selected_features]
y = data['success']

# Daten in Trainings- und Testdatensätze aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistische Regression trainieren
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Erfolgswahrscheinlichkeit vorhersagen
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]  # Erfolgswahrscheinlichkeit (1)

# Entscheidung basierend auf Regeln
def choose_psp_and_explain(success_prob, success_fee, failure_fee, threshold_high=0.7, threshold_low=0.4):
    explanation = ""
    rule_applied = ""
    if success_prob >= threshold_high:
        # Hohe Erfolgswahrscheinlichkeit
        chosen_psp = success_fee.idxmin()
        rule_applied = "Regel: Hohe Erfolgswahrscheinlichkeit."
        explanation = f"{rule_applied} Erfolgswahrscheinlichkeit ({success_prob:.2f}) >= {threshold_high}. Günstigster PSP ({chosen_psp}) gewählt."
    elif success_prob < threshold_low:
        # Niedrige Erfolgswahrscheinlichkeit
        chosen_psp = success_fee.idxmax()
        rule_applied = "Regel: Niedrige Erfolgswahrscheinlichkeit."
        explanation = f"{rule_applied} Erfolgswahrscheinlichkeit ({success_prob:.2f}) < {threshold_low}. PSP mit höchster Erfolgswahrscheinlichkeit ({chosen_psp}) gewählt."
    else:
        # Mittlere Erfolgswahrscheinlichkeit
        cost_score = success_prob - 0.5 * (success_fee + failure_fee)
        chosen_psp = cost_score.idxmax()
        rule_applied = "Regel: Mittlere Erfolgswahrscheinlichkeit."
        explanation = f"{rule_applied} Erfolgswahrscheinlichkeit ({success_prob:.2f}) zwischen {threshold_low} und {threshold_high}. PSP mit Balance aus Kosten und Erfolgswahrscheinlichkeit ({chosen_psp}) gewählt."
    return chosen_psp, explanation, rule_applied

# Anwendung der Regel und Erklärung auf Testdaten
chosen_psps = []
explanations = []
applied_rules = []

for i in range(len(y_pred_prob)):
    success_prob = y_pred_prob[i]
    chosen_psp, explanation, rule_applied = choose_psp_and_explain(
        success_prob, X_test['success_fee'], X_test['failure_fee']
    )
    chosen_psps.append(chosen_psp)
    explanations.append(explanation)
    applied_rules.append(rule_applied)

X_test['chosen_psp'] = chosen_psps
X_test['explanation'] = explanations
X_test['applied_rule'] = applied_rules

# Ergebnisanalyse
print("\nAnzahl der gewählten PSPs basierend auf den Regeln:")
chosen_psp_counts = X_test['chosen_psp'].value_counts()
print(chosen_psp_counts)

# Anwendung der Regeln
print("\nAnwendung der Regeln:")
print(X_test['applied_rule'].value_counts())

# Beispielhafte Entscheidungsbegründungen
print("\nBeispielhafte Entscheidungsbegründungen:")
print(X_test[['chosen_psp', 'applied_rule', 'explanation']].head(10))

# Durchschnittliche Transaktionskosten
chosen_costs = np.where(y_pred_prob >= 0.7, X_test['success_fee'], X_test['failure_fee'])
print(f"\nDurchschnittliche Transaktionskosten (gewählte Regeln): {np.mean(chosen_costs):.2f} Euro")

# Erfolgsrate der Vorhersagen
predicted_success_rate = np.mean(y_pred_prob >= 0.5)
print(f"Erfolgsrate basierend auf Regeln: {predicted_success_rate:.4f}")

# AUC-Score berechnen
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC-Score: {auc_score:.4f}")