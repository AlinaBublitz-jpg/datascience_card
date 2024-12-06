import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb

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

# Features und Zielvariable
features = ['PSP', 'failure_fee', 'success_fee', '3D_secured', 'amount', 'hour']
X = data[features]
y = data['success']

# Dummy-Kodierung für kategorische Variablen
X = pd.get_dummies(X, columns=['PSP'], drop_first=True)

# Datenaufteilung: Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# XGBoost-Modell
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
print(f"AUC-Score nach Korrektur: {auc:.4f}")
print(f"Accuracy nach Korrektur: {accuracy:.4f}")

# Confusion Matrix ausgeben
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
