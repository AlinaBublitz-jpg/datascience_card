import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# Schritt 1: PSP-spezifische Merkmale
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Gebührenstruktur hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Schritt 2: Transaktionsspezifische Merkmale
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour
data['weekday'] = pd.to_datetime(data['tmsp']).dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Hinzufügen von attempt_count
data['attempt_count'] = data.groupby(['tmsp', 'country', 'amount']).cumcount() + 1

# One-Hot-Encoding für kategorische Variablen
data = pd.get_dummies(data, columns=['country', 'card'], drop_first=False)

# Schritt 3: Aktualisiertes Feature-Set
selected_features = [
                        'psp_success_rate', 'failure_fee', 'success_fee', '3D_secured', 'amount',
                        'hour', 'is_weekend', 'attempt_count'
                    ] + [col for col in data.columns if col.startswith('country_') or col.startswith('card_')]

# Zielvariable und Features definieren
X = data[selected_features]
y = data['success']

# Feature-Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Schritt 4: Random Forest Feature Importance
print("\nRandom Forest Feature Importance:")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
rf_importance = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)
print(rf_importance)

# Visualisierung der Feature-Wichtigkeit
plt.figure(figsize=(10, 6))
rf_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.show()

# Schritt 5: SHAP-Werte mit KernelExplainer
print("\nSHAP-Werte mit KernelExplainer:")
try:
    # Sample-Daten (nur 100 Zeilen für schnellere Berechnung)
    sample_data = pd.DataFrame(X_scaled[:100], columns=X.columns)

    # Modellvorhersage-Funktion für KernelExplainer
    def model_predict(data):
        return rf.predict_proba(data)[:, 1]  # Wahrscheinlichkeiten für Klasse 1

    # KernelExplainer erstellen
    explainer = shap.KernelExplainer(model_predict, sample_data)

    # SHAP-Werte berechnen
    shap_values = explainer.shap_values(sample_data)

    # Sicherstellen, dass die Dimensionen passen
    shap_values_to_use = shap_values[0] if isinstance(shap_values, list) else shap_values

    # SHAP-Feature-Namen korrigieren (falls numpy-Array verwendet wurde)
    shap_features = sample_data.columns.tolist()

    # Visualisierung der SHAP-Werte
    shap.summary_plot(shap_values_to_use, sample_data, feature_names=shap_features, plot_type="bar")

    # SHAP-Werte speichern
    shap_df = pd.DataFrame(shap_values_to_use, columns=shap_features, index=sample_data.index)
    shap_df.to_csv('./shap_values_kernel.csv', index=False)
    print("\nSHAP-Werte wurden erfolgreich berechnet und gespeichert (shap_values_kernel.csv).")

except Exception as e:
    print(f"Fehler bei der SHAP-Berechnung: {e}")

# Schritt 6: Lasso-Regression für Feature-Auswahl
print("\nFeature-Auswahl mit Lasso-Regression:")
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
lasso_coefficients = pd.Series(lasso.coef_, index=selected_features).sort_values(ascending=False)
lasso_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()
print(lasso_coefficients[lasso_coefficients != 0])

# Schritt 7: PCA
print("\nPrincipal Component Analysis (PCA):")
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(f"Erklärte Varianz durch die ersten 5 Hauptkomponenten: {pca.explained_variance_ratio_.sum():.4f}")

# Visualisierung der PCA-Komponenten
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='green')
plt.title('Erklärte Varianz durch PCA-Komponenten')
plt.xlabel('Komponenten')
plt.ylabel('Erklärte Varianz')
plt.show()

# Zusammenfassung der ausgewählten Features
rf_features = rf_importance.head(10).index.tolist()
shap_features = sample_data.columns.tolist()  # Alle Features, die SHAP berücksichtigt
final_features = list(set(rf_features) & set(lasso_features) & set(shap_features))

# Hinzufügen von attempt_count zu den finalen Features
final_features_with_attempt_count = list(set(final_features + ['attempt_count']))

print("\nZusammenfassung der ausgewählten Features:")
print(f"Random Forest Top-10 Features: {rf_features}")
print(f"Lasso-Regression Features: {lasso_features}")
print(f"Gemeinsame Features (Finale Auswahl): {final_features_with_attempt_count}")

# Speichern des erweiterten Datensatzes
data[final_features_with_attempt_count].to_csv('./final_selected_features_data.csv', index=False)
print("\nDatensatz mit den finalen ausgewählten Features wurde gespeichert (final_selected_features_data.csv).")
