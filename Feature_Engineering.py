import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Laden der Daten aus einer Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' als Integer gespeichert wird
data['success'] = data['success'].astype(int)

# Berechnung PSP-spezifischer Merkmale
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Hinzufügen von PSP-spezifischen Gebührenstrukturen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Transaktionsspezifische Merkmale berechnen
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour  # Extract hour from the timestamp
data['weekday'] = pd.to_datetime(data['tmsp']).dt.weekday  # Extract weekday from the timestamp
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # Flag weekends


data['attempt_count'] = data.groupby(['tmsp', 'country', 'amount']).cumcount() + 1

# One-Hot-Encoding für kategoriale Variablen
data = pd.get_dummies(data, columns=['country', 'card'], drop_first=False)

# Definieren der relevanten Features für das Modell
selected_features = [
                        'psp_success_rate', 'failure_fee', 'success_fee', '3D_secured', 'amount',
                        'hour', 'is_weekend', 'attempt_count'
                    ] + [col for col in data.columns if col.startswith('country_') or col.startswith('card_')]

#  Definieren der Eingangsvariablen (X) und der Zielvariable (y)
X = data[selected_features]
y = data['success']

# Standardisierung der Features zur Verbesserung der Modellleistung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bestimmung der Feature-Wichtigkeit mit Random Forest
print("\nRandom Forest Feature Importance:")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
rf_importance = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)
print(rf_importance)

# Visualisierung der wichtigsten Features
plt.figure(figsize=(10, 6))
rf_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.show()

# Berechnung der SHAP-Werte mit KernelExplainer
print("\nSHAP Values with KernelExplainer:")
try:
    # Verwende eine Stichprobe von 100 Datenpunkten für eine schnellere Berechnung
    sample_data = pd.DataFrame(X_scaled[:100], columns=X.columns)

    # Definiere eine Vorhersagefunktion für den SHAP-Erklärer
    def model_predict(data):
        return rf.predict_proba(data)[:, 1]  # Probabilities for class 1

    # Erstellen des KernelExplainers für das Modell
    explainer = shap.KernelExplainer(model_predict, sample_data)

    shap_values = explainer.shap_values(sample_data)

    shap_values_to_use = shap_values[0] if isinstance(shap_values, list) else shap_values

    shap_features = sample_data.columns.tolist()

    # Visualisierung SHAP-Werte
    shap.summary_plot(shap_values_to_use, sample_data, feature_names=shap_features, plot_type="bar")

    # Speichern der SHAP-Werte als CSV
    shap_df = pd.DataFrame(shap_values_to_use, columns=shap_features, index=sample_data.index)
    shap_df.to_csv('./shap_values_kernel.csv', index=False)
    print("\nSHAP values successfully computed and saved (shap_values_kernel.csv).")

except Exception as e:
    print(f"Error during SHAP computation: {e}")

# Feature Selection mit Lasso Regression
print("\nFeature Selection with Lasso Regression:")
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
lasso_coefficients = pd.Series(lasso.coef_, index=selected_features).sort_values(ascending=False)
lasso_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()
print(lasso_coefficients[lasso_coefficients != 0])

# Hauptkomponentenanalyse (PCA)
print("\nPrincipal Component Analysis (PCA):")
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance by the first 5 components: {pca.explained_variance_ratio_.sum():.4f}")

# Visualisierung der erklärten Varianz durch PCA-Komponenten
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='green')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show()

# Zusammenfassung der gewählten Features
rf_features = rf_importance.head(10).index.tolist()
shap_features = sample_data.columns.tolist()  # All features considered by SHAP
final_features = list(set(rf_features) & set(lasso_features) & set(shap_features))

# Sicherstellen, dass 'attempt_count' in den endgültigen Features enthalten ist
final_features_with_attempt_count = list(set(final_features + ['attempt_count']))

print("\nSummary of Selected Features:")
print(f"Top-10 Features from Random Forest: {rf_features}")
print(f"Features from Lasso Regression: {lasso_features}")
print(f"Final Selected Features: {final_features_with_attempt_count}")

# Speichern des finalen Datensatzes mit den gewählten Features
data[final_features_with_attempt_count].to_csv('./final_selected_features_data.csv', index=False)
print("\nDataset with final selected features saved (final_selected_features_data.csv).")
