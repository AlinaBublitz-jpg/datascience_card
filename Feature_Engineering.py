import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Laden der Daten aus einer Excel-Datei
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' als Integer gespeichert wird
data['success'] = data['success'].astype(int)

# Aufteilen des Datensatzes in Trainings- und Testdaten, um Data Leakage zu vermeiden
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['success'])

# Berechnung PSP-spezifischer Erfolgsraten nur auf den Trainingsdaten
train_psp_success_rate = train_data.groupby('PSP')['success'].mean()

# Anwenden der auf den Trainingsdaten berechneten Erfolgsraten auf beide Datensätze
train_data['psp_success_rate'] = train_data['PSP'].map(train_psp_success_rate)
test_data['psp_success_rate'] = test_data['PSP'].map(train_psp_success_rate)
# Für PSPs, die im Testdatensatz nicht in den Trainingsdaten vorkommen, fülle mit dem globalen Mittelwert aus den Trainingsdaten
test_data['psp_success_rate'] = test_data['psp_success_rate'].fillna(train_data['success'].mean())

# Vereinigen der Trainings- und Testdaten, sodass in der weiteren Feature-Engineering-Pipeline keine Information aus den Testdaten in die PSP-Rate einfliest
data = pd.concat([train_data, test_data]).sort_index()

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

# Definieren der Eingangsvariablen (X) und der Zielvariable (y)
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

# Berechnung der SHAP-Werte mit TreeExplainer
print("\nSHAP Values with TreeExplainer:")
try:
    # Verwende eine Stichprobe von 100 Datenpunkten für eine schnellere Berechnung
    sample_data = pd.DataFrame(X_scaled[:100], columns=X.columns)

    # TreeExplainer ist für baumbasierte Modelle wie RandomForest effizienter
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample_data)

    # Überprüfen, ob shap_values ein Listentyp ist oder als Array mit 3 Dimensionen zurückgegeben wird
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1]  # für Klasse 1
    elif shap_values.ndim == 3:
        shap_values_to_use = shap_values[:, :, 1]
    else:
        shap_values_to_use = shap_values

    shap_features = sample_data.columns.tolist()

    # Extrahiere Top-40%-SHAP-Features basierend auf dem durchschnittlichen absoluten SHAP-Wert
    shap_importance_df = pd.DataFrame(shap_values_to_use, columns=shap_features).abs().mean()
    num_top = int(np.ceil(0.4 * len(shap_importance_df)))
    shap_top_features = shap_importance_df.sort_values(ascending=False).head(num_top).index.tolist()

    # Ausgabe der berechneten SHAP-Werte in der Konsole
    print("\nSHAP Average Absolute Values:")
    print(shap_importance_df)
    print("\nTop 40% SHAP Features:")
    print(shap_top_features)

    # Visualisierung der SHAP-Werte
    shap.summary_plot(shap_values_to_use, sample_data, feature_names=sample_data.columns, plot_type="bar")

    # Speichern der SHAP-Werte als CSV
    shap_df = pd.DataFrame(shap_values_to_use, columns=sample_data.columns, index=sample_data.index)
    shap_df.to_csv('./shap_values_tree.csv', index=False)
    print("\nSHAP values successfully computed and saved (shap_values_tree.csv).")

except Exception as e:
    print(f"Error during SHAP computation: {e}. SHAP analysis will be skipped.")
    # Falls ein Fehler auftritt, setzen wir shap_top_features auf alle Features
    shap_top_features = X.columns.tolist()

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

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='green')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show()

# Zusammenfassung der gewählten Features
rf_features = rf_importance.head(10).index.tolist()
# Hier verwenden wir die durch SHAP ermittelten Top-Features (Top 40% der Features)
final_features = list(set(rf_features) & set(lasso_features) & set(shap_top_features))
final_features_with_attempt_count = list(set(final_features + ['attempt_count']))

print("\nSummary of Selected Features:")
print(f"Top-10 Features from Random Forest: {rf_features}")
print(f"Features from Lasso Regression: {lasso_features}")
print(f"Top-40% SHAP Features: {shap_top_features}")
print(f"Final Selected Features: {final_features_with_attempt_count}")

# Speichern des finalen Datensatzes mit den gewählten Features
data[final_features_with_attempt_count].to_csv('./final_selected_features_data.csv', index=False)
print("\nDataset with final selected features saved (final_selected_features_data.csv).")
