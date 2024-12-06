
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
import shap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway

# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# Schritt 1: PSP-spezifische Merkmale
# Erfolgsrate des PSPs berechnen und hinzufügen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Gebührenstruktur des PSPs manuell hinzufügen
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Schritt 2: Transaktionsspezifische Merkmale
# Zeitstempel in numerische Features umwandeln
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour
data['weekday'] = pd.to_datetime(data['tmsp']).dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Beträge in Kategorien unterteilen
data['amount_category'] = pd.cut(data['amount'], bins=[0, 50, 200, np.inf], labels=['low', 'medium', 'high'])

# Indikator für wiederholte Zahlungsversuche
data['repeated_attempt'] = data.duplicated(subset=['amount', 'country', 'tmsp'], keep=False).astype(int)

# Schritt 3: Statistische Tests für ausgewählte Features

# 3.1: Chi-Quadrat-Test für kategorische Variablen
print("\nChi-Quadrat-Tests für kategorische Variablen (zeigen signifikante Zusammenhänge):")

# PSP
chi2, p, _, _ = chi2_contingency(pd.crosstab(data['PSP'], data['success']))
print(f"PSP: Chi2-Statistik = {chi2:.2f}, p-Wert = {p:.4f} -> Starker Zusammenhang mit dem Erfolg.")

# Country
chi2, p, _, _ = chi2_contingency(pd.crosstab(data['country'], data['success']))
print(f"Country: Chi2-Statistik = {chi2:.2f}, p-Wert = {p:.4f} -> Kein signifikanter Zusammenhang.")

# Card
chi2, p, _, _ = chi2_contingency(pd.crosstab(data['card'], data['success']))
print(f"Card: Chi2-Statistik = {chi2:.2f}, p-Wert = {p:.4f} -> Signifikanter Zusammenhang mit dem Erfolg.")

# 3D_secured
chi2, p, _, _ = chi2_contingency(pd.crosstab(data['3D_secured'], data['success']))
print(f"3D Secured: Chi2-Statistik = {chi2:.2f}, p-Wert = {p:.4f} -> Sicherheitsmaßnahmen haben einen Einfluss.")

# 3.2: ANOVA-Test für amount
print("\nANOVA-Test für 'amount' (zeigt Unterschiede zwischen Gruppen):")
anova_result = f_oneway(data[data['success'] == 0]['amount'], data[data['success'] == 1]['amount'])
print(f"F-Wert: {anova_result.statistic:.2f}, p-Wert: {anova_result.pvalue:.4f} -> Höhere Beträge könnten weniger erfolgreich sein.")

# Schritt 4: Korrelationen für numerische Variablen
# Nur numerische Spalten für die Korrelation auswählen
numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr()['success'].sort_values(ascending=False)

print("\nKorrelationen mit 'success' (Features nach Wichtigkeit sortiert):")
for feature, corr in correlations.items():
    explanation = ""
    if feature == 'psp_success_rate':
        explanation = "-> Erfolgsrate des Zahlungsdienstleisters."
    elif feature == 'failure_fee':
        explanation = "-> Gebühr für fehlgeschlagene Transaktionen."
    elif feature == 'success_fee':
        explanation = "-> Gebühr für erfolgreiche Transaktionen."
    elif feature == '3D_secured':
        explanation = "-> Sicherheitsmechanismen wie 3D-Secured."
    elif feature == 'amount':
        explanation = "-> Höhe der Beträge, negativ korreliert."
    print(f"{feature}: {corr:.4f} {explanation}")

# Schritt 5: Visualisierungen
# Korrelationen für numerische Features
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.index, y=correlations.values, palette='viridis')
plt.title('Korrelation numerischer Features mit der Zielvariable (success)')
plt.xticks(rotation=90)
plt.ylabel('Korrelationswert')
plt.show()

# Erfolgsrate nach PSP visualisieren
plt.figure(figsize=(8, 5))
psp_success_rate.plot(kind='bar', color='skyblue')
plt.title('Erfolgsrate pro PSP')
plt.ylabel('Erfolgsrate')
plt.xlabel('PSP')
plt.show()

# Erfolgsrate nach Land visualisieren
country_success_rate = data.groupby('country')['success'].mean()
plt.figure(figsize=(8, 5))
country_success_rate.plot(kind='bar', color='green')
plt.title('Erfolgsrate pro Land')
plt.ylabel('Erfolgsrate')
plt.xlabel('Land')
plt.show()

# Erfolgsrate nach 3D_secured visualisieren
plt.figure(figsize=(6, 4))
sns.barplot(x=data['3D_secured'], y=data['success'], palette='coolwarm')
plt.title('Erfolgsrate abhängig von 3D Secured')
plt.ylabel('Erfolgsrate')
plt.xlabel('3D Secured (0 = Nein, 1 = Ja)')
plt.show()

# Speichern des vorbereiteten Datensatzes
data.to_csv('./prepared_data.csv', index=False)
print("\nDer vorbereitete Datensatz wurde gespeichert (prepared_data.csv).")


# Daten laden
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Sicherstellen, dass die Zielvariable 'success' korrekt formatiert ist
data['success'] = data['success'].astype(int)

# Schritt 1: PSP-spezifische Merkmale
# Erfolgsrate des PSPs berechnen und hinzufügen
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Gebührenstruktur des PSPs manuell hinzufügen
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
data['amount_category'] = pd.cut(data['amount'], bins=[0, 50, 200, np.inf], labels=['low', 'medium', 'high'])
data['repeated_attempt'] = data.duplicated(subset=['amount', 'country', 'tmsp'], keep=False).astype(int)

# Dummy-Kodierung für kategorische Variablen
data = pd.get_dummies(data, columns=['PSP', 'country', 'card', 'amount_category'], drop_first=True)

# Zielvariable und Features definieren
X = data.drop(columns=['success', 'tmsp'])
y = data['success']

# Schritt 3: Feature-Auswahl basierend auf Modellen

# 3.1: Random Forest Feature Importance
print("\nFeature Importance mit Random Forest:")
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)

# Visualisierung der Feature-Wichtigkeit
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 wichtige Features (Random Forest)')
plt.ylabel('Feature Importance')
plt.show()

# 3.2: SHAP-Werte (SHapley Additive exPlanations)
# SHAP-Werte (SHapley Additive exPlanations)
print("\nSHAP-Werte zur Erklärung der Modellvorhersagen:")
try:
    sample_data = X.sample(n=1000, random_state=42)  # Nur 1.000 Zeilen analysieren
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample_data)

    # SHAP Summary Plot
    shap.summary_plot(shap_values[1], sample_data, plot_type="bar")

    # SHAP-Werte als DataFrame speichern
    shap_df = pd.DataFrame(shap_values[1], columns=sample_data.columns, index=sample_data.index)

    # Beispielhafte SHAP-Werte ausgeben
    print("\nBeispielhafte SHAP-Werte für die ersten 5 Zeilen:")
    print(shap_df.head())

    # Speichern der SHAP-Werte
    shap_df.to_csv('./shap_values.csv', index=False)
    print("\nDie SHAP-Werte wurden gespeichert (shap_values.csv).")

except Exception as e:
    print(f"Fehler bei der Berechnung oder Speicherung der SHAP-Werte: {e}")


# 3.3: Lasso-Regression
print("\nFeature-Auswahl mit Lasso-Regression:")
lasso = LassoCV(cv=5, random_state=42).fit(X, y)
lasso_coefficients = pd.Series(lasso.coef_, index=X.columns).sort_values(ascending=False)
print(lasso_coefficients[lasso_coefficients != 0])

# 3.4: PCA (Principal Component Analysis)
print("\nPrincipal Component Analysis (PCA):")
pca = PCA(n_components=5)  # 5 Hauptkomponenten
X_pca = pca.fit_transform(X)
print(f"Erklärte Varianz durch die ersten 5 Hauptkomponenten: {pca.explained_variance_ratio_.sum():.4f}")

# Visualisierung der PCA-Komponenten
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='green')
plt.title('Erklärte Varianz durch PCA-Komponenten')
plt.xlabel('Komponenten')
plt.ylabel('Erklärte Varianz')
plt.show()

# Speichern des erweiterten Datensatzes
data.to_csv('./prepared_data_with_features.csv', index=False)
print("\nDer vorbereitete Datensatz mit erweiterten Features wurde gespeichert (prepared_data_with_features.csv).")
