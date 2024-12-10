import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

# Pfad zur Excel-Datei
file_path = './Excel1.xlsx'

try:
    # Daten aus der Excel-Datei einlesen
    data = pd.read_excel(file_path, engine='openpyxl')

    # Sicherstellen, dass 'success' nur gültige Werte (0 und 1) enthält
    data['success'] = data['success'].astype(int)

    # PSP-spezifische Erfolgsrate berechnen
    psp_success_rate = data.groupby('PSP')['success'].mean()
    data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

    # Erfolgsraten ausgeben
    print("Durchschnittliche Erfolgsraten pro PSP:")
    print(psp_success_rate)

    # Features und Zielvariable definieren
    X = data[['psp_success_rate']]  # Für das Baseline-Modell nur PSP-Erfolgsrate
    y = data['success']

    # Daten in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline erstellen: Skalierung + Logistische Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Skalierung der Features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistische Regression
    ])

    # Pipeline trainieren
    pipeline.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Gesamt-Metriken berechnen (dies bleibt unverändert)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Ergebnisse für das gesamte Modell ausgeben
    print("\nGesamte Modell-Metriken:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # PSP-spezifische Metriken berechnen und ausgeben
    print("\nMetriken pro PSP:")
    for psp in data['PSP'].unique():
        psp_data = data[data['PSP'] == psp]  # Filter für den aktuellen PSP
        if len(psp_data['success'].unique()) > 1:  # Sicherstellen, dass beide Klassen vorhanden sind
            psp_X = psp_data[['psp_success_rate']]
            psp_y = psp_data['success']

            # Vorhersagen mit der Pipeline
            psp_pred = pipeline.predict(psp_X)
            psp_pred_proba = pipeline.predict_proba(psp_X)[:, 1]

            # Berechnung der Metriken
            accuracy_psp = accuracy_score(psp_y, psp_pred)
            auc_psp = roc_auc_score(psp_y, psp_pred_proba)
            precision_psp = precision_score(psp_y, psp_pred, zero_division=0)
            recall_psp = recall_score(psp_y, psp_pred, zero_division=0)
            f1_psp = f1_score(psp_y, psp_pred)
            conf_matrix_psp = confusion_matrix(psp_y, psp_pred)

            print(f"\n{psp}:")
            print(f"  Accuracy: {accuracy_psp:.2f}")
            print(f"  AUC: {auc_psp:.2f}")
            print(f"  Precision: {precision_psp:.2f}")
            print(f"  Recall: {recall_psp:.2f}")
            print(f"  F1-Score: {f1_psp:.2f}")
            print(f"  Confusion Matrix:\n{conf_matrix_psp}")
        else:
            print(f"\n{psp}: Zu wenige Klassen für die Berechnung von Metriken (nur eine Klasse vorhanden).")

    # Precision-Recall-Kurve berechnen
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Precision-Recall-Kurve visualisieren
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Logistic Regression)')
    plt.legend()
    plt.grid()
    plt.show()

except FileNotFoundError:
    print("Die Excel-Datei wurde nicht gefunden. Bitte den Pfad überprüfen.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
