import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistische Regression: Cross-Validation und Hyperparameter-Tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Inverser Regularisierungsparameter
        'solver': ['liblinear', 'lbfgs']  # Unterschiedliche Optimierungsmethoden
    }
    logistic_model = LogisticRegression()
    grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, scoring='roc_auc', cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Ergebnisse aus der Cross-Validation
    results = pd.DataFrame(grid_search.cv_results_)
    print("\nCross-Validation-Ergebnisse:")
    print(results[['param_C', 'param_solver', 'mean_test_score']])

    # Visualisierung der Cross-Validation-Ergebnisse
    plt.figure(figsize=(10, 6))
    for solver in results['param_solver'].unique():
        subset = results[results['param_solver'] == solver]
        plt.plot(subset['param_C'], subset['mean_test_score'], label=f"Solver: {solver}")
    plt.xscale('log')
    plt.xlabel('C (Regularisierungsparameter)')
    plt.ylabel('Mean Test AUC')
    plt.title('Cross-Validation-Ergebnisse (AUC)')
    plt.legend()
    plt.grid()
    plt.show()

    # Bestes Modell ausgeben
    best_model = grid_search.best_estimator_
    print("\nBestes Modell und Hyperparameter:")
    print(grid_search.best_params_)

    # Modell auf den Testdaten evaluieren
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für Klasse 1

    # Gesamt-Metriken berechnen
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
    best_psp = None
    best_auc = 0
    print("\nMetriken pro PSP:")
    for psp in data['PSP'].unique():
        psp_data = data[data['PSP'] == psp]  # Filter für den aktuellen PSP
        if len(psp_data['success'].unique()) > 1:  # Sicherstellen, dass beide Klassen vorhanden sind
            psp_X = psp_data[['psp_success_rate']]
            psp_y = psp_data['success']
            psp_pred = best_model.predict(psp_X)
            psp_pred_proba = best_model.predict_proba(psp_X)[:, 1]

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

            # Bester PSP basierend auf AUC
            if auc_psp > best_auc:
                best_auc = auc_psp
                best_psp = psp
        else:
            print(f"\n{psp}: Zu wenige Klassen für die Berechnung von Metriken (nur eine Klasse vorhanden).")

    # Empfehlung ausgeben
    print("\nEmpfohlener PSP:")
    print(f"Der PSP mit der besten AUC ist: {best_psp} mit einer AUC von {best_auc:.2f}")

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
