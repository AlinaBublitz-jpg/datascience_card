import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

# Path to the Excel file
file_path = './Excel1.xlsx'

try:
    # Read data from the Excel file
    data = pd.read_excel(file_path, engine='openpyxl')

    # Ensure the 'success' column contains only valid values (0 and 1)
    data['success'] = data['success'].astype(int)

    # Calculate PSP-specific success rates
    psp_success_rate = data.groupby('PSP')['success'].mean()
    data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

    # Print average success rates per PSP
    print("Average success rates per PSP:")
    print(psp_success_rate)

    # Define features (X) and target variable (y)
    X = data[['psp_success_rate']]  # Baseline model uses only PSP success rate
    y = data['success']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create a pipeline for feature scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale the features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic regression model
    ])

    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display overall model metrics
    print("\nOverall Model Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and display metrics for each PSP
    print("\nMetrics per PSP:")
    for psp in data['PSP'].unique():
        psp_data = data[data['PSP'] == psp]  # Filter data for the current PSP
        if len(psp_data['success'].unique()) > 1:  # Ensure both classes are present
            psp_X = psp_data[['psp_success_rate']]
            psp_y = psp_data['success']

            # Predict using the pipeline
            psp_pred = pipeline.predict(psp_X)
            psp_pred_proba = pipeline.predict_proba(psp_X)[:, 1]

            # Calculate metrics
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
            print(f"\n{psp}: Insufficient classes to calculate metrics (only one class present).")

    # Calculate the precision-recall curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Logistic Regression)')
    plt.legend()
    plt.grid()
    plt.show()

except FileNotFoundError:
    print("The Excel file was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
