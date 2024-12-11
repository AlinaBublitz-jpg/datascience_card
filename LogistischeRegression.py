import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Load data from Excel file
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Ensure 'success' is correctly formatted as an integer for binary classification (0 or 1)
data['success'] = data['success'].astype(int)

# Reduce timestamps to minute-level precision
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')

# Create a grouping key for grouping similar transactions
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Count the number of attempts for grouped transactions
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Add PSP-specific fees (success and failure)
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Calculate the average success rate per PSP
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Extract the hour from the timestamp as a feature
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Perform one-hot encoding for the 'country' column
data['original_country'] = data['country']  # Preserve the original column for later use
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Aggregate data by the grouping key
aggregation = {
    'tmsp': 'first',
    'amount': 'first',
    'success': 'max',
    'PSP': 'first',
    '3D_secured': 'first',
    'card': 'first',
    'attempt_count': 'first',
    'success_fee': 'first',
    'failure_fee': 'first',
    'psp_success_rate': 'first',
    'hour': 'first',
    'original_country': 'first',
}

# Include all one-hot encoded country columns in the aggregation
for col in data.columns:
    if col.startswith('country_'):
        aggregation[col] = 'first'

data = data.groupby('group_key').agg(aggregation).reset_index()

# Drop temporary columns no longer needed
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Display cleaned and aggregated data
print("\nCleaned and aggregated data with relevant fields:")
print(data[['tmsp', 'original_country', 'attempt_count']].head(6))

# Define final features for the model
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Extract unique PSP values for processing
psps = data['PSP'].unique()

# Initialize dictionaries to store results and success probabilities
results = {}
psp_success_probabilities = {}

# Train and evaluate models for each PSP
for psp in psps:
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]  # Features
    y = psp_data['success']       # Target variable

    # Split data into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create a pipeline for scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic regression model
    ])

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularization strength
        'model__penalty': ['l1', 'l2'],  # Regularization type
        'model__solver': ['liblinear', 'saga']  # Solvers compatible with penalties
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Use the best model found by GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict probabilities and labels on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Store the average success probability for the PSP
    psp_success_probabilities[psp] = y_pred_proba.mean()

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    # Save results for the PSP
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': matrix,
        'Model': best_model
    }

    # Output evaluation metrics
    print(f"\nModel for PSP: {psp}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Output success probabilities for all PSPs
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: Success Probability = {prob:.2f}")

# Select a specific row (e.g., row index 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Calculate success probabilities for the selected row
psp_success_probabilities_row = {}

for psp in psps:
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]

    # Features (X) and target variable (y)
    X = psp_data[final_features]
    y = psp_data['success']

    # Ensure at least two classes exist for the PSP
    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create and train a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Scale the selected features
    selected_features_scaled = best_model.named_steps['scaler'].transform(selected_features)

    # Predict success probability for the selected row
    success_probability = best_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Output success probabilities for the selected row
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Extract probabilities into a list
all_probs = list(psp_success_probabilities_row.values())

# Initialize the chosen PSP as None
chosen_psp = None

# Rule 1: If all probabilities are 0, choose Simplecard
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Maximum success probability
    max_prob = max(all_probs)

    # Rule 2: If Simplecard is the highest or within 0.1 of the maximum, choose Simplecard
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Rule 3: If UK_Card is the highest or within 0.1 of the maximum, choose UK_Card
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Rule 4: If Moneycard is the highest or within 0.1 of the maximum, choose Moneycard
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Rule 5: If Goldcard is the highest, choose Goldcard
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: If no rules apply, choose Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Output the final decision
print(f"\nDecision: Use {chosen_psp} as the PSP.")





