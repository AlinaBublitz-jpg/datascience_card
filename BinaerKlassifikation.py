import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Load the dataset from an Excel file
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Ensure the target variable 'success' is formatted as integers (0 and 1)
data['success'] = data['success'].astype(int)

# Round the timestamps to the nearest minute for consistency
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')

# Generate a unique grouping key based on timestamp, country, and amount
data['group_key'] = data['tmsp_min'].astype(str) + '_' + data['country'] + '_' + data['amount'].astype(str)

# Calculate the number of attempts for each transaction group
data['attempt_count'] = data.groupby('group_key')['group_key'].transform('count')

# Add fees for each Payment Service Provider (PSP)
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Calculate the success rate for each PSP and map it to the dataset
psp_success_rate = data.groupby('PSP')['success'].mean()
data['psp_success_rate'] = data['PSP'].map(psp_success_rate)

# Extract the hour from the timestamp as a new feature
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Perform One-Hot-Encoding for the 'country' variable
data['original_country'] = data['country']  # Preserve the original column
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Aggregate data by group key, keeping only the first occurrence of each feature
aggregation = {
    'tmsp': 'first',  # Keep the first timestamp for each group
    'amount': 'first',  # Keep the first transaction amount
    'success': 'max',  # Use the maximum value for success (binary)
    'PSP': 'first',  # Keep the first PSP
    '3D_secured': 'first',  # Keep the first 3D secured value
    'card': 'first',  # Keep the first card type
    'attempt_count': 'first',  # Keep the first attempt count
    'success_fee': 'first',  # Keep the first success fee
    'failure_fee': 'first',  # Keep the first failure fee
    'psp_success_rate': 'first',  # Keep the first PSP success rate
    'hour': 'first',  # Keep the first hour
    'original_country': 'first'  # Keep the original country
}
# Include all dummy columns for aggregation
for col in data.columns:
    if col.startswith('country_'):
        aggregation[col] = 'first'

data = data.groupby('group_key').agg(aggregation).reset_index()

# Remove temporary columns
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Print the cleaned and aggregated data
print("\nCleaned and aggregated data with relevant fields, fees, and attempt counts:")
print(data[['tmsp', 'original_country', 'attempt_count']].head(6))

# Final features (excluding 'success')
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_Germany']

# Extract unique PSP values
psps = data['PSP'].unique()

# Initialize dictionaries for storing results and success probabilities
results = {}
psp_success_probabilities = {}

# Train and evaluate models for each PSP
for psp in psps:
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]

    # Define features (X) and target variable (y)
    X = psp_data[final_features]  # Only use features, exclude 'success'
    y = psp_data['success']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create a pipeline for scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale the features
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Logistic regression
    ])

    # Perform hyperparameter tuning with GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularization parameter
        'model__penalty': ['l1', 'l2'],  # Regularization type
        'model__solver': ['liblinear', 'saga']  # Optimization methods
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearch
    best_model = grid_search.best_estimator_

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (1)

    # Store the success probability for the current PSP
    psp_success_probabilities[psp] = y_pred_proba.mean()

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    # Store results
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': matrix,
        'Model': best_model
    }

    # Print results for the current PSP
    print(f"\nModel for PSP: {psp}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# Print success probabilities for each PSP
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: Success Probability = {prob:.2f}")

# Select a row (e.g., row with index 19)
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Calculate success probabilities for each PSP for the selected row
psp_success_probabilities_row = {}

for psp in psps:
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]

    # Define features (X) and target variable (y)
    X = psp_data[final_features]
    y = psp_data['success']

    # Check if there are at least two classes in the target variable
    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create a pipeline for scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Perform hyperparameter tuning using GridSearchCV
    param_grid = {
        'model__C': [0.1, 1, 10, 100],  # Regularization strength
        'model__penalty': ['l1', 'l2'],  # L1 or L2 regularization
        'model__solver': ['liblinear', 'saga']  # Optimization solvers
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Use the best model from GridSearch
    best_model = grid_search.best_estimator_

    # Calculate success probability for the selected row
    selected_features_scaled = best_model.named_steps['scaler'].transform(selected_features)
    success_probability = best_model.predict_proba(selected_features_scaled)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Print success probabilities for the selected row
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Extract the list of probabilities
all_probs = list(psp_success_probabilities_row.values())

# Initialize decision as None
chosen_psp = None

# Rule 1: If all probabilities are exactly 0, choose Simplecard
if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'

else:
    # Determine the maximum success probability
    max_prob = max(all_probs)

    # Rule 2: If Simplecard has the highest probability or is less than 0.1 worse than the highest, choose Simplecard
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Rule 3: If UK_Card has the highest probability or is less than 0.1 worse than the highest, choose UK_Card
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Rule 4: If Moneycard has the highest probability or is less than 0.1 worse than the highest, choose Moneycard
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Rule 5: If Goldcard has the highest probability, choose Goldcard
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: If no rule applies, choose Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Print the decision
print(f"\nDecision: Use {chosen_psp} as the PSP.")

