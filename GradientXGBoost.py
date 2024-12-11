import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the data from an Excel file
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Ensure the target variable 'success' is formatted as integers (0 and 1)
data['success'] = data['success'].astype(int)

# Round the timestamps to the nearest minute and generate unique group keys
data['tmsp_min'] = pd.to_datetime(data['tmsp']).dt.floor('min')
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

# Encode categorical variable 'country' using Label Encoding
label_encoder = LabelEncoder()
data['country_encoded'] = label_encoder.fit_transform(data['country'])

# Aggregate data by the group key and keep the first occurrence of each feature
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
    'country_encoded': 'first'
}
data = data.groupby('group_key').agg(aggregation).reset_index()

# Drop temporary columns that are no longer needed
data = data.drop(columns=['group_key', 'tmsp_min'], errors='ignore')

# Define the final features to be used for training and predictions
final_features = ['psp_success_rate', '3D_secured', 'hour', 'attempt_count', 'amount', 'country_encoded']

# Initialize a dictionary to store results for each PSP
results = {}

# Train and evaluate models for each PSP
for psp in data['PSP'].unique():
    print(f"\nModel for PSP: {psp}")

    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Define the XGBoost model with specified hyperparameters
    xgb_model = xgb.XGBClassifier(
        max_depth=3,            # Maximum depth of a tree
        min_child_weight=5,     # Minimum sum of instance weight needed in a child
        gamma=1,                # Minimum loss reduction required to make a split
        subsample=0.8,          # Fraction of samples used for training each tree
        colsample_bytree=0.8,   # Fraction of features used per tree
        eval_metric='logloss',  # Evaluation metric (logarithmic loss)
        random_state=42         # Random seed for reproducibility
    )
    xgb_model.fit(X_train, y_train)

    # Predict probabilities for the test data
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Apply a threshold to predict classes
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Evaluate the model using performance metrics
    auc = roc_auc_score(y_test, y_pred_prob)  # Area Under the ROC Curve
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix

    # Save results for the current PSP
    results[psp] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Model': xgb_model
    }

    # Print the evaluation metrics
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Output success probabilities for all PSPs
print("\nSuccess probabilities for each PSP:")
for psp, result in results.items():
    print(f"{psp}: Success Probability = {result['AUC']:.2f}")

# --- Rule-Based Selection for the Best PSP ---

# Select a specific row (e.g., index 19) for prediction
selected_row_index = 19
selected_row = data.iloc[selected_row_index]
selected_features = selected_row[final_features].values.reshape(1, -1)

# Initialize a dictionary to store success probabilities for the selected row
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]
    X = psp_data[final_features]
    y = psp_data['success']

    # Ensure at least two classes exist
    if len(y.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train the XGBoost model
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

    # Predict success probability for the selected row
    success_probability = xgb_model.predict_proba(selected_features)[:, 1][0]
    psp_success_probabilities_row[psp] = success_probability

# Output success probabilities for the selected row
print("\nSuccess probabilities for the selected row:")
for psp, prob in psp_success_probabilities_row.items():
    print(f"{psp}: {prob:.4f}")

# Apply rules to decide the best PSP
all_probs = list(psp_success_probabilities_row.values())
chosen_psp = None

if all(prob == 0 for prob in all_probs):
    chosen_psp = 'Simplecard'
else:
    max_prob = max(all_probs)
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: If no rules apply, choose Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Print the final decision
print(f"\nDecision: Use {chosen_psp} as the PSP.")
