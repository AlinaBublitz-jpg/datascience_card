import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Load the dataset from an Excel file
file_path = './Excel1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Ensure the target variable 'success' is properly formatted as an integer (binary: 0 or 1)
data['success'] = data['success'].astype(int)

# Add PSP-specific fee structure (success and failure fees)
fees = {
    'Moneycard': {'success_fee': 5, 'failure_fee': 2},
    'Goldcard': {'success_fee': 10, 'failure_fee': 5},
    'UK_Card': {'success_fee': 3, 'failure_fee': 1},
    'Simplecard': {'success_fee': 1, 'failure_fee': 0.5},
}
data['success_fee'] = data['PSP'].apply(lambda x: fees[x]['success_fee'])
data['failure_fee'] = data['PSP'].apply(lambda x: fees[x]['failure_fee'])

# Convert timestamps to hourly values
data['hour'] = pd.to_datetime(data['tmsp']).dt.hour

# Perform one-hot encoding for the 'country' column
data = pd.get_dummies(data, columns=['country'], drop_first=False)

# Define relevant features for the model
final_features = ['failure_fee', 'success_fee', '3D_secured', 'amount', 'hour', 'country_Germany']

# Initialize dictionaries to store results and success probabilities
results = {}
psp_success_probabilities = {}

# Loop through each PSP (Payment Service Provider) to train SVM models
for psp in data['PSP'].unique():
    print(f"\nModel for PSP: {psp}")

    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_psp, y_psp, test_size=0.3, random_state=42, stratify=y_psp
    )

    # Create a pipeline for scaling and training an SVM model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features to improve SVM performance
        ('svm', SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)  # Convert probabilities to binary predictions

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate the average success probability for the entire PSP dataset
    success_probability = pipeline.predict_proba(X_psp)[:, 1].mean()
    psp_success_probabilities[psp] = success_probability

    # Store evaluation metrics
    results[psp] = {
        'Model': pipeline,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
    }

    # Print evaluation metrics
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Success Probability: {success_probability:.4f}")

# Output success probabilities for all PSPs
print("\nSuccess probabilities for each PSP:")
for psp, prob in psp_success_probabilities.items():
    print(f"{psp}: {prob:.4f}")

# Rule-based decision
# Select a specific row (e.g., row index 80)
selected_row_index = 80
selected_row = data.iloc[selected_row_index]

# Extract features for the selected row
selected_features = selected_row[final_features].values.reshape(1, -1)

# Calculate success probabilities for the selected row for each PSP
psp_success_probabilities_row = {}

for psp in data['PSP'].unique():
    # Filter data for the current PSP
    psp_data = data[data['PSP'] == psp]
    X_psp = psp_data[final_features]
    y_psp = psp_data['success']

    # Ensure there are at least two classes for the PSP
    if len(y_psp.unique()) < 2:
        print(f"Not enough classes for PSP: {psp}")
        continue

    # Scale the data
    scaler = StandardScaler()
    X_psp_scaled = scaler.fit_transform(X_psp)
    selected_features_scaled = scaler.transform(selected_features)

    # Train an SVM model for the current PSP
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_psp_scaled, y_psp)

    # Predict the success probability for the selected row
    success_probability = svm_model.predict_proba(selected_features_scaled)[:, 1][0]
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

    # Rule 2: Choose Simplecard if it is the highest or within 0.1 of the maximum
    if 'Simplecard' in psp_success_probabilities_row:
        simplecard_prob = psp_success_probabilities_row['Simplecard']
        if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
            chosen_psp = 'Simplecard'

    # Rule 3: Choose UK_Card if it is the highest or within 0.1 of the maximum
    if chosen_psp is None and 'UK_Card' in psp_success_probabilities_row:
        uk_card_prob = psp_success_probabilities_row['UK_Card']
        if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
            chosen_psp = 'UK_Card'

    # Rule 4: Choose Moneycard if it is the highest or within 0.1 of the maximum
    if chosen_psp is None and 'Moneycard' in psp_success_probabilities_row:
        moneycard_prob = psp_success_probabilities_row['Moneycard']
        if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
            chosen_psp = 'Moneycard'

    # Rule 5: Choose Goldcard if it is the highest
    if chosen_psp is None and 'Goldcard' in psp_success_probabilities_row:
        goldcard_prob = psp_success_probabilities_row['Goldcard']
        if goldcard_prob == max_prob:
            chosen_psp = 'Goldcard'

# Fallback: If no rules apply, choose Simplecard
if chosen_psp is None:
    chosen_psp = 'Simplecard'

# Print the final decision
print(f"\nDecision: Use {chosen_psp} as the PSP.")
