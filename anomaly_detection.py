from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings

# Suppress warnings that might clutter the output
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Phase 3: Intermediate Machine Learning (Anomaly Detection) ---")

# --- 1. Data Loading ---
# Load the master file containing all engineered features
file_path = r'data\NEET_Master_Analysis_Data.csv'
try:
    df_results = pd.read_csv(file_path)
    print(f"Successfully loaded master data. Total centers: {len(df_results)}")
except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}. Please ensure it is in the same directory.")
    exit()

# --- 2. Feature Selection and One-Hot Encoding ---

# Features used for Anomaly Detection:
features = [
    'Center_v_National_Gap',  # Performance (Inequality Magnitude)
    'Center_Skewness',        # Distribution Shape (Anomaly)
    'Center_Kurtosis',        # Distribution Peak/Tails (Anomaly)
    'Ultra_High_Score_Ratio', # Top Performer Concentration
    'total_students'          # Center Size (Context)
]

# We include 'state' and use One-Hot Encoding (converting states into numerical columns)
# This allows the model to learn if being in a specific state makes a center more anomalous.
df_ml = pd.get_dummies(df_results[features + ['state']], columns=['state'], drop_first=True)

# Define the final feature list for scaling (excluding 'total_students' which isn't used as a feature, only for context)
X_features = df_ml.drop(columns=['total_students']).columns.tolist()


# --- 3. Data Scaling ---

# Standardizing (scaling) the data is mandatory for Isolation Forest.
# It prevents features with larger numerical ranges (like Center_v_National_Gap) from dominating the model.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml[X_features])


# --- 4. Model Training (Isolation Forest) ---

# Initialize the model:
# n_estimators=100 is standard.
# contamination=0.015 means we assume 1.5% of our data points (centers) are true anomalies.
model = IsolationForest(
    n_estimators=100,
    contamination=0.015,
    random_state=42,
    verbose=0
)

# Fit the model to the scaled data
model.fit(X_scaled)

# Predict the anomaly classification (-1 = Anomaly, 1 = Normal)
df_results['Anomaly_Flag'] = model.predict(X_scaled)

# Calculate the Anomaly Score (lower score = more isolated/anomalous)
df_results['Anomaly_Score'] = model.decision_function(X_scaled)


# --- 5. Integrate Results and Output ---

# Convert the numerical flag to a descriptive category
df_results['Anomaly_Type'] = df_results['Anomaly_Flag'].apply(
    lambda x: 'Anomalous Center' if x == -1 else 'Normal Center'
)

# Display the top 10 most anomalous centers (lowest scores)
df_anomalies = df_results[df_results['Anomaly_Flag'] == -1].sort_values(by='Anomaly_Score', ascending=True)

print("\n--- ML Insight: Top 10 Centers Flagged as Anomalies ---")
print("These centers are statistically the most irregular based on a combination of performance and score shape.")
print(df_anomalies[[
    'state',
    'city',
    'center_name',
    'Center_v_National_Gap',
    'Ultra_High_Score_Ratio',
    'Center_Skewness',
    'Anomaly_Score'
]].head(10).round(2).to_markdown(index=False))

# Save the master file with the ML results for dashboarding
output_file = r"data\NEET_Master_ML_Data.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✅ Phase 3 ML Complete! Anomaly results saved to {output_file}")