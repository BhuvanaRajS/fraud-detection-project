import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("bank_transactions_data_2.csv")

# Create fraud label
data['Fraud'] = ((data['TransactionAmount'] > 10000) & (data['LoginAttempts'] > 3)).astype(int)

# Select useful columns
data = data[['TransactionAmount', 'LoginAttempts', 'AccountBalance', 'TransactionDuration', 'Channel', 'Fraud']]

# Convert Channel text to numbers
encoder = LabelEncoder()
data['Channel'] = encoder.fit_transform(data['Channel'])

# Features and target
X = data[['TransactionAmount', 'LoginAttempts', 'AccountBalance', 'TransactionDuration', 'Channel']]
y = data['Fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")

print("Model trained successfully ✅")