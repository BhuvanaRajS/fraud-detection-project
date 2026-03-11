import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("bank_transactions_data_2.csv")

# Create fraud label
data['Fraud'] = ((data['TransactionAmount'] > 10000) & (data['LoginAttempts'] > 3)).astype(int)

# Select columns
data = data[['TransactionAmount', 'LoginAttempts', 'AccountBalance', 'TransactionDuration', 'Channel', 'Fraud']]

# Convert channel text to number
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

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save graph
fraud_counts = data['Fraud'].value_counts()
plt.bar(['Genuine', 'Fraud'], fraud_counts)
plt.title('Fraud Percentage Graph')
plt.savefig('static/fraud_graph.png')
plt.close()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    amount = float(request.form['amount'])
    login = int(request.form['login'])
    balance = float(request.form['balance'])
    duration = float(request.form['duration'])
    channel = int(request.form['channel'])

    result = model.predict([[amount, login, balance, duration, channel]])

    if result[0] == 1:
        prediction = "Fraud Transaction Detected - Transaction Blocked"
    elif amount > 15000:
        prediction = "Suspicious Transaction - OTP Verification Required"
    else:
        prediction = "Genuine Transaction Approved"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)