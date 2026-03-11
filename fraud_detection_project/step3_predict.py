import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# User input
amount = float(input("Enter transaction amount: "))
login = int(input("Enter login attempts: "))
balance = float(input("Enter account balance: "))
duration = float(input("Enter transaction duration: "))
channel = int(input("Enter channel (0=ATM, 1=Online, 2=Branch): "))

# Prediction
result = model.predict([[amount, login, balance, duration, channel]])

# Decision
if result[0] == 1:
    print("❌ Fraud Transaction Detected")
    print("🚫 Transaction Blocked")

elif amount > 15000:
    print("⚠️ Suspicious Transaction")
    print("🔒 Temporarily Blocked for Verification")

else:
    print("✅ Genuine Transaction")
    print("✔️ Transaction Approved")