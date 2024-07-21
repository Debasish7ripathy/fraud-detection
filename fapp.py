import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('fraud_detection_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transaction_history = request.form['transaction_history']
    employment_verification = request.form['employment_verification']
    loan_purpose = request.form['loan_purpose']
    collateral = request.form['collateral']
    geographic_location = request.form['geographic_location']
    race = request.form['race']
    previous_loan_credit_score = request.form['previous_loan_credit_score']
    annual_income = request.form['annual_income']
    loan_amount_requested = request.form['loan_amount_requested']
    tenure_period = request.form['tenure_period']

    # Prepare the data for prediction
    input_data = [
        transaction_history,
        employment_verification,
        loan_purpose,
        collateral,
        geographic_location,
        race,
        previous_loan_credit_score,
        annual_income,
        loan_amount_requested,
        tenure_period
    ]

    # Convert input data to appropriate format
    input_data = [float(x) if x.isdigit() else x for x in input_data]  # Example conversion, adjust as necessary

    # Predict using the loaded model
    prediction = model.predict([input_data])[0]

    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
