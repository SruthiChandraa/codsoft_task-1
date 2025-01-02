from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('titanic_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        # Prepare data for prediction
        passenger_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        scaled_data = scaler.transform(passenger_data)
        prediction = model.predict(scaled_data)

        # Return prediction result
        result = 'Survived' if prediction[0] == 1 else 'Did not survive'
        return render_template('index.html', prediction_text=f"The passenger {result}.")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
