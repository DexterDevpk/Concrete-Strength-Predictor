from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('compressive_strength_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ash = float(request.form['ash'])
        days = float(request.form['days'])
        input_data = np.array([[ash, days]])
        prediction = model.predict(input_data)[0]
        return render_template('index.html',
                               prediction_text=f'Predicted Compressive Strength: {round(prediction, 2)} MPa')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
