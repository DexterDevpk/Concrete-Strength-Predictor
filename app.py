from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Mapping of model filenames to their descriptions
model_mapping = {
    "best_model1.pkl": "Compressive strength of Sustainable concrete",
    "best_model2.pkl": "Compressive strength of Marine concrete",
    "best_model3.pkl": "Tensile strength of Sustainable Concrete",
    "best_model4.pkl": "Tensile strength of Marine Concrete",
    "best_model5.pkl": "Flexural strength of Sustainable  Concrete",
    "best_model6.pkl": "Flexural strength of Marine Concrete"
}

@app.route('/')
def home():
    return render_template('index.html', model_options=model_mapping)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ash = float(request.form['ash'])
        days = float(request.form['days'])
        selected_model = request.form['model']

        # Load the selected model
        model = joblib.load(selected_model)

        input_data = np.array([[ash, days]])
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted Strength: {round(prediction, 2)} MPa',
            model_options=model_mapping,
            selected_model=selected_model
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {e}',
            model_options=model_mapping
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
