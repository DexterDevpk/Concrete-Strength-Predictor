import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for matplotlib

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

volcanic_models_directory = 'Models'
flyash_models_directory = 'FA Model'

model_mapping = {
    # Volcanic Ash models
    "best_model1.pkl": "Compressive strength of Sustainable concrete",
    "best_model2.pkl": "Compressive strength of Marine concrete",
    "best_model3.pkl": "Tensile strength of Sustainable Concrete",
    "best_model4.pkl": "Tensile strength of Marine Concrete",
    "best_model5.pkl": "Flexural strength of Sustainable Concrete",
    "best_model6.pkl": "Flexural strength of Marine Concrete",
    # Fly Ash models example
    "best_modelFA.pkl": "Compressive Strength Model"
}

def get_model_files(directory):
    model_files = {}
    if not os.path.exists(directory):
        return model_files
    for model_file in os.listdir(directory):
        if model_file.endswith('.pkl'):
            description = model_mapping.get(model_file, model_file)
            model_files[model_file] = description
    return model_files

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/get_models/<ash_type>')
def get_models_api(ash_type):
    if ash_type == 'flyash':
        models_directory = flyash_models_directory
    else:
        models_directory = volcanic_models_directory
    models = get_model_files(models_directory)
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ash_type = data.get('ash_type')
    selected_model = data.get('model')
    ash = data.get('ash')
    days = data.get('days')

    # Validate inputs
    try:
        ash_val = float(ash)
        days_val = float(days)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid input values for ash percentage or days.'}), 400

    if ash_type == 'flyash':
        models_directory = flyash_models_directory
        # Exact column names used to train Fly Ash models
        input_df = pd.DataFrame([[ash_val, days_val]], columns=['FA (%)', 'Curing time (Days)'])
    else:
        models_directory = volcanic_models_directory
        # Exact column names used to train Volcanic Ash models
        input_df = pd.DataFrame([[ash_val, days_val]], columns=['Volcanic Ash (%)', 'Curing Time (Days)'])

    model_path = os.path.join(models_directory, selected_model)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Selected model file not found.'}), 400

    model = joblib.load(model_path)
    prediction = model.predict(input_df)[0]

    plot_url = generate_plot(ash_val, days_val, prediction)

    return jsonify({
        'prediction': round(prediction, 2),
        'plot_url': plot_url
    })

def generate_plot(ash, days, prediction):
    x_values = np.linspace(0, 30, 30)
    y_values = prediction * np.ones_like(x_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values, label='Predicted Strength', color='blue', linewidth=2)
    ax.scatter([days], [prediction], color='red', label='Input Prediction', zorder=5)
    ax.set_title('Predicted Strength vs. Days', fontsize=14)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Predicted Strength (MPa)', fontsize=12)
    ax.legend()

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf8')
    return f"data:image/png;base64,{img_b64}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
