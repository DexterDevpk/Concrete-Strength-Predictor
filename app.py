from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Path to Models directory
models_directory = 'Models'

# Mapping of model filenames to their descriptions
model_mapping = {
    "best_model1.pkl": "Compressive strength of Sustainable concrete",
    "best_model2.pkl": "Compressive strength of Marine concrete",
    "best_model3.pkl": "Tensile strength of Sustainable Concrete",
    "best_model4.pkl": "Tensile strength of Marine Concrete",
    "best_model5.pkl": "Flexural strength of Sustainable Concrete",
    "best_model6.pkl": "Flexural strength of Marine Concrete"
}

# Dynamically fetch the models from the "Models" folder
def get_model_files():
    model_files = {}
    for model_file in os.listdir(models_directory):
        if model_file.endswith('.pkl'):
            model_name = model_file
            if model_name in model_mapping:
                model_files[model_name] = model_mapping[model_name]
    return model_files

@app.route('/')
def home():
    # Get model options dynamically
    model_options = get_model_files()
    return render_template('index.html', model_options=model_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ash = float(request.form['ash'])
        days = float(request.form['days'])
        selected_model = request.form['model']

        # Load the selected model
        model_path = os.path.join(models_directory, selected_model)
        model = joblib.load(model_path)

        input_data = np.array([[ash, days]])
        prediction = model.predict(input_data)[0]

        # Generate visualizations
        plot_url = generate_plot(ash, days, prediction)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Strength: {round(prediction, 2)} MPa',
            model_options=get_model_files(),
            selected_model=selected_model,
            plot_url=plot_url
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {e}',
            model_options=get_model_files()
        )

# Function to generate graph as a base64 string for display in the browser
def generate_plot(ash, days, prediction):
    # Creating a simple graph for visualization
    x_values = np.linspace(0, 30, 30)  # Sample range for days
    y_values = prediction * np.ones_like(x_values)  # Constant prediction line

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values, label='Predicted Strength', color='blue', linewidth=2)
    ax.scatter([days], [prediction], color='red', label='Input Prediction', zorder=5)
    ax.set_title('Predicted Strength vs. Days', fontsize=14)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Predicted Strength (MPa)', fontsize=12)
    ax.legend()

    # Convert plot to PNG image
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf8')  # Base64 encoding
    return f"data:image/png;base64,{img_b64}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
