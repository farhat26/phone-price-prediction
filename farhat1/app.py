from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained Random Forest model
with open("phone_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mapping Brand and Model names to numeric IDs (adjust to your dataset)
brand_map = {
    "Samsung": 0,
    "Apple": 12,
    "Xiaomi": 9,
    "OnePlus": 15
}

model_map = {
    "Galaxy S21": 232,
    "iPhone 12": 100,
    "Redmi Note 9": 19,
    "OnePlus 8": 178
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Convert selected names to numeric IDs
        Brand_name = request.form.get("Brand")
        Model_name = request.form.get("Model")
        Brand = brand_map.get(Brand_name, 0)
        Model_num = model_map.get(Model_name, 0)

        # Get other feature inputs
        Storage = int(request.form.get("Storage", 0))
        RAM = int(request.form.get("RAM", 0))
        Screen = float(request.form.get("Screen", 0))
        Camera = int(request.form.get("Camera", 0))
        Battery = int(request.form.get("Battery", 0))

        # Create DataFrame for prediction
        input_df = pd.DataFrame([[Brand, Model_num, Storage, RAM, Screen, Camera, Battery]],
                                columns=["Brand", "Model", "Storage", "RAM", 
                                         "Screen Size (inches)", "Camera (MP)", "Battery Capacity (mAh)"])

        # Strip spaces from columns to match model
        input_df.columns = input_df.columns.str.strip()

        # Predict price
        prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction, brands=brand_map.keys(), models=model_map.keys())

if __name__ == "__main__":
    app.run(debug=True)
