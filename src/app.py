from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "waste_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Get all columns expected by model
model_columns = list(model.feature_names_in_)

# Default numeric values
default_numeric = {
    'shelf_life_days': 7,
    'unit_price': 50,
    'stock_qty': 100,
    'sales_qty': 0,
    'returned_qty': 0,
    'discount_applied': 10,
    'month': 1,
    'holiday_flag': 0,
    'temperature': 25,
    'rainfall': 0
}

# Categorical options
items = ['Bread','Croissant','Orange','Apple','Tomato','Yogurt','Potato','Butter','Mango','Carrot','Cake']
categories = ['Bakery','Fruits','Vegetables','Dairy']
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

@app.route("/", methods=["GET","POST"])
def index():
    predicted_waste = None
    reduced_waste = None

    # Initialize features to 0
    features = {col: 0 for col in model_columns}

    # Initialize form_values with defaults
    form_values = {**default_numeric, 'item_name':'Bread','category':'Bakery','day_of_week':'Monday'}

    if request.method == "POST":
        # Read numeric fields
        for field in default_numeric:
            value = request.form.get(field)
            if value:
                features[field] = float(value)
                form_values[field] = value
            else:
                features[field] = default_numeric[field]
                form_values[field] = default_numeric[field]

        # Read categorical fields
        item_name = request.form.get("item_name") or "Bread"
        category = request.form.get("category") or "Bakery"
        day_of_week = request.form.get("day_of_week") or "Monday"

        form_values["item_name"] = item_name
        form_values["category"] = category
        form_values["day_of_week"] = day_of_week

        # One-hot encoding for categorical fields
        item_col = f"item_name_{item_name}"
        if item_col in features:
            features[item_col] = 1

        category_col = f"category_{category}"
        if category_col in features:
            features[category_col] = 1

        day_col = f"day_of_week_{day_of_week}"
        if day_col in features:
            features[day_col] = 1

        # Create DataFrame
        input_data = pd.DataFrame([features], columns=model_columns)

        # Predict
        predicted_waste = model.predict(input_data)[0]
        reduced_waste = predicted_waste * (1 - features["discount_applied"]/100)

    return render_template("index.html",
                           predicted_waste=predicted_waste,
                           reduced_waste=reduced_waste,
                           form_values=form_values,
                           items=items,
                           categories=categories,
                           days=days)

if __name__ == "__main__":
    app.run(debug=True)
