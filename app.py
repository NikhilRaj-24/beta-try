import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import traceback
from datetime import datetime
import os

# Google Sheets setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = os.getenv('SHEET_ID')
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')

# Authenticate with Google Sheets API
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(credentials)
sheet = gc.open_by_key(SHEET_ID).sheet1

# Load the dataset to fetch unique dropdown options
df = pd.read_csv('fin-app.csv', encoding='latin1')
df.drop_duplicates(inplace=True)

# Initialize the Flask app
app = Flask(__name__)

# Preprocess the data
X = df.drop(['Price_numeric'], axis=1)
y = df['Price_numeric']

# Identify numeric, one-hot, and binary features
num_features = ['Distance_numeric', 'Age']
onehot_columns = ['Fuel Type', 'Transmission', 'City']
binary_columns = ['Make', 'Model', 'Variant']

# Define preprocessing steps
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
binary_transformer = BinaryEncoder()

# Combine the preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)
    ],
    remainder='passthrough'
)

# Train the model once when the app starts
def train_model():
    # Transform the data using the preprocessor
    X_transformed = preprocessor.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Define the RandomForestRegressor model
    best_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=40,
        bootstrap=False,
        random_state=42
    )

    # Train the model
    best_model.fit(X_train, y_train)
    
    return best_model

# Initialize and train the model once
best_model = train_model()

# Route for the main page
@app.route('/')
def index():
    makes = sorted(df['Make'].unique())
    cities = sorted(df['City'].unique())

    return render_template('index.html', makes=makes, cities=cities)

# API route to update models based on selected make
@app.route('/get_models', methods=['POST'])
def get_models():
    make = request.form['make']
    models = sorted(df[df['Make'] == make]['Model'].unique().tolist())
    return jsonify(models)

# API route to update variants based on selected model
@app.route('/get_variants', methods=['POST'])
def get_variants():
    model = request.form['model']
    variants = sorted(df[df['Model'] == model]['Variant'].unique().tolist())
    return jsonify(variants)

# API route to update transmission and fuel type based on variant
@app.route('/get_details', methods=['POST'])
def get_transmission_fuel():
    variant = request.form['variant']
    filtered_df = df[df['Variant'] == variant]
    transmission = sorted(filtered_df['Transmission'].unique().tolist())
    fuel_type = sorted(filtered_df['Fuel Type'].unique().tolist())
    return jsonify({'transmission': transmission, 'fuel_type': fuel_type})

# Route for handling form submission and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # Get user input from the form
        make = form_data['make']
        model = form_data['model']
        variant = form_data['variant']
        transmission = form_data['transmission']
        fuel_type = form_data['fuel_type']
        city = form_data['city']
        distance_numeric = float(form_data['distance_numeric'])
        age = float(form_data['age'])

        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Make': [make],
            'Model': [model],
            'Variant': [variant],
            'Transmission': [transmission],
            'Fuel Type': [fuel_type],
            'City': [city],
            'Distance_numeric': [distance_numeric],
            'Age': [age]
        })

        # Preprocess the input data
        input_transformed = preprocessor.transform(input_data)

        # Make a prediction
        predicted_price = best_model.predict(input_transformed)[0]

        # Calculate the count of occurrences in the dataset
        count = df[
            (df['Make'] == make) &
            (df['Model'] == model) &
            (df['Variant'] == variant) &
            (df['Transmission'] == transmission) &
            (df['Fuel Type'] == fuel_type)
        ].shape[0]

        # Add data to Google Sheets with placeholder for feedback and suggested price
        row = [make, model, variant, transmission, fuel_type, city, distance_numeric, age,
               predicted_price, "", "", str(datetime.now())]

        sheet.append_row(row)
        row_number = len(sheet.get_all_values())  # Get the row number for future updates

        return jsonify({'predicted_price': round(predicted_price, -2), 'row_number': row_number, 'count': count})

    except Exception as e:
        # Print stack trace for debugging
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

# Route for updating feedback and suggested price
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        form_data = request.form

        row_number = int(form_data['row_number'])  # Row number to update
        suggested_price = form_data.get('suggested_price', '')
        feedback = form_data.get('feedback', '')

        # Update feedback and suggested price in the same row
        sheet.update(f'J{row_number}:K{row_number}', [[suggested_price, feedback]])

        return jsonify({'success': True})

    except Exception as e:
        # Print stack trace for debugging
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred during feedback submission. Please try again.'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
