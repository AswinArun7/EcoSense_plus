import joblib
import requests
from datetime import datetime
import csv
import os

# ---------- Load Models ----------
def load_models():
    models = {
        "Smoke": joblib.load("smoke_model.pkl"),
        "Air": joblib.load("air_model.pkl"),
        "Battery": joblib.load("battery_model.pkl"),
        "Engine": joblib.load("engine_model.pkl")
    }
    print("✅ Models loaded successfully!")
    return models

# ---------- Fetch Sensor Data ----------
def fetch_sensor_data():
    url = "https://65e4-117-239-78-56.ngrok-free.app/arduino-data"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('data', {})
    except:
        pass

    # fallback
    return {
        'Temperature': 75,
        'Humidity': 65,
        'SmokeValue': 30,
        'OilLevel': 60,
        'VibrationValue': 45
    }

# ---------- Prediction ----------
def make_predictions(sensor_data, models):
    label_map = {
        0: "🟢 GOOD",
        1: "🟡 WARNING",
        2: "🔴 CRITICAL"
    }

    predictions = {
        'Smoke': models['Smoke'].predict([[sensor_data['SmokeValue']]])[0],
        'Air': models['Air'].predict([[sensor_data['Humidity']]])[0],
        'Battery': models['Battery'].predict([[sensor_data['VibrationValue']]])[0],
        'Engine': models['Engine'].predict([[sensor_data['OilLevel'], sensor_data['Temperature'], sensor_data['VibrationValue']]])[0]
    }

    return {k: label_map[v] for k, v in predictions.items()}

# ---------- Logging ----------
def log_predictions(predictions):
    file = "predictions_log.csv"
    row = {
        "Timestamp": datetime.now(),
        **predictions
    }

    exists = os.path.isfile(file)
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---------- Main ----------
def run_prediction():
    models = load_models()
    sensor_data = fetch_sensor_data()
    predictions = make_predictions(sensor_data, models)
    log_predictions(predictions)

    print("\n🔍 Predictions:")
    for k, v in predictions.items():
        print(f"{k}: {v}")

    return predictions

if __name__ == "__main__":
    run_prediction()
