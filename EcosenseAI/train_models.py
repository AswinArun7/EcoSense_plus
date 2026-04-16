import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- Load Dataset ----------
def load_dataset():
    possible_paths = [
        "ecosense_dataset.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecosense_dataset.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ecosense_dataset.csv"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"📥 Loading dataset from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("Dataset not found!")

# ---------- Train Model ----------
def train_model(df, feature, label):
    X = df[[feature]] if isinstance(feature, str) else df[feature]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    return model

# ---------- Train and Save All Models ----------
def train_and_save_models():
    df = load_dataset()

    models = {
        "smoke_model.pkl": train_model(df, 'SmokeValue', 'Smoke_Label'),
        "air_model.pkl": train_model(df, 'Humidity', 'Air_Label'),
        "battery_model.pkl": train_model(df, 'VibrationValue', 'Battery_Label'),
        "engine_model.pkl": train_model(df, ['OilLevel', 'Temperature', 'VibrationValue'], 'Engine_Label')
    }

    for name, model in models.items():
        joblib.dump(model, name)
        print(f"✅ Saved: {name}")

if __name__ == "__main__":
    train_and_save_models()
