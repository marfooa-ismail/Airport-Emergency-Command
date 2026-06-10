from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.config import PROCESSED_DATA_PATH, MODEL_DIR, RISK_MODEL_PATH, ATC_MODEL_PATH, RESPONSE_MODEL_PATH
from src.data_pipeline import prepare_training_data

FEATURES = [
    "weather_condition", "phase_of_flight", "aircraft_damage", "number_of_engines",
    "fatal_injuries", "serious_injuries", "minor_injuries", "uninjured",
    "emergency_type", "severity_score",
]
CATEGORICAL = ["weather_condition", "phase_of_flight", "aircraft_damage", "emergency_type"]
NUMERIC = ["number_of_engines", "fatal_injuries", "serious_injuries", "minor_injuries", "uninjured", "severity_score"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def load_training_frame(data_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        return prepare_training_data(data_path)
    df = pd.read_csv(data_path, encoding="latin1", low_memory=False)
    if df.empty or not set(FEATURES + ["risk_level", "atc_action", "response_time_minutes"]).issubset(df.columns):
        return prepare_training_data(data_path)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC + ["response_time_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["risk_level"] = df["risk_level"].fillna("Medium").astype(str)
    df["atc_action"] = df["atc_action"].fillna("Continue Monitoring").astype(str)
    return df.dropna(subset=["risk_level", "atc_action", "response_time_minutes"]).reset_index(drop=True)


def split_classification(X: pd.DataFrame, y: pd.Series):
    if y.nunique() > 1 and y.value_counts().min() >= 2:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_save_models(data_path: Path = PROCESSED_DATA_PATH) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_training_frame(data_path)
    X = df[FEATURES]
    metrics = {}

    X_train, X_test, y_train, y_test = split_classification(X, df["risk_level"])
    risk_model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ])
    risk_model.fit(X_train, y_train)
    risk_pred = risk_model.predict(X_test)
    metrics["risk_accuracy"] = round(float(accuracy_score(y_test, risk_pred)), 3)
    metrics["risk_report"] = classification_report(y_test, risk_pred, zero_division=0)
    joblib.dump(risk_model, RISK_MODEL_PATH)

    X_train, X_test, y_train, y_test = split_classification(X, df["atc_action"])
    atc_model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])
    atc_model.fit(X_train, y_train)
    atc_pred = atc_model.predict(X_test)
    metrics["atc_accuracy"] = round(float(accuracy_score(y_test, atc_pred)), 3)
    metrics["atc_report"] = classification_report(y_test, atc_pred, zero_division=0)
    joblib.dump(atc_model, ATC_MODEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, df["response_time_minutes"], test_size=0.2, random_state=42)
    # Use RandomForestRegressor instead of GradientBoostingRegressor so saved models
    # remain more stable across Python/scikit-learn versions.
    response_model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1)),
    ])
    response_model.fit(X_train, y_train)
    response_pred = response_model.predict(X_test)
    metrics["response_mae"] = round(float(mean_absolute_error(y_test, response_pred)), 3)
    metrics["response_r2"] = round(float(r2_score(y_test, response_pred)), 3)
    joblib.dump(response_model, RESPONSE_MODEL_PATH)
    return metrics


def models_exist() -> bool:
    return RISK_MODEL_PATH.exists() and ATC_MODEL_PATH.exists() and RESPONSE_MODEL_PATH.exists()


def load_models() -> tuple:
    """Load trained models, rebuilding them automatically if joblib files are missing
    or incompatible with the user's Python/scikit-learn version.

    This prevents errors such as: ModuleNotFoundError: No module named '_loss'.
    """
    if not models_exist():
        train_and_save_models()
    try:
        return joblib.load(RISK_MODEL_PATH), joblib.load(ATC_MODEL_PATH), joblib.load(RESPONSE_MODEL_PATH)
    except Exception:
        # Old joblib/sklearn artifacts can break on newer Python installations.
        # Delete and rebuild all models from the included dataset.
        for model_path in (RISK_MODEL_PATH, ATC_MODEL_PATH, RESPONSE_MODEL_PATH):
            try:
                model_path.unlink(missing_ok=True)
            except TypeError:  # Python < 3.8 fallback
                if model_path.exists():
                    model_path.unlink()
        train_and_save_models()
        return joblib.load(RISK_MODEL_PATH), joblib.load(ATC_MODEL_PATH), joblib.load(RESPONSE_MODEL_PATH)


def make_prediction(input_row: dict) -> dict:
    risk_model, atc_model, response_model = load_models()
    X = pd.DataFrame([input_row])[FEATURES]
    return {
        "risk_level": str(risk_model.predict(X)[0]),
        "atc_action": str(atc_model.predict(X)[0]),
        "response_time_minutes": round(float(response_model.predict(X)[0]), 1),
    }
