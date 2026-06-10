from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "airport_emergency_training_data.csv"
MODEL_DIR = ROOT_DIR / "trained_models"
RISK_MODEL_PATH = MODEL_DIR / "risk_random_forest.joblib"
ATC_MODEL_PATH = MODEL_DIR / "atc_extra_trees.joblib"
RESPONSE_MODEL_PATH = MODEL_DIR / "response_gradient_boosting.joblib"
OUTPUT_DIR = ROOT_DIR / "outputs"
DOCS_DIR = ROOT_DIR / "docs"

AIRPORT_NAME = "Allama Iqbal International Airport"
AIRPORT_CODE = "LHE / OPLA"
AIRPORT_CITY = "Lahore, Pakistan"
AIRPORT_ELEVATION = "712 ft"
RUNWAYS = ["18L", "36R", "18R", "36L"]
GATES = ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
FLIGHT_OPERATORS = ["Emirates", "Qatar Airways", "PIA", "Gulf Air", "Saudia", "Turkish Airlines", "Etihad", "Airblue"]
AIRCRAFT_TYPES = ["Airbus A320", "Boeing 737", "Boeing 777", "Airbus A330", "ATR 72", "Airbus A350"]

EMERGENCY_TYPES = [
    "Engine Failure",
    "Bird Strike",
    "Fuel Low",
    "Medical Emergency",
    "Cabin Pressure Failure",
    "Near Collision",
    "Runway Incursion",
    "Landing Gear Issue",
]

EMERGENCY_VOICE_OPTIONS = [
    "Airport Rescue and Fire Fighting",
    "Ambulance Medical Response",
    "Fire Brigade Standby",
    "Rescue 1122 Coordination",
    "Security and Airside Operations",
    "ATC Priority Landing Clearance",
]
