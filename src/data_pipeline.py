from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_DATA_PATH, EMERGENCY_TYPES

PHASES = ["Takeoff", "Climb", "Cruise", "Approach", "Landing", "Taxi"]
WEATHER = ["VMC", "IMC", "Rain", "Fog", "Windy", "Storm"]
DAMAGE = ["None", "Minor", "Substantial", "Destroyed"]
ATC_ACTIONS = ["Priority Landing", "Holding Pattern", "Emergency Descent", "Runway Change", "Continue Monitoring", "Go Around"]
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def get_col(df: pd.DataFrame, names: list[str], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df))


def score_to_risk(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def action_from_risk(row: pd.Series) -> str:
    risk = row["risk_level"]
    emergency = row["emergency_type"]
    phase = row["phase_of_flight"]
    if risk == "Critical":
        return "Priority Landing"
    if emergency == "Cabin Pressure Failure":
        return "Emergency Descent"
    if emergency == "Near Collision":
        return "Go Around"
    if risk == "High" and phase in ["Cruise", "Climb"]:
        return "Holding Pattern"
    if risk == "High":
        return "Runway Change"
    return "Continue Monitoring"


def build_from_real_csvs(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    frames = []
    for file in raw_dir.glob("*.csv"):
        df = safe_read_csv(file)
        if df.empty:
            continue
        df = normalize_columns(df)
        out = pd.DataFrame()
        out["weather_condition"] = get_col(df, ["weather_condition", "weather", "wx_condition"], "VMC").fillna("VMC")
        out["phase_of_flight"] = get_col(df, ["broad_phase_of_flight", "phase_of_flight"], "Landing").fillna("Landing")
        out["aircraft_damage"] = get_col(df, ["aircraft_damage", "damage"], "Minor").fillna("Minor")
        out["number_of_engines"] = pd.to_numeric(get_col(df, ["number_of_engines", "engines"], 2), errors="coerce").fillna(2)
        out["fatal_injuries"] = pd.to_numeric(get_col(df, ["total_fatal_injuries", "fatal_injuries"], 0), errors="coerce").fillna(0)
        out["serious_injuries"] = pd.to_numeric(get_col(df, ["total_serious_injuries", "serious_injuries"], 0), errors="coerce").fillna(0)
        out["minor_injuries"] = pd.to_numeric(get_col(df, ["total_minor_injuries", "minor_injuries"], 0), errors="coerce").fillna(0)
        out["uninjured"] = pd.to_numeric(get_col(df, ["total_uninjured", "uninjured"], 50), errors="coerce").fillna(50)
        out["emergency_type"] = np.random.choice(EMERGENCY_TYPES, len(out))
        frames.append(out.head(8000))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def generate_synthetic_data(n: int = 3500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "weather_condition": rng.choice(WEATHER, n, p=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05]),
        "phase_of_flight": rng.choice(PHASES, n),
        "aircraft_damage": rng.choice(DAMAGE, n, p=[0.45, 0.30, 0.20, 0.05]),
        "number_of_engines": rng.choice([1, 2, 3, 4], n, p=[0.15, 0.65, 0.05, 0.15]),
        "fatal_injuries": rng.poisson(0.18, n),
        "serious_injuries": rng.poisson(0.45, n),
        "minor_injuries": rng.poisson(1.2, n),
        "uninjured": rng.integers(20, 250, n),
        "emergency_type": rng.choice(EMERGENCY_TYPES, n),
    })
    return df


def enrich_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weather_points = {"VMC": 5, "IMC": 18, "Rain": 18, "Fog": 22, "Windy": 18, "Storm": 30}
    damage_points = {"None": 0, "Minor": 12, "Substantial": 30, "Destroyed": 45}
    emergency_points = {
        "Engine Failure": 35,
        "Bird Strike": 25,
        "Fuel Low": 28,
        "Medical Emergency": 18,
        "Cabin Pressure Failure": 38,
        "Near Collision": 42,
    }
    phase_points = {"Takeoff": 18, "Climb": 12, "Cruise": 8, "Approach": 15, "Landing": 20, "Taxi": 5}

    for col, default in [
        ("weather_condition", "VMC"), ("phase_of_flight", "Landing"), ("aircraft_damage", "Minor"), ("emergency_type", "Medical Emergency")]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": default, "": default})

    for col in ["number_of_engines", "fatal_injuries", "serious_injuries", "minor_injuries", "uninjured"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["severity_score"] = (
        df["weather_condition"].map(weather_points).fillna(12)
        + df["aircraft_damage"].map(damage_points).fillna(10)
        + df["emergency_type"].map(emergency_points).fillna(20)
        + df["phase_of_flight"].map(phase_points).fillna(10)
        + df["fatal_injuries"].clip(0, 10) * 9
        + df["serious_injuries"].clip(0, 15) * 4
        + df["minor_injuries"].clip(0, 20) * 1.5
        - df["number_of_engines"].clip(1, 4) * 1.5
    ).clip(0, 100)

    df["risk_level"] = df["severity_score"].apply(score_to_risk)
    df["atc_action"] = df.apply(action_from_risk, axis=1)
    df["response_time_minutes"] = (
        4 + df["severity_score"] * 0.18
        + df["weather_condition"].isin(["Fog", "Storm", "IMC"]).astype(int) * 4
        + df["phase_of_flight"].isin(["Cruise", "Climb"]).astype(int) * 3
        + np.random.default_rng(7).normal(0, 2, len(df))
    ).clip(3, 35).round(1)
    return df


def prepare_training_data(output_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    real_df = build_from_real_csvs()
    synth_df = generate_synthetic_data()
    df = pd.concat([real_df, synth_df], ignore_index=True) if not real_df.empty else synth_df
    df = enrich_targets(df)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    data = prepare_training_data()
    print(f"Prepared training data: {data.shape} -> {PROCESSED_DATA_PATH}")
