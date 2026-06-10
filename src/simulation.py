from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict
import math
import random
import pandas as pd

from src.config import EMERGENCY_TYPES, RUNWAYS, FLIGHT_OPERATORS, AIRCRAFT_TYPES
from src.modeling import make_prediction


@dataclass
class Aircraft:
    flight_id: str
    operator: str
    aircraft_type: str
    x: float
    y: float
    altitude: int
    speed: int
    heading: float
    phase_of_flight: str
    emergency_type: str
    weather_condition: str
    aircraft_damage: str
    number_of_engines: int
    fatal_injuries: int
    serious_injuries: int
    minor_injuries: int
    uninjured: int
    severity_score: float
    risk_level: str = "Low"
    atc_action: str = "Continue Monitoring"
    response_time_minutes: float = 0.0
    assigned_runway: str = "Unassigned"
    rescue_status: str = "Standby"
    status: str = "Normal"

    def to_model_row(self) -> dict:
        return {
            "weather_condition": self.weather_condition,
            "phase_of_flight": self.phase_of_flight,
            "aircraft_damage": self.aircraft_damage,
            "number_of_engines": self.number_of_engines,
            "fatal_injuries": self.fatal_injuries,
            "serious_injuries": self.serious_injuries,
            "minor_injuries": self.minor_injuries,
            "uninjured": self.uninjured,
            "emergency_type": self.emergency_type,
            "severity_score": self.severity_score,
        }


class AirportSimulation:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.time_step = 0
        self.aircraft: List[Aircraft] = []
        self.events: List[Dict] = []
        self.rescue_units = {
            "ARFF-1 Fire": "Standby",
            "ARFF-2 Foam": "Standby",
            "Ambulance-1": "Standby",
            "Rescue-1122": "Standby",
            "Airside-Ops": "Standby",
            "Security-1": "Standby",
        }
        self.runway_status = {r: "Available" for r in RUNWAYS}
        for _ in range(7):
            self.spawn_aircraft(force_normal=True)
        self.log("Airport", "Allama Iqbal International Airport command dashboard initialized.")

    def log(self, event_type: str, message: str, flight_id: str = "-"):
        self.events.insert(0, {"t": self.time_step, "flight_id": flight_id, "event_type": event_type, "message": message})
        self.events = self.events[:100]

    def compute_severity(self, emergency: str, weather: str, damage: str, phase: str, fatal=0, serious=0, minor=0) -> float:
        ep = {"Engine Failure": 36, "Bird Strike": 26, "Fuel Low": 30, "Medical Emergency": 19, "Cabin Pressure Failure": 40, "Near Collision": 45, "Runway Incursion": 42, "Landing Gear Issue": 34}
        wp = {"VMC": 5, "Rain": 18, "Fog": 24, "Windy": 18, "Storm": 31, "IMC": 20}
        dp = {"None": 0, "Minor": 14, "Substantial": 32, "Destroyed": 45}
        pp = {"Takeoff": 18, "Climb": 12, "Cruise": 8, "Approach": 15, "Landing": 20, "Taxi": 5}
        return min(100, max(0, ep.get(emergency, 20) + wp.get(weather, 8) + dp.get(damage, 0) + pp.get(phase, 10) + fatal*9 + serious*4 + minor*1.5 - 3))

    def _flight_id(self, operator: str) -> str:
        prefixes = {"Emirates": "EK", "Qatar Airways": "QR", "PIA": "PK", "Gulf Air": "GF", "Saudia": "SV", "Turkish Airlines": "TK", "Etihad": "EY", "Airblue": "PA"}
        return f"{prefixes.get(operator, 'PK')}-{self.rng.randint(100, 999)}"

    def spawn_aircraft(self, force_normal: bool = False, emergency_type: str | None = None):
        operator = self.rng.choice(FLIGHT_OPERATORS)
        phase = self.rng.choice(["Approach", "Landing", "Cruise", "Climb", "Takeoff", "Taxi"])
        weather = self.rng.choice(["VMC", "Rain", "Fog", "Windy", "Storm", "IMC"])
        emergency = emergency_type or self.rng.choice(EMERGENCY_TYPES)
        damage = "None" if force_normal else self.rng.choice(["Minor", "Substantial", "None"])
        minor = 0 if force_normal else self.rng.randint(0, 7)
        serious = 0 if force_normal else self.rng.randint(0, 3)
        fatal = 0 if force_normal else self.rng.choice([0, 0, 0, 1])
        severity = 10 if force_normal else self.compute_severity(emergency, weather, damage, phase, fatal, serious, minor)
        ac = Aircraft(
            flight_id=self._flight_id(operator), operator=operator, aircraft_type=self.rng.choice(AIRCRAFT_TYPES),
            x=self.rng.uniform(6, 94), y=self.rng.uniform(8, 92), altitude=self.rng.randint(1200, 37000),
            speed=self.rng.randint(145, 470), heading=self.rng.uniform(0, 360), phase_of_flight=phase,
            emergency_type=emergency, weather_condition=weather, aircraft_damage=damage, number_of_engines=self.rng.choice([1, 2, 2, 2, 4]),
            fatal_injuries=fatal, serious_injuries=serious, minor_injuries=minor, uninjured=self.rng.randint(38, 270),
            severity_score=float(severity), status="Normal" if force_normal else "Emergency",
        )
        if not force_normal:
            pred = make_prediction(ac.to_model_row())
            ac.risk_level = pred["risk_level"]
            ac.atc_action = pred["atc_action"]
            ac.response_time_minutes = pred["response_time_minutes"]
            self.assign_resources(ac)
            self.log("Emergency", f"{ac.operator} {ac.flight_id}: {ac.emergency_type}. AI risk {ac.risk_level}; {ac.atc_action}; runway {ac.assigned_runway}.", ac.flight_id)
        self.aircraft.append(ac)
        return ac

    def assign_resources(self, ac: Aircraft):
        available = [r for r, s in self.runway_status.items() if s == "Available"] or list(self.runway_status)
        runway = available[0]
        ac.assigned_runway = runway
        if ac.risk_level in ["High", "Critical"] or ac.atc_action in ["Priority Landing", "Emergency Descent", "Runway Change"]:
            self.runway_status[runway] = f"Reserved for {ac.flight_id}"
            for unit in ["ARFF-1 Fire", "ARFF-2 Foam", "Ambulance-1", "Rescue-1122"]:
                self.rescue_units[unit] = f"Dispatched to {runway}"
            ac.rescue_status = "Rescue Dispatched"
        elif ac.risk_level == "Medium":
            self.runway_status[runway] = f"Inspection Standby {ac.flight_id}"
            self.rescue_units["Airside-Ops"] = f"Monitoring {runway}"
            ac.rescue_status = "Ops Monitoring"
        else:
            ac.rescue_status = "Standby"

    def trigger_emergency(self, emergency_type: str = "Engine Failure"):
        return self.spawn_aircraft(force_normal=False, emergency_type=emergency_type)

    def step(self):
        self.time_step += 1
        for ac in self.aircraft:
            if ac.status == "Emergency":
                ac.x += (50 - ac.x) * 0.12
                ac.y += (50 - ac.y) * 0.12
                ac.altitude = max(0, ac.altitude - 900)
                ac.speed = max(118, ac.speed - 14)
                if ac.altitude <= 450 and abs(ac.x - 50) < 7 and abs(ac.y - 50) < 7:
                    ac.status = "Landed / Response Active"
                    self.log("Landing", f"{ac.flight_id} landed. Rescue response active on runway {ac.assigned_runway}.", ac.flight_id)
            elif ac.status == "Landed / Response Active":
                ac.speed = max(0, ac.speed - 20)
            else:
                angle = math.radians(ac.heading)
                ac.x = (ac.x + math.cos(angle) * 1.4) % 100
                ac.y = (ac.y + math.sin(angle) * 1.4) % 100
                ac.heading += self.rng.uniform(-3, 3)
        if self.rng.random() < 0.06:
            self.spawn_aircraft(force_normal=True)
        if len(self.aircraft) > 16:
            self.aircraft = self.aircraft[-16:]

    def aircraft_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.aircraft])

    def event_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def rescue_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"unit": k, "status": v} for k, v in self.rescue_units.items()])

    def runway_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"runway": k, "status": v} for k, v in self.runway_status.items()])
