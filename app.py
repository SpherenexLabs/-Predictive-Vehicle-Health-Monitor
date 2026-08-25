

from flask import Flask, jsonify, request, render_template_string
from collections import deque
from datetime import datetime
import math
import random
import threading

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ============================================================
# AI-BASED PREDICTIVE VEHICLE HEALTH MONITORING SYSTEM
# Single-file Flask + HTML/CSS/JS + Random Forest demo
# Hardware values are simulated for prototype/demo use.
# ============================================================

FEATURES = [
    "engine_temp",
    "coolant_level",
    "battery_voltage",
    "battery_current",
    "steering_angle",
    "brake_pressure",
    "engine_vibration",
    "battery_vibration",
    "gear_position",
]

FAULT_LABELS = {
    "NORMAL": "Healthy",
    "ENGINE_OVERHEAT": "Engine Overheat",
    "COOLANT_LOW": "Low Coolant",
    "BATTERY_FAULT": "Battery Fault",
    "BRAKE_FAULT": "Brake System Warning",
    "HIGH_VIBRATION": "High Vibration",
}

FAULT_RECOMMENDATIONS = {
    "NORMAL": "Vehicle parameters are within the expected operating range.",
    "ENGINE_OVERHEAT": "Inspect cooling fan, coolant circulation and radiator. Avoid prolonged high-load operation.",
    "COOLANT_LOW": "Check coolant level, hose leakage and reservoir condition. Refill only after safe inspection.",
    "BATTERY_FAULT": "Inspect battery terminals, charging circuit and alternator output. Check battery condition.",
    "BRAKE_FAULT": "Inspect brake pressure, hydraulic line and braking mechanism before further operation.",
    "HIGH_VIBRATION": "Inspect engine mounting, rotating components, wheel balance and battery mounting.",
}

GEARS = ["P", "R", "N", "D", "1", "2", "3", "4", "5"]
GEAR_TO_NUM = {g: i for i, g in enumerate(GEARS)}

rng = np.random.default_rng(42)
history = deque(maxlen=90)
lock = threading.Lock()

simulator = {
    "mode": "AUTO",
    "active_fault": "NORMAL",
    "ticks_left": 0,
    "phase": 0.0,
}


def clamp(value, low, high):
    return max(low, min(high, value))


def synthesize_training_data(samples=9000):
    """
    Create a synthetic training dataset for the prototype.
    The final hardware project should replace this with real, labelled vehicle data.
    """
    X = []
    y = []

    classes = list(FAULT_LABELS.keys())

    for _ in range(samples):
        label = rng.choice(
            classes,
            p=[0.48, 0.11, 0.10, 0.11, 0.09, 0.11],
        )

        engine_temp = rng.normal(84, 7)
        coolant = rng.normal(78, 11)
        voltage = rng.normal(12.7, 0.45)
        current = rng.normal(28, 13)
        steering = rng.normal(0, 14)
        brake = rng.normal(53, 17)
        engine_vib = abs(rng.normal(2.1, 0.9))
        battery_vib = abs(rng.normal(1.0, 0.45))
        gear = int(rng.integers(0, len(GEARS)))

        if label == "ENGINE_OVERHEAT":
            engine_temp = rng.normal(112, 7)
            coolant -= abs(rng.normal(12, 7))
            engine_vib += abs(rng.normal(1.8, 0.8))

        elif label == "COOLANT_LOW":
            coolant = rng.normal(18, 8)
            engine_temp += abs(rng.normal(7, 5))

        elif label == "BATTERY_FAULT":
            if rng.random() < 0.5:
                voltage = rng.normal(10.8, 0.45)
            else:
                voltage = rng.normal(15.4, 0.45)
            current = rng.normal(78, 18)
            battery_vib += abs(rng.normal(1.3, 0.7))

        elif label == "BRAKE_FAULT":
            if rng.random() < 0.5:
                brake = rng.normal(12, 7)
            else:
                brake = rng.normal(96, 5)

        elif label == "HIGH_VIBRATION":
            engine_vib = abs(rng.normal(8.4, 1.4))
            battery_vib = abs(rng.normal(4.6, 1.0))

        row = [
            clamp(engine_temp, 40, 140),
            clamp(coolant, 0, 100),
            clamp(voltage, 8, 17),
            clamp(current, -30, 130),
            clamp(steering, -50, 50),
            clamp(brake, 0, 110),
            clamp(engine_vib, 0, 14),
            clamp(battery_vib, 0, 9),
            gear,
        ]
        X.append(row)
        y.append(label)

    return np.array(X, dtype=float), np.array(y)


X, y = synthesize_training_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=180,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)
model.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, model.predict(X_test))


def choose_auto_fault():
    # Mostly healthy, with occasional temporary abnormalities to demonstrate prediction.
    return random.choices(
        population=[
            "NORMAL",
            "ENGINE_OVERHEAT",
            "COOLANT_LOW",
            "BATTERY_FAULT",
            "BRAKE_FAULT",
            "HIGH_VIBRATION",
        ],
        weights=[58, 9, 8, 9, 7, 9],
        k=1,
    )[0]


def health_score(sample):
    score = 100.0

    temp = sample["engine_temp"]
    coolant = sample["coolant_level"]
    voltage = sample["battery_voltage"]
    current = abs(sample["battery_current"])
    brake = sample["brake_pressure"]
    ev = sample["engine_vibration"]
    bv = sample["battery_vibration"]

    if temp > 95:
        score -= min(30, (temp - 95) * 1.8)
    if temp < 60:
        score -= min(8, (60 - temp) * 0.4)

    if coolant < 55:
        score -= min(25, (55 - coolant) * 0.55)

    voltage_deviation = abs(voltage - 12.7)
    if voltage_deviation > 1.0:
        score -= min(22, (voltage_deviation - 1.0) * 9)

    if current > 65:
        score -= min(15, (current - 65) * 0.3)

    if brake < 20 or brake > 90:
        score -= 18

    if ev > 4.5:
        score -= min(20, (ev - 4.5) * 3.5)

    if bv > 2.8:
        score -= min(12, (bv - 2.8) * 3.0)

    return round(clamp(score, 5, 100), 1)


def build_sensor_sample():
    with lock:
        simulator["phase"] += 0.22
        phase = simulator["phase"]

        requested_mode = simulator["mode"]

        if requested_mode != "AUTO":
            fault = requested_mode
        else:
            if simulator["ticks_left"] <= 0:
                simulator["active_fault"] = choose_auto_fault()
                simulator["ticks_left"] = random.randint(7, 18)
            fault = simulator["active_fault"]
            simulator["ticks_left"] -= 1

    # Smooth normal baseline + small random noise.
    engine_temp = 84 + 3.5 * math.sin(phase / 3.2) + random.gauss(0, 1.2)
    coolant = 79 + 5.0 * math.sin(phase / 5.0) + random.gauss(0, 1.6)
    voltage = 12.7 + 0.18 * math.sin(phase / 2.7) + random.gauss(0, 0.08)
    current = 28 + 7.5 * math.sin(phase / 2.1) + random.gauss(0, 2.5)
    steering = 22 * math.sin(phase / 1.7) + random.gauss(0, 2.0)
    brake = 48 + 21 * max(0, math.sin(phase / 2.4)) + random.gauss(0, 3.0)
    engine_vib = abs(2.0 + 0.6 * math.sin(phase * 1.8) + random.gauss(0, 0.25))
    battery_vib = abs(0.9 + 0.25 * math.sin(phase * 1.3) + random.gauss(0, 0.12))
    gear = random.choices(
        ["P", "R", "N", "D", "1", "2", "3", "4", "5"],
        weights=[3, 3, 4, 35, 14, 14, 11, 9, 7],
        k=1,
    )[0]

    if fault == "ENGINE_OVERHEAT":
        engine_temp += random.uniform(24, 34)
        coolant -= random.uniform(8, 18)
        engine_vib += random.uniform(1.4, 2.7)

    elif fault == "COOLANT_LOW":
        coolant = random.uniform(8, 28)
        engine_temp += random.uniform(5, 11)

    elif fault == "BATTERY_FAULT":
        voltage = random.choice([
            random.uniform(10.2, 11.3),
            random.uniform(14.9, 16.1),
        ])
        current = random.uniform(68, 96)
        battery_vib += random.uniform(1.0, 2.2)

    elif fault == "BRAKE_FAULT":
        brake = random.choice([
            random.uniform(4, 17),
            random.uniform(94, 103),
        ])

    elif fault == "HIGH_VIBRATION":
        engine_vib = random.uniform(7.1, 10.5)
        battery_vib = random.uniform(3.7, 6.2)

    sample = {
        "engine_temp": round(clamp(engine_temp, 35, 140), 1),
        "coolant_level": round(clamp(coolant, 0, 100), 1),
        "battery_voltage": round(clamp(voltage, 8, 17), 2),
        "battery_current": round(clamp(current, -30, 130), 1),
        "steering_angle": round(clamp(steering, -50, 50), 1),
        "brake_pressure": round(clamp(brake, 0, 110), 1),
        "engine_vibration": round(clamp(engine_vib, 0, 14), 2),
        "battery_vibration": round(clamp(battery_vib, 0, 9), 2),
        "gear_position": gear,
    }

    features = np.array([[
        sample["engine_temp"],
        sample["coolant_level"],
        sample["battery_voltage"],
        sample["battery_current"],
        sample["steering_angle"],
        sample["brake_pressure"],
        sample["engine_vibration"],
        sample["battery_vibration"],
        GEAR_TO_NUM[sample["gear_position"]],
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities) * 100.0)

    score = health_score(sample)

    if score >= 80 and prediction == "NORMAL":
        condition = "Healthy"
    elif score < 50 or prediction in {"ENGINE_OVERHEAT", "BRAKE_FAULT"}:
        condition = "Critical"
    else:
        condition = "Warning"

    # Prototype-only RUL estimate based on current synthetic health state.
    remaining_useful_life = int(round(1500 * (score / 100.0) ** 1.7))

    sample.update({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "iso_time": datetime.now().isoformat(timespec="seconds"),
        "health_score": score,
        "condition": condition,
        "fault_code": prediction,
        "fault": FAULT_LABELS[prediction],
        "prediction_confidence": round(confidence, 1),
        "recommendation": FAULT_RECOMMENDATIONS[prediction],
        "remaining_useful_life_hours": remaining_useful_life,
        "simulated_scenario": fault,
        "model_accuracy": round(float(test_accuracy) * 100.0, 2),
    })

    return sample


def maintenance_insights(sample):
    items = []

    if sample["engine_temp"] > 100:
        items.append("Cooling system inspection is recommended.")
    if sample["coolant_level"] < 35:
        items.append("Coolant level is low; inspect reservoir and possible leakage.")
    if sample["battery_voltage"] < 11.6 or sample["battery_voltage"] > 14.8:
        items.append("Battery/charging voltage is outside the normal demo range.")
    if sample["engine_vibration"] > 5.5:
        items.append("Elevated engine vibration detected; inspect mounts and rotating parts.")
    if sample["battery_vibration"] > 3.2:
        items.append("Battery mounting vibration is high; inspect mounting security.")
    if sample["brake_pressure"] < 20 or sample["brake_pressure"] > 90:
        items.append("Brake pressure requires inspection.")
    if not items:
        items.append("No immediate maintenance action is predicted from the current sample.")

    return items[:3]


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/telemetry")
def telemetry():
    sample = build_sensor_sample()
    sample["insights"] = maintenance_insights(sample)

    with lock:
        history.append(sample.copy())

    return jsonify(sample)


@app.route("/api/history")
def get_history():
    with lock:
        return jsonify(list(history))


@app.route("/api/simulator", methods=["POST"])
def set_simulator():
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "AUTO")).upper()

    allowed = {
        "AUTO",
        "NORMAL",
        "ENGINE_OVERHEAT",
        "COOLANT_LOW",
        "BATTERY_FAULT",
        "BRAKE_FAULT",
        "HIGH_VIBRATION",
    }

    if mode not in allowed:
        return jsonify({"ok": False, "error": "Invalid simulator mode"}), 400

    with lock:
        simulator["mode"] = mode
        simulator["ticks_left"] = 0

    return jsonify({"ok": True, "mode": mode})


@app.route("/api/model")
def model_info():
    importances = sorted(
        zip(FEATURES, model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return jsonify({
        "algorithm": "RandomForestClassifier",
        "estimators": model.n_estimators,
        "test_accuracy": round(float(test_accuracy) * 100.0, 2),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "feature_importance": [
            {"feature": name, "importance": round(value * 100, 2)}
            for name, value in importances
        ],
        "note": "The demo model is trained on synthetic data. Replace it with labelled real vehicle data for final validation.",
    })


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Vehicle Health AI</title>

    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>

    <style>
        :root {
            --bg: #07101f;
            --panel: rgba(14, 27, 48, 0.82);
            --panel-2: rgba(20, 37, 63, 0.72);
            --line: rgba(255,255,255,0.09);
            --text: #f5f8ff;
            --muted: #91a1bb;
            --cyan: #48d8ff;
            --blue: #6d7cff;
            --green: #55e59d;
            --amber: #ffca5b;
            --red: #ff6474;
            --shadow: 0 24px 80px rgba(0,0,0,.32);
        }

        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 10%, rgba(72,216,255,.14), transparent 32%),
                radial-gradient(circle at 85% 0%, rgba(109,124,255,.16), transparent 32%),
                linear-gradient(180deg, #07101f 0%, #091426 55%, #060d18 100%);
            overflow-x: hidden;
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
            background-size: 34px 34px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,.7), transparent 80%);
        }

        .shell {
            width: min(1500px, calc(100% - 28px));
            margin: 0 auto;
            padding: 22px 0 34px;
            position: relative;
            z-index: 1;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 18px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 13px;
        }

        .brand-icon {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            border-radius: 15px;
            font-weight: 900;
            letter-spacing: -1px;
            background: linear-gradient(135deg, rgba(72,216,255,.22), rgba(109,124,255,.3));
            border: 1px solid rgba(72,216,255,.35);
            box-shadow: 0 0 30px rgba(72,216,255,.15);
        }

        .brand h1 {
            margin: 0;
            font-size: clamp(18px, 2.4vw, 29px);
            line-height: 1.05;
            letter-spacing: -0.6px;
        }

        .brand p {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 13px;
        }

        .connection {
            display: flex;
            align-items: center;
            gap: 9px;
            color: #c8d4e8;
            font-size: 13px;
            padding: 10px 13px;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,255,255,.035);
            white-space: nowrap;
        }

        .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 14px rgba(85,229,157,.8);
            animation: pulse 1.7s infinite;
        }

        @keyframes pulse {
            50% { transform: scale(.72); opacity: .62; }
        }

        .grid {
            display: grid;
            grid-template-columns: 1.35fr .95fr;
            gap: 16px;
        }

        .card {
            background: linear-gradient(160deg, rgba(18,35,59,.86), rgba(9,20,36,.82));
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }

        .hero {
            min-height: 410px;
            position: relative;
            overflow: hidden;
        }

        .hero-overlay {
            position: absolute;
            inset: 0;
            z-index: 2;
            pointer-events: none;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .eyebrow {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.6px;
            color: var(--cyan);
            font-weight: 800;
        }

        .hero-title {
            margin-top: 7px;
            font-size: 18px;
            font-weight: 800;
        }

        .status-chip {
            border: 1px solid var(--line);
            background: rgba(7,16,31,.62);
            border-radius: 999px;
            padding: 9px 12px;
            font-weight: 800;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-chip .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--green);
        }

        #threeCanvas {
            position: absolute;
            inset: 0;
        }

        #threeCanvas canvas {
            display: block;
            width: 100% !important;
            height: 100% !important;
        }

        .summary {
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .health-row {
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 18px;
            align-items: center;
        }

        .ring {
            width: 142px;
            aspect-ratio: 1;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: conic-gradient(var(--green) 0 87%, rgba(255,255,255,.08) 87% 100%);
            position: relative;
            box-shadow: inset 0 0 30px rgba(0,0,0,.22), 0 0 35px rgba(85,229,157,.12);
        }

        .ring::before {
            content: "";
            position: absolute;
            width: 108px;
            aspect-ratio: 1;
            border-radius: 50%;
            background: #0a1527;
            border: 1px solid rgba(255,255,255,.07);
        }

        .ring-content {
            position: relative;
            text-align: center;
        }

        .ring-value {
            font-size: 32px;
            font-weight: 900;
            line-height: 1;
        }

        .ring-label {
            color: var(--muted);
            font-size: 10px;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .summary h2 {
            margin: 0 0 5px;
            font-size: 24px;
        }

        .muted {
            color: var(--muted);
        }

        .prediction-box {
            margin-top: 12px;
            background: rgba(255,255,255,.035);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px;
        }

        .prediction-box b {
            display: block;
            margin-top: 5px;
            font-size: 16px;
        }

        .progress {
            height: 8px;
            background: rgba(255,255,255,.07);
            border-radius: 99px;
            margin-top: 10px;
            overflow: hidden;
        }

        .progress > div {
            height: 100%;
            width: 0%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--cyan), var(--blue));
            transition: width .5s ease;
        }

        .model-meta {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 9px;
        }

        .mini {
            padding: 10px;
            background: rgba(255,255,255,.035);
            border: 1px solid var(--line);
            border-radius: 14px;
        }

        .mini span {
            display: block;
            color: var(--muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .8px;
        }

        .mini strong {
            display: block;
            margin-top: 5px;
            font-size: 15px;
        }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 18px 0 11px;
        }

        .section-title h3 {
            margin: 0;
            font-size: 17px;
        }

        .section-title span {
            color: var(--muted);
            font-size: 12px;
        }

        .sensors {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }

        .sensor {
            padding: 14px;
            min-height: 118px;
            position: relative;
            overflow: hidden;
        }

        .sensor::after {
            content: "";
            position: absolute;
            width: 72px;
            height: 72px;
            right: -25px;
            bottom: -30px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(72,216,255,.16), transparent 68%);
        }

        .sensor-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            color: var(--muted);
            font-size: 11px;
        }

        .sensor-icon {
            width: 28px;
            height: 28px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            color: var(--cyan);
            background: rgba(72,216,255,.08);
            border: 1px solid rgba(72,216,255,.15);
        }

        .sensor-value {
            margin-top: 13px;
            font-weight: 900;
            font-size: 23px;
            letter-spacing: -.5px;
        }

        .sensor-value small {
            font-size: 11px;
            color: var(--muted);
            font-weight: 700;
            margin-left: 4px;
        }

        .sensor-sub {
            margin-top: 5px;
            color: var(--muted);
            font-size: 10px;
        }

        .lower-grid {
            display: grid;
            grid-template-columns: 1.28fr .72fr;
            gap: 16px;
            margin-top: 16px;
        }

        .chart-card,
        .insight-card,
        .controls-card {
            padding: 17px;
        }

        .chart-wrap {
            height: 250px;
            margin-top: 8px;
        }

        .insight-list {
            display: grid;
            gap: 9px;
            margin-top: 10px;
        }

        .insight {
            padding: 11px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,.035);
            border: 1px solid var(--line);
            font-size: 12px;
            color: #d9e3f3;
            line-height: 1.5;
        }

        .rul {
            margin-top: 12px;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 12px;
            padding: 13px;
            border-radius: 15px;
            background: linear-gradient(135deg, rgba(72,216,255,.08), rgba(109,124,255,.08));
            border: 1px solid rgba(72,216,255,.15);
        }

        .rul strong {
            font-size: 26px;
        }

        .rul span {
            font-size: 11px;
            color: var(--muted);
        }

        .controls-card {
            margin-top: 16px;
        }

        .control-row {
            display: flex;
            gap: 9px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        select, button {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 11px 13px;
            background: rgba(255,255,255,.04);
            color: var(--text);
            font: inherit;
            outline: none;
        }

        select {
            flex: 1 1 230px;
        }

        select option {
            background: #0b172a;
            color: white;
        }

        button {
            cursor: pointer;
            font-weight: 800;
            background: linear-gradient(135deg, rgba(72,216,255,.19), rgba(109,124,255,.22));
        }

        button:hover {
            border-color: rgba(72,216,255,.35);
        }

        .footer {
            margin-top: 18px;
            text-align: center;
            color: #6f819d;
            font-size: 11px;
        }

        .healthy { color: var(--green) !important; }
        .warning { color: var(--amber) !important; }
        .critical { color: var(--red) !important; }

        @media (max-width: 1100px) {
            .grid,
            .lower-grid {
                grid-template-columns: 1fr;
            }

            .sensors {
                grid-template-columns: repeat(3, 1fr);
            }

            .hero {
                min-height: 360px;
            }
        }

        @media (max-width: 720px) {
            .shell {
                width: min(100% - 18px, 1500px);
                padding-top: 12px;
            }

            .topbar {
                align-items: flex-start;
            }

            .brand-icon {
                width: 42px;
                height: 42px;
                border-radius: 13px;
            }

            .brand p {
                max-width: 245px;
                font-size: 11px;
            }

            .connection {
                padding: 8px 10px;
                font-size: 0;
            }

            .hero {
                min-height: 310px;
                border-radius: 18px;
            }

            .summary {
                padding: 15px;
            }

            .health-row {
                grid-template-columns: 112px 1fr;
                gap: 12px;
            }

            .ring {
                width: 110px;
            }

            .ring::before {
                width: 84px;
            }

            .ring-value {
                font-size: 25px;
            }

            .summary h2 {
                font-size: 20px;
            }

            .model-meta {
                grid-template-columns: 1fr 1fr 1fr;
            }

            .mini strong {
                font-size: 12px;
            }

            .sensors {
                grid-template-columns: repeat(2, 1fr);
                gap: 9px;
            }

            .sensor {
                min-height: 106px;
                padding: 12px;
                border-radius: 17px;
            }

            .sensor-value {
                font-size: 20px;
            }

            .chart-wrap {
                height: 220px;
            }
        }

        @media (max-width: 430px) {
            .sensors {
                grid-template-columns: 1fr 1fr;
            }

            .hero-overlay {
                padding: 14px;
            }

            .hero-title {
                font-size: 15px;
            }

            .status-chip {
                padding: 7px 9px;
                font-size: 10px;
            }

            .health-row {
                grid-template-columns: 1fr;
                text-align: center;
            }

            .ring {
                margin: 0 auto;
            }

            .prediction-box {
                text-align: left;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <header class="topbar">
            <div class="brand">
                <div class="brand-icon">AI</div>
                <div>
                    <h1>Predictive Vehicle Health Monitor</h1>
                    <p>Random Forest diagnostics • simulated hardware telemetry • predictive maintenance</p>
                </div>
            </div>
            <div class="connection">
                <span class="dot"></span>
                <span>LIVE SIMULATION</span>
            </div>
        </header>

        <section class="grid">
            <div class="card hero">
                <div class="hero-overlay">
                    <div>
                        <div class="eyebrow">Three.js Digital Twin</div>
                        <div class="hero-title">Vehicle Condition Visualization</div>
                    </div>
                    <div class="status-chip">
                        <span class="status-dot" id="statusDot"></span>
                        <span id="vehicleStatus">Loading...</span>
                    </div>
                </div>
                <div id="threeCanvas"></div>
            </div>

            <div class="card summary">
                <div class="section-title" style="margin:0;">
                    <h3>AI Health Summary</h3>
                    <span id="clock">--:--:--</span>
                </div>

                <div class="health-row">
                    <div class="ring" id="healthRing">
                        <div class="ring-content">
                            <div class="ring-value" id="healthScore">--%</div>
                            <div class="ring-label">Health Score</div>
                        </div>
                    </div>

                    <div>
                        <div class="eyebrow">Random Forest Prediction</div>
                        <h2 id="faultTitle">Waiting for data</h2>
                        <div class="muted" id="recommendation">Initializing simulated telemetry...</div>

                        <div class="prediction-box">
                            <span class="muted" style="font-size:11px;">Prediction confidence</span>
                            <b><span id="confidence">--</span>%</b>
                            <div class="progress">
                                <div id="confidenceBar"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="model-meta">
                    <div class="mini">
                        <span>Model</span>
                        <strong>Random Forest</strong>
                    </div>
                    <div class="mini">
                        <span>Accuracy</span>
                        <strong id="accuracy">--%</strong>
                    </div>
                    <div class="mini">
                        <span>Scenario</span>
                        <strong id="scenario">AUTO</strong>
                    </div>
                </div>
            </div>
        </section>

        <div class="section-title">
            <h3>Live Vehicle Sensor Telemetry</h3>
            <span>Random demo values • refresh every second</span>
        </div>

        <section class="sensors">
            <article class="card sensor">
                <div class="sensor-top"><span>ENGINE TEMP</span><span class="sensor-icon">T</span></div>
                <div class="sensor-value"><span id="engineTemp">--</span><small>°C</small></div>
                <div class="sensor-sub">Engine thermal condition</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>COOLANT LEVEL</span><span class="sensor-icon">C</span></div>
                <div class="sensor-value"><span id="coolant">--</span><small>%</small></div>
                <div class="sensor-sub">Reservoir level</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>BATTERY VOLTAGE</span><span class="sensor-icon">V</span></div>
                <div class="sensor-value"><span id="batteryVoltage">--</span><small>V</small></div>
                <div class="sensor-sub">Charging system voltage</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>BATTERY CURRENT</span><span class="sensor-icon">A</span></div>
                <div class="sensor-value"><span id="batteryCurrent">--</span><small>A</small></div>
                <div class="sensor-sub">Electrical load current</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>STEERING</span><span class="sensor-icon">S</span></div>
                <div class="sensor-value"><span id="steering">--</span><small>°</small></div>
                <div class="sensor-sub">Steering angle position</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>BRAKE PRESSURE</span><span class="sensor-icon">B</span></div>
                <div class="sensor-value"><span id="brake">--</span><small>%</small></div>
                <div class="sensor-sub">Brake pressure simulation</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>ENGINE VIBRATION</span><span class="sensor-icon">E</span></div>
                <div class="sensor-value"><span id="engineVib">--</span><small>g</small></div>
                <div class="sensor-sub">Engine vibration magnitude</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>BATTERY VIBRATION</span><span class="sensor-icon">M</span></div>
                <div class="sensor-value"><span id="batteryVib">--</span><small>g</small></div>
                <div class="sensor-sub">Battery mounting vibration</div>
            </article>

            <article class="card sensor">
                <div class="sensor-top"><span>GEAR POSITION</span><span class="sensor-icon">G</span></div>
                <div class="sensor-value"><span id="gear">--</span></div>
                <div class="sensor-sub">Current transmission position</div>
            </article>
        </section>

        <section class="lower-grid">
            <div class="card chart-card">
                <div class="section-title" style="margin:0;">
                    <h3>Live Health Trend</h3>
                    <span>30-second rolling window</span>
                </div>
                <div class="chart-wrap">
                    <canvas id="healthChart"></canvas>
                </div>
            </div>

            <div class="card insight-card">
                <div class="section-title" style="margin:0;">
                    <h3>Predictive Maintenance</h3>
                    <span>AI insights</span>
                </div>

                <div class="rul">
                    <div>
                        <span>Estimated remaining useful life</span><br>
                        <strong id="rul">--</strong>
                    </div>
                    <span>hours<br>prototype estimate</span>
                </div>

                <div class="insight-list" id="insightList">
                    <div class="insight">Waiting for first prediction...</div>
                </div>
            </div>
        </section>

        <section class="card controls-card">
            <div class="section-title" style="margin:0;">
                <h3>Demo Fault Injection</h3>
                <span>Use AUTO for random hardware simulation</span>
            </div>

            <div class="control-row">
                <select id="modeSelect">
                    <option value="AUTO">AUTO — Random scenarios</option>
                    <option value="NORMAL">NORMAL — Healthy vehicle</option>
                    <option value="ENGINE_OVERHEAT">ENGINE OVERHEAT</option>
                    <option value="COOLANT_LOW">LOW COOLANT</option>
                    <option value="BATTERY_FAULT">BATTERY FAULT</option>
                    <option value="BRAKE_FAULT">BRAKE FAULT</option>
                    <option value="HIGH_VIBRATION">HIGH VIBRATION</option>
                </select>
                <button id="applyMode">Apply Scenario</button>
            </div>
        </section>

        <div class="footer">
            Prototype dashboard • synthetic data and AI model are for demonstration before real hardware integration
        </div>
    </main>

    <script>
        const $ = (id) => document.getElementById(id);

        // ====================================================
        // CHART.JS
        // ====================================================
        const chartCtx = $("healthChart").getContext("2d");
        const healthChart = new Chart(chartCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Health %",
                        data: [],
                        borderColor: "#55e59d",
                        backgroundColor: "rgba(85,229,157,.08)",
                        tension: .35,
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: "y",
                        fill: true,
                    },
                    {
                        label: "Engine °C",
                        data: [],
                        borderColor: "#48d8ff",
                        tension: .35,
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: "y",
                    },
                    {
                        label: "Battery V",
                        data: [],
                        borderColor: "#ffca5b",
                        tension: .35,
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: "y1",
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: "index" },
                plugins: {
                    legend: {
                        labels: {
                            color: "#91a1bb",
                            boxWidth: 10,
                            usePointStyle: true,
                            font: { size: 10 }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: "#6f819d", maxTicksLimit: 8, font: { size: 9 } },
                        grid: { color: "rgba(255,255,255,.04)" }
                    },
                    y: {
                        min: 0,
                        max: 140,
                        ticks: { color: "#6f819d", font: { size: 9 } },
                        grid: { color: "rgba(255,255,255,.04)" }
                    },
                    y1: {
                        position: "right",
                        min: 8,
                        max: 17,
                        ticks: { color: "#6f819d", font: { size: 9 } },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });

        function pushChart(data) {
            const maxPoints = 30;
            healthChart.data.labels.push(data.timestamp);
            healthChart.data.datasets[0].data.push(data.health_score);
            healthChart.data.datasets[1].data.push(data.engine_temp);
            healthChart.data.datasets[2].data.push(data.battery_voltage);

            if (healthChart.data.labels.length > maxPoints) {
                healthChart.data.labels.shift();
                healthChart.data.datasets.forEach(ds => ds.data.shift());
            }
            healthChart.update("none");
        }

        // ====================================================
        // THREE.JS DIGITAL-TWIN STYLE CAR
        // ====================================================
        const holder = $("threeCanvas");
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
        camera.position.set(5.8, 3.6, 7.3);
        camera.lookAt(0, 0.9, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        holder.appendChild(renderer.domElement);

        const ambient = new THREE.HemisphereLight(0x9edcff, 0x08111f, 2.1);
        scene.add(ambient);

        const keyLight = new THREE.DirectionalLight(0xffffff, 3.3);
        keyLight.position.set(4, 7, 5);
        scene.add(keyLight);

        const rimLight = new THREE.PointLight(0x4ddcff, 24, 12);
        rimLight.position.set(-4, 2, -1);
        scene.add(rimLight);

        const car = new THREE.Group();
        scene.add(car);

        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x1d5a86,
            metalness: .78,
            roughness: .25,
            emissive: 0x061a2a,
            emissiveIntensity: .75
        });

        const glassMaterial = new THREE.MeshStandardMaterial({
            color: 0x74dfff,
            metalness: .15,
            roughness: .15,
            transparent: true,
            opacity: .62
        });

        const darkMaterial = new THREE.MeshStandardMaterial({
            color: 0x0b1119,
            metalness: .25,
            roughness: .62
        });

        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0x55e59d,
            emissive: 0x55e59d,
            emissiveIntensity: 1.25,
            metalness: .2,
            roughness: .45
        });

        const body = new THREE.Mesh(new THREE.BoxGeometry(4.6, .85, 2.05), bodyMaterial);
        body.position.y = .8;
        car.add(body);

        const hood = new THREE.Mesh(new THREE.BoxGeometry(1.45, .35, 1.9), bodyMaterial);
        hood.position.set(1.55, 1.33, 0);
        car.add(hood);

        const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.15, 1.05, 1.72), glassMaterial);
        cabin.position.set(-.35, 1.58, 0);
        cabin.rotation.z = -0.03;
        car.add(cabin);

        const engineBlock = new THREE.Mesh(new THREE.BoxGeometry(.7, .5, 1.25), glowMaterial);
        engineBlock.position.set(1.48, 1.62, 0);
        car.add(engineBlock);

        const wheelGeometry = new THREE.CylinderGeometry(.52, .52, .44, 24);
        const wheelPositions = [
            [1.48, .48, 1.08],
            [-1.48, .48, 1.08],
            [1.48, .48, -1.08],
            [-1.48, .48, -1.08],
        ];

        const wheels = [];
        wheelPositions.forEach(p => {
            const wheel = new THREE.Mesh(wheelGeometry, darkMaterial);
            wheel.rotation.x = Math.PI / 2;
            wheel.position.set(...p);
            car.add(wheel);
            wheels.push(wheel);
        });

        const ground = new THREE.Mesh(
            new THREE.CircleGeometry(4.3, 64),
            new THREE.MeshBasicMaterial({
                color: 0x0b2841,
                transparent: true,
                opacity: .22
            })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -.05;
        scene.add(ground);

        const rings = [];
        for (let i = 0; i < 3; i++) {
            const geo = new THREE.RingGeometry(2.6 + i * .5, 2.63 + i * .5, 64);
            const mat = new THREE.MeshBasicMaterial({
                color: 0x48d8ff,
                transparent: true,
                opacity: .10 - i * .02,
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(geo, mat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = .02;
            scene.add(ring);
            rings.push(ring);
        }

        let currentHealth = 100;
        let currentVibration = 1;
        let currentSteering = 0;
        let statusColor = new THREE.Color(0x55e59d);

        function resizeThree() {
            const w = holder.clientWidth;
            const h = holder.clientHeight;
            if (!w || !h) return;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }

        window.addEventListener("resize", resizeThree);
        window.addEventListener("orientationchange", resizeThree);
        if (window.ResizeObserver) {
            new ResizeObserver(resizeThree).observe(holder);
        }
        resizeThree();

        function animateThree() {
            requestAnimationFrame(animateThree);

            const t = performance.now() * 0.001;
            const shake = Math.min(currentVibration / 70, .10);

            car.rotation.y = Math.sin(t * .28) * .17;
            car.position.y = Math.sin(t * 1.5) * .025 + Math.sin(t * 22) * shake;
            car.rotation.z = Math.sin(t * 18) * shake * .12;

            wheels.forEach((wheel, index) => {
                wheel.rotation.y += .013 + currentSteering * .00002;
                if (index === 0 || index === 2) {
                    wheel.rotation.z = THREE.MathUtils.degToRad(currentSteering * .18);
                }
            });

            engineBlock.material.color.lerp(statusColor, .08);
            engineBlock.material.emissive.lerp(statusColor, .08);

            rings.forEach((ring, i) => {
                ring.rotation.z += .0006 * (i + 1);
                ring.material.opacity = .04 + .04 * Math.sin(t * .8 + i);
            });

            camera.lookAt(0, 0.9 + car.position.y, 0);
            renderer.render(scene, camera);
        }

        animateThree();

        // ====================================================
        // DASHBOARD
        // ====================================================
        function conditionColor(condition) {
            if (condition === "Healthy") return "#55e59d";
            if (condition === "Warning") return "#ffca5b";
            return "#ff6474";
        }

        function setText(id, value) {
            const el = $(id);
            if (el) el.textContent = value;
        }

        function updateDashboard(d) {
            setText("clock", d.timestamp);
            setText("vehicleStatus", d.condition);
            setText("healthScore", `${d.health_score}%`);
            setText("faultTitle", d.fault);
            setText("recommendation", d.recommendation);
            setText("confidence", d.prediction_confidence);
            setText("accuracy", `${d.model_accuracy}%`);
            setText("scenario", d.simulated_scenario.replaceAll("_", " "));
            setText("rul", d.remaining_useful_life_hours);

            setText("engineTemp", d.engine_temp);
            setText("coolant", d.coolant_level);
            setText("batteryVoltage", d.battery_voltage);
            setText("batteryCurrent", d.battery_current);
            setText("steering", d.steering_angle);
            setText("brake", d.brake_pressure);
            setText("engineVib", d.engine_vibration);
            setText("batteryVib", d.battery_vibration);
            setText("gear", d.gear_position);

            $("confidenceBar").style.width = `${d.prediction_confidence}%`;

            const c = conditionColor(d.condition);
            $("statusDot").style.background = c;
            $("healthRing").style.background =
                `conic-gradient(${c} 0 ${d.health_score}%, rgba(255,255,255,.08) ${d.health_score}% 100%)`;

            const scoreEl = $("healthScore");
            scoreEl.className = "ring-value " +
                (d.condition === "Healthy" ? "healthy" : d.condition === "Warning" ? "warning" : "critical");

            currentHealth = d.health_score;
            currentVibration = d.engine_vibration + d.battery_vibration;
            currentSteering = d.steering_angle;
            statusColor = new THREE.Color(
                d.condition === "Healthy" ? 0x55e59d :
                d.condition === "Warning" ? 0xffca5b : 0xff6474
            );

            $("insightList").innerHTML = "";
            d.insights.forEach(text => {
                const item = document.createElement("div");
                item.className = "insight";
                item.textContent = text;
                $("insightList").appendChild(item);
            });

            pushChart(d);
        }

        async function fetchTelemetry() {
            try {
                const res = await fetch("/api/telemetry", { cache: "no-store" });
                if (!res.ok) throw new Error("Telemetry request failed");
                const data = await res.json();
                updateDashboard(data);
            } catch (err) {
                console.error(err);
                setText("vehicleStatus", "Disconnected");
                $("statusDot").style.background = "#ff6474";
            }
        }

        $("applyMode").addEventListener("click", async () => {
            const mode = $("modeSelect").value;
            $("applyMode").disabled = true;
            $("applyMode").textContent = "Applying...";

            try {
                await fetch("/api/simulator", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ mode })
                });
                await fetchTelemetry();
            } finally {
                $("applyMode").disabled = false;
                $("applyMode").textContent = "Apply Scenario";
            }
        });

        fetchTelemetry();
        setInterval(fetchTelemetry, 1000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 68)
    print("AI-Based Predictive Vehicle Health Monitoring System")
    print(f"Random Forest synthetic test accuracy: {test_accuracy * 100:.2f}%")
    print("Open: http://127.0.0.1:5000")
    print("=" * 68)
    app.run(host="0.0.0.0", port=5000, debug=True)
