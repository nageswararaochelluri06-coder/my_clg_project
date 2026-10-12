import os
import sqlite3
import traceback
from functools import wraps
import pandas as pd
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
 
# Gemini (old but stable for Flask apps)
import google.generativeai as genai


# =========================================================
# Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "cardiovascular_risk_random_forest_pipeline.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_this_secret_key")


# =========================================================
# Auth + SQLite helpers
# =========================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def login_required(view_fn):
    @wraps(view_fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_fn(*args, **kwargs)

    return wrapper


init_db()


# =========================================================
# Gemini Setup
# =========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

gemini_configured = False
GEMINI_MODEL = "gemini-1.5-flash"


def configure_gemini():
    global gemini_configured

    if not GEMINI_API_KEY:
        print("Gemini API key missing. AI suggestions disabled.")
        gemini_configured = False
        return

    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # test call
        model = genai.GenerativeModel(GEMINI_MODEL)
        model.generate_content("ping")

        gemini_configured = True
        print(f"Gemini configured successfully using model: {GEMINI_MODEL}")

    except Exception as e:
        print("Gemini configuration failed:", e)
        traceback.print_exc()
        gemini_configured = False


configure_gemini()


# =========================================================
# Load ML Pipeline
# =========================================================
pipeline = None
expected_features = []

try:
    print(f"Loading model from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    final_numerical_cols = [
        "age_years", "height", "weight",
        "ap_hi", "ap_lo", "bmi", "pulse_pressure"
    ]

    final_categorical_cols = [
        "gender", "cholesterol", "gluc",
        "smoke", "alco", "active", "bp_category"
    ]

    expected_features = final_numerical_cols + final_categorical_cols

    print("Model loaded successfully")

except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()
    pipeline = None


# =========================================================
# Helper Functions
# =========================================================
def map_bp_to_category(ap_hi, ap_lo):
    try:
        ap_hi = float(ap_hi)
        ap_lo = float(ap_lo)

        if ap_hi < 120 and ap_lo < 80:
            return "Normal"

        elif 120 <= ap_hi < 130 and ap_lo < 80:
            return "Elevated"

        elif 130 <= ap_hi < 140 or 80 <= ap_lo < 90:
            return "Hypertension Stage 1"

        elif ap_hi >= 140 or ap_lo >= 90:
            return "Hypertension Stage 2+"

        return "Undefined"

    except:
        return "Undefined"


def parse_gemini_response(text):
    factors = []
    suggestions = []

    current = None

    for line in text.split("\n"):

        line = line.strip()

        if line.lower().startswith("contributing"):
            current = factors
            continue

        if line.lower().startswith("suggested"):
            current = suggestions
            continue

        if line.startswith(("*", "-", "•")):
            line = line[1:].strip()

        if current is not None and line:
            current.append(line)

    if not factors:
        factors = ["No specific risk factors identified by AI."]

    if not suggestions:
        suggestions = ["Maintain a healthy lifestyle and consult a healthcare professional."]

    return factors, suggestions


# =========================================================
# Gemini Suggestions
# =========================================================
def get_local_suggestions(input_data, risk_score):
    factors = []
    suggestions = []

    age = float(input_data.get("age_years", 0))
    bmi = float(input_data.get("bmi", 0))
    ap_hi = float(input_data.get("ap_hi", 0))
    ap_lo = float(input_data.get("ap_lo", 0))
    cholesterol = int(input_data.get("cholesterol", 1))
    gluc = int(input_data.get("gluc", 1))
    smoke = int(input_data.get("smoke", 0))
    alco = int(input_data.get("alco", 0))
    active = int(input_data.get("active", 1))
    bp_cat = (input_data.get("bp_category") or "").strip()
    pulse_pressure = float(input_data.get("pulse_pressure", ap_hi - ap_lo))

    # Contributing factors
    if age >= 55:
        factors.append(f"Age ({int(age)}) is associated with higher cardiovascular risk.")
    if bmi >= 30:
        factors.append(f"BMI ({bmi}) suggests obesity, which increases cardiovascular strain.")
    elif bmi >= 25:
        factors.append(f"BMI ({bmi}) suggests overweight, which can elevate risk over time.")

    if bp_cat:
        if "Hypertension" in bp_cat:
            factors.append(f"Blood pressure category is {bp_cat}, which is a major risk driver.")
        elif bp_cat == "Elevated":
            factors.append("Blood pressure is elevated, which can increase long-term risk.")
    else:
        if ap_hi >= 140 or ap_lo >= 90:
            factors.append("Blood pressure readings are in a hypertensive range.")

    if pulse_pressure >= 60:
        factors.append(f"Pulse pressure ({int(pulse_pressure)}) is high, which can indicate arterial stiffness.")

    if cholesterol >= 3:
        factors.append("Cholesterol is well above normal, increasing atherosclerosis risk.")
    elif cholesterol == 2:
        factors.append("Cholesterol is above normal, which can contribute to plaque buildup.")

    if gluc >= 3:
        factors.append("Glucose is well above normal, which is linked with vascular complications.")
    elif gluc == 2:
        factors.append("Glucose is above normal, which can worsen cardiovascular outcomes.")

    if smoke == 1:
        factors.append("Smoking significantly increases cardiovascular risk.")
    if alco == 1:
        factors.append("Alcohol intake can worsen blood pressure and metabolic health in some individuals.")
    if active == 0:
        factors.append("Low physical activity is associated with higher cardiovascular risk.")

    # Suggestions
    if "Hypertension" in bp_cat or ap_hi >= 140 or ap_lo >= 90:
        suggestions.append("Monitor blood pressure regularly and discuss a BP management plan with a clinician.")
        suggestions.append("Reduce sodium intake, prioritize whole foods, and limit ultra-processed foods.")
    elif bp_cat == "Elevated":
        suggestions.append("Improve blood pressure through regular exercise, better sleep, and reduced sodium.")

    if bmi >= 25:
        suggestions.append("Aim for gradual weight loss through a calorie-aware diet and consistent activity.")

    if cholesterol >= 2:
        suggestions.append("Increase fiber (oats, legumes, vegetables) and reduce saturated/trans fats.")

    if gluc >= 2:
        suggestions.append("Limit sugary drinks/refined carbs and prioritize balanced meals with protein and fiber.")

    if smoke == 1:
        suggestions.append("Consider a smoking cessation plan (counseling, nicotine replacement, or clinician support).")

    if active == 0:
        suggestions.append("Target at least 150 minutes/week of moderate activity (e.g., brisk walking).")

    if alco == 1:
        suggestions.append("Reduce alcohol intake and avoid binge drinking to support blood pressure and metabolism.")

    # Risk-score framing
    if risk_score >= 60:
        suggestions.insert(0, "Because your estimated risk is high, consider scheduling a medical check-up soon.")
    elif risk_score >= 30:
        suggestions.insert(0, "Because your estimated risk is moderate, lifestyle changes can meaningfully reduce risk.")
    else:
        suggestions.insert(0, "Your estimated risk is lower, but maintaining healthy habits helps long-term prevention.")

    if not factors:
        factors = ["No single strong driver identified from the provided inputs; overall risk may reflect combined effects."]
    if not suggestions:
        suggestions = ["Maintain a heart-healthy lifestyle and consult a healthcare professional for personalized advice."]

    # Keep output concise
    return factors[:8], suggestions[:8]


def get_gemini_suggestions(input_data, risk_score):

    if not gemini_configured:
        return get_local_suggestions(input_data, risk_score)

    prompt = f"""
Analyze cardiovascular risk based on the following patient data.

Age: {input_data['age_years']}
BMI: {input_data['bmi']}
Blood Pressure: {input_data['ap_hi']}/{input_data['ap_lo']}
Cholesterol level: {input_data['cholesterol']}
Glucose level: {input_data['gluc']}
Smoker: {input_data['smoke']}
Alcohol: {input_data['alco']}
Physically Active: {input_data['active']}

Predicted cardiovascular risk score: {risk_score}%

Return only:

Contributing Factors:
* ...

Suggested Improvements:
* ...
"""

    try:

        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(prompt)

        text = response.text

        return parse_gemini_response(text)

    except Exception as e:
        print("Gemini error:", e)
        traceback.print_exc()

        return (
            ["Unable to generate AI explanation."],
            ["Please consult a healthcare provider."]
        )


# =========================================================
# Routes
# =========================================================
@app.route("/")
@login_required
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():

    if pipeline is None:
        flash("Prediction model unavailable", "error")
        return redirect(url_for("home"))

    try:

        form = request.form

        input_data = {
            "age_years": int(form["age_years"]),
            "height": float(form["height"]),
            "weight": float(form["weight"]),
            "ap_hi": int(form["ap_hi"]),
            "ap_lo": int(form["ap_lo"]),
            "gender": int(form["gender"]),
            "cholesterol": int(form["cholesterol"]),
            "gluc": int(form["gluc"]),
            "smoke": int(form["smoke"]),
            "alco": int(form["alco"]),
            "active": int(form["active"]),
        }

        # feature engineering
        height_m = input_data["height"] / 100

        input_data["bmi"] = round(input_data["weight"] / (height_m ** 2), 2)

        input_data["pulse_pressure"] = input_data["ap_hi"] - input_data["ap_lo"]

        input_data["bp_category"] = map_bp_to_category(
            input_data["ap_hi"],
            input_data["ap_lo"]
        )

        input_df = pd.DataFrame([input_data])[expected_features]

        prediction = pipeline.predict_proba(input_df)[0][1]

        risk_score = round(prediction * 100, 2)

        factors, suggestions = get_gemini_suggestions(input_data, risk_score)

        return render_template(
            "results.html",
            risk_score=risk_score,
            contributing_factors=factors,
            suggested_improvements=suggestions
        )

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()

        flash("Error processing input", "error")

        return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("register.html")

    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""

    if not username or not email or not password:
        flash("All fields are required.", "error")
        return redirect(url_for("register"))

    if len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
        return redirect(url_for("register"))

    if "@" not in email or "." not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("register"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("register"))

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))

    password_hash = generate_password_hash(password)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        flash("Username or email already exists. Please use another.", "error")
        return redirect(url_for("register"))
    finally:
        conn.close()

    flash("Registration successful. Please log in.", "success")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("login.html")

    identifier = (request.form.get("identifier") or "").strip()
    password = request.form.get("password") or ""

    if not identifier or not password:
        flash("Please enter your username/email and password.", "error")
        return redirect(url_for("login"))

    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower()),
        ).fetchone()
    finally:
        conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid credentials.", "error")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    flash(f"Welcome, {user['username']}!", "success")
    return redirect(url_for("home"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)