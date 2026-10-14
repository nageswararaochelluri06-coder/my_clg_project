
import os
import sqlite3
import traceback
from functools import wraps

import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai


# =========================================================
# Configuration
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "cardiovascular_risk_random_forest_pipeline.joblib"
HF_REPO_ID = "nagesh1729/cardiovascular-risk-model"

DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_this_secret_key")


# =========================================================
# Database and Authentication
# =========================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
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
# Gemini Configuration
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
gemini_configured = False


def configure_gemini():
    global gemini_configured

    if not GEMINI_API_KEY:
        print("Gemini API key missing. Using local suggestions.")
        gemini_configured = False
        return

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        print("Gemini API configured.")
    except Exception as e:
        print("Gemini configuration failed:", e)
        gemini_configured = False


configure_gemini()


# =========================================================
# Load ML Pipeline from Hugging Face
# =========================================================

pipeline = None
expected_features = []

final_numerical_cols = [
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "bmi",
    "pulse_pressure",
]

final_categorical_cols = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bp_category",
]

expected_features = final_numerical_cols + final_categorical_cols

try:
    HF_TOKEN = os.environ.get("HF_TOKEN")

    print("Downloading model from Hugging Face...")

    MODEL_PATH = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN or None,
    )

    print(f"Loading model from: {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

except Exception as e:
    print("Error loading model:", repr(e))
    traceback.print_exc()
    pipeline = None


# =========================================================
# Blood Pressure Category
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

    except (ValueError, TypeError):
        return "Undefined"


# =========================================================
# Parse Gemini Response
# =========================================================

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
        suggestions = [
            "Maintain a healthy lifestyle and consult a healthcare professional."
        ]

    return factors, suggestions


# =========================================================
# Local Risk Suggestions
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
    pulse_pressure = float(
        input_data.get("pulse_pressure", ap_hi - ap_lo)
    )

    # Contributing factors
    if age >= 55:
        factors.append(
            f"Age ({int(age)}) is associated with higher cardiovascular risk."
        )

    if bmi >= 30:
        factors.append(
            f"BMI ({bmi}) is in the obesity range and may increase cardiovascular strain."
        )
    elif bmi >= 25:
        factors.append(
            f"BMI ({bmi}) is in the overweight range."
        )

    if "Hypertension" in bp_cat:
        factors.append(
            f"Blood pressure category is {bp_cat}."
        )
    elif bp_cat == "Elevated":
        factors.append(
            "Blood pressure is elevated and may increase long-term risk."
        )
    elif not bp_cat and (ap_hi >= 140 or ap_lo >= 90):
        factors.append(
            "Blood pressure readings are in a hypertensive range."
        )

    if pulse_pressure >= 60:
        factors.append(
            f"Pulse pressure ({int(pulse_pressure)}) is high."
        )

    if cholesterol >= 3:
        factors.append(
            "Cholesterol is well above normal."
        )
    elif cholesterol == 2:
        factors.append(
            "Cholesterol is above normal."
        )

    if gluc >= 3:
        factors.append(
            "Glucose is well above normal."
        )
    elif gluc == 2:
        factors.append(
            "Glucose is above normal."
        )

    if smoke == 1:
        factors.append("Smoking increases cardiovascular risk.")

    if alco == 1:
        factors.append(
            "Alcohol intake may affect blood pressure and metabolic health."
        )

    if active == 0:
        factors.append(
            "Low physical activity is associated with higher cardiovascular risk."
        )

    # Suggestions
    if "Hypertension" in bp_cat or ap_hi >= 140 or ap_lo >= 90:
        suggestions.append(
            "Monitor blood pressure regularly and discuss management with a clinician."
        )
        suggestions.append(
            "Consider reducing sodium intake and choosing minimally processed foods."
        )
    elif bp_cat == "Elevated":
        suggestions.append(
            "Discuss blood pressure monitoring and lifestyle changes with a clinician."
        )

    if bmi >= 25:
        suggestions.append(
            "Consider gradual, sustainable dietary and activity changes."
        )

    if cholesterol >= 2:
        suggestions.append(
            "Choose fiber-rich foods and discuss cholesterol management with a clinician."
        )

    if gluc >= 2:
        suggestions.append(
            "Limit sugary drinks and discuss glucose management with a clinician."
        )

    if smoke == 1:
        suggestions.append(
            "Consider support for stopping smoking."
        )

    if active == 0:
        suggestions.append(
            "If medically appropriate, gradually increase regular physical activity."
        )

    if alco == 1:
        suggestions.append(
            "Consider reducing alcohol intake."
        )

    # Risk-score framing
    if risk_score >= 60:
        suggestions.insert(
            0,
            "Consider arranging a medical review to discuss your results."
        )
    elif risk_score >= 30:
        suggestions.insert(
            0,
            "Discuss your results and risk-reduction options with a healthcare professional."
        )
    else:
        suggestions.insert(
            0,
            "Continue preventive health habits and routine medical check-ups."
        )

    if not factors:
        factors = [
            "No single strong contributing factor was identified from these inputs."
        ]

    if not suggestions:
        suggestions = [
            "Maintain healthy habits and consult a healthcare professional for personalized advice."
        ]

    return factors[:8], suggestions[:8]


# =========================================================
# Gemini Suggestions
# =========================================================

def get_gemini_suggestions(input_data, risk_score):
    if not gemini_configured:
        return get_local_suggestions(input_data, risk_score)

    prompt = f"""
Analyze cardiovascular risk factors based on the following patient data.

Age: {input_data['age_years']}
BMI: {input_data['bmi']}
Blood Pressure: {input_data['ap_hi']}/{input_data['ap_lo']}
Cholesterol level: {input_data['cholesterol']}
Glucose level: {input_data['gluc']}
Smoker: {input_data['smoke']}
Alcohol: {input_data['alco']}
Physically Active: {input_data['active']}

Model-estimated risk score: {risk_score}%

This is an educational prediction, not a medical diagnosis.
Do not claim certainty.

Return only:

Contributing Factors:
* ...

Suggested Improvements:
* ...
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        response_text = response.text

        if not response_text:
            raise ValueError("Gemini returned an empty response.")

        return parse_gemini_response(response_text)

    except Exception as e:
        print("Gemini error:", repr(e))
        traceback.print_exc()
        return get_local_suggestions(input_data, risk_score)


# =========================================================
# Home Page
# =========================================================

@app.route("/")
@login_required
def home():
    return render_template("index.html")


# =========================================================
# Cardiovascular Risk Prediction
# =========================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if pipeline is None:
        flash(
            "Prediction model is unavailable. Please try again later.",
            "error",
        )
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

        if input_data["height"] <= 0 or input_data["weight"] <= 0:
            flash("Height and weight must be greater than zero.", "error")
            return redirect(url_for("home"))

        if input_data["ap_hi"] <= 0 or input_data["ap_lo"] <= 0:
            flash("Blood pressure values must be positive.", "error")
            return redirect(url_for("home"))

        # Feature engineering
        height_m = input_data["height"] / 100

        input_data["bmi"] = round(
            input_data["weight"] / (height_m ** 2), 2
        )

        input_data["pulse_pressure"] = (
            input_data["ap_hi"] - input_data["ap_lo"]
        )

        input_data["bp_category"] = map_bp_to_category(
            input_data["ap_hi"],
            input_data["ap_lo"],
        )

        input_df = pd.DataFrame([input_data])[expected_features]

        probabilities = pipeline.predict_proba(input_df)[0]

        # Assumes the model's positive class is the second class.
        risk_score = round(float(probabilities[1]) * 100, 2)

        factors, suggestions = get_gemini_suggestions(
            input_data,
            risk_score,
        )

        return render_template(
            "results.html",
            risk_score=risk_score,
            contributing_factors=factors,
            suggested_improvements=suggestions,
        )

    except Exception as e:
        print("Prediction error:", repr(e))
        traceback.print_exc()

        flash("Error processing the prediction. Please check your inputs.", "error")
        return redirect(url_for("home"))


# =========================================================
# Registration
# =========================================================

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
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, email, password_hash),
        )
        conn.commit()

    except sqlite3.IntegrityError:
        flash(
            "Username or email already exists. Please use another.",
            "error",
        )
        return redirect(url_for("register"))

    except sqlite3.Error as e:
        print("Registration database error:", repr(e))
        traceback.print_exc()
        flash("Registration failed because of a database error.", "error")
        return redirect(url_for("register"))

    finally:
        conn.close()

    flash("Registration successful. Please log in.", "success")
    return redirect(url_for("login"))


# =========================================================
# Login
# =========================================================

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
            """
            SELECT id, username, email, password_hash
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identifier, identifier.lower()),
        ).fetchone()

    except sqlite3.Error as e:
        print("Login database error:", repr(e))
        traceback.print_exc()
        flash("Login failed because of a database error.", "error")
        return redirect(url_for("login"))

    finally:
        conn.close()

    if not user or not check_password_hash(
        user["password_hash"], password
    ):
        flash("Invalid username/email or password.", "error")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    flash(f"Welcome, {user['username']}!", "success")
    return redirect(url_for("home"))


# =========================================================
# Logout
# =========================================================

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
