# 🫀 Cardiovascular Disease Risk Assessment Web App

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-powered web app for assessing cardiovascular risk based on health metrics. Built with Flask, trained in Jupyter, and enhanced with Google Gemini for personalized AI insights.

**🔗 Live Demo:** [`Youtube Link - Demo`](https://www.youtube.com/watch?v=KYPaH3m38Wg)

**📓 Notebook:** [`Cardio_Analysis.ipynb`](Cardio_Analysis.ipynb)

![Screenshot1](https://github.com/user-attachments/assets/e92b7543-566c-461b-8130-1d19760b58e9)
![Screenshot2](https://github.com/user-attachments/assets/a092c689-b14a-4781-a7f6-1a4d9a054503)



---

## 🚀 Features

### 🧠 ML-Powered Risk Prediction
- Data cleaning, preprocessing, and EDA with visualizations
- Feature engineering (e.g., BMI, pulse pressure)
- Model training with cross-validation (KNN, Random Forest, Logistic Regression)
- Best model saved via `joblib`

### 🌐 Web Interface (Flask)
- Intuitive input form for key health parameters
- Loads pre-trained ML model for risk scoring
- Uses **Google Gemini API** to:
  - Identify likely contributing factors
  - Suggest lifestyle improvements (with a typing effect)
- Responsive layout with a compact two-column design

### ⚠️ Educational Use Only
- Includes a built-in disclaimer: not a substitute for medical advice

---

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **ML & Data:** Scikit-learn, Pandas, NumPy
- **AI Integration:** Google Gemini API
- **Frontend:** HTML, CSS, JavaScript
- **Visualization (Notebook):** Matplotlib, Seaborn
- **Model Persistence:** Joblib

---

## 📁 Project Structure

```
cardio_risk_app/
├── app.py                        # Flask app logic
├── cardiovascular_risk_*.joblib # Trained ML model
├── Cardio_Analysis.ipynb         # Jupyter Notebook for model training
├── requirements.txt              # Dependencies
├── templates/
│   ├── index.html                # Input form
│   └── results.html              # Risk results page
├── static/
│   ├── style.css                 # App styling
│   └── gemini_logo.png           # Gemini branding
├── screenshot.png                # Optional UI screenshot
└── .gitignore                    # Git exclusions
```

---

## ⚙️ Setup & Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Activate:
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train or load the ML model:**
   - Run `Cardio_Analysis.ipynb` to generate `cardiovascular_risk_*.joblib`, or
   - Use the pre-trained model if already provided

5. **Set the Gemini API key (do not hardcode it):**
   ```bash
   # Windows (cmd)
   set GEMINI_API_KEY=your_key_here

   # PowerShell
   $env:GEMINI_API_KEY='your_key_here'

   # macOS/Linux
   export GEMINI_API_KEY='your_key_here'
   ```

---

## ▶️ Running the App

Make sure your API key is set, then run:

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 💡 How to Use

1. Fill in all the health inputs on the form.
2. Click **“Assess My Risk”**.
3. View:
   - Estimated risk score
   - AI-generated contributing factors
   - Suggested lifestyle adjustments (displayed word-by-word)
4. Read the disclaimer.
5. Optionally, click **“Assess Again”** to restart.

---

## 🤝 Contributing

Contributions are welcome!  
To contribute:

1. Fork this repo  
2. Create a feature branch: `git checkout -b feature/YourFeature`  
3. Commit: `git commit -m 'Add your feature'`  
4. Push: `git push origin feature/YourFeature`  
5. Open a Pull Request 🚀

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Cardiovascular dataset (e.g., from Kaggle)
- Flask Framework
- Scikit-learn
- Google Generative AI (Gemini)

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and **not** intended to diagnose or treat any medical condition. Always consult a healthcare professional for medical advice.

---

Let me know if you'd like a more minimalist version or want to tweak the tone/style!
