from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("student_grade_model.pkl")
scaler = joblib.load("scaler.pkl")

# Full list of features used by the model
all_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
    'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'school_MS',
    'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T', 'Mjob_health', 'Mjob_other',
    'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other', 'Fjob_services',
    'Fjob_teacher', 'reason_home', 'reason_other', 'reason_reputation',
    'guardian_mother', 'guardian_other', 'schoolsup_yes', 'famsup_yes', 'paid_yes',
    'activities_yes', 'nursery_yes', 'higher_yes', 'internet_yes', 'romantic_yes'
]

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("form.html")

# Prediction from web form
@app.route("/predict_web", methods=["POST"])
def predict_web():
    try:
        # Get data from form
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        # Convert all columns to numeric
        df = df.apply(pd.to_numeric)

        # Fill missing features with default 0
        for f in all_features:
            if f not in df.columns:
                df[f] = 0

        # Reorder columns
        df = df[all_features]

        # Scale numeric features
        df_scaled = scaler.transform(df)

        # Predict grade
        pred_grade = model.predict(df_scaled)[0]

        # Risk category
        if pred_grade < 10:
            risk = "High"
        elif pred_grade < 15:
            risk = "Medium"
        else:
            risk = "Low"

        return render_template("result.html", grade=round(pred_grade, 2), risk=risk)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
