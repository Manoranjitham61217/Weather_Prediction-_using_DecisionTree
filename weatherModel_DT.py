from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("weather_pipeline_model.pkl")
encoders = joblib.load("weather_label_encoders.pkl")

@app.route("/", methods=["GET"])
def home():
    return {"message": "Weather Prediction API is Running!"}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([data])
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)[0]

    return jsonify({
        "input_data": data,
        "prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)
