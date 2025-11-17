from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("weather_pipeline_model.pkl")
encoders = joblib.load("weather_label_encoders.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Weather Prediction API is Running!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid input. Please send JSON data."}), 400

        input_df = pd.DataFrame([data])
        for col in input_df.columns:
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except Exception as e:
                    return jsonify({
                        "error": f"Value '{input_df[col].iloc[0]}' is not valid for column '{col}'.",
                        "valid_values": list(encoders[col].classes_)
                    }), 400
        prediction = model.predict(input_df)[0]

        return jsonify({
            "input_data": data,
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
