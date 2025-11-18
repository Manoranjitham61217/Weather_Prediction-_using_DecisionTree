from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("weather_pipeline_model.pkl")
encoders = joblib.load("weather_label_encoders.pkl")
@app.route('/home')
def home():
    return render_template("index.html")



@app.route("/", methods=["GET"])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        precipitation = float(data['precipitation'])
        cloud_cover = float(data['cloud_cover'])
        uv_index = float(data['uv_index'])
        season = float(data['season'])
        visibility = float(data['visibility'])
 
        final_input = [[
            temperature,
            humidity,
            precipitation,
            cloud_cover,
            uv_index,
            season,
            visibility
        ]]

        prediction = model.predict(final_input)[0]

        return jsonify({"prediction": str(prediction)})


    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
