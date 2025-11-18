from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("weather_pipeline_model.pkl")
encoders = joblib.load("weather_label_encoders.pkl")
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        precipitation = float(request.form['precipitation'])
        cloud_cover = float(request.form['cloud_cover'])
        uv_index = float(request.form['uv_index'])
        season = float(request.form['season'])
        visibility = float(request.form['visibility'])

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
        pred_word = encoders["weather_type"].inverse_transform([prediction])[0]

        return render_template("index.html", result=pred_word)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
