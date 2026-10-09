
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("fake_news_model.pkl")


FEATURES = [
    "source_credibility",
    "author_credibility",
    "clickbait_score",
    "sentiment_score",
    "article_length",
    "num_quotes",
    "num_images",
    "num_links",
    "social_shares",
    "verified_source"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running!",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide JSON input."
            }), 400

        # Check that all required features are provided
        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Arrange input features in the same order as training
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        # Make prediction
        prediction = int(model.predict(input_data)[0])

        # Convert numeric prediction into a readable label
        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({
            "prediction": prediction,
            "result": result
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "Invalid input values",
            "details": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

