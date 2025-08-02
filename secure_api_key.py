from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import nltk
import logging
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Securely load API key (example, not used here)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask
app = Flask(__name__)

# Ensure NLTK punkt is downloaded (do this once, quietly)
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(
    filename="app.log",  # Logs saved to file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from flask import render_template

@app.route('/')
def home():
    return render_template('form.html')


@app.route('/keywords', methods=['POST'])
def extract_keywords():
    start_time = time.time()  # measure latency start

    try:
        # Log route hit
        logging.info("Route /keywords hit")

        data = request.get_json()

        # Edge case: Missing 'text'
        if not data or 'text' not in data:
            logging.error("Missing 'text' key in request")
            return jsonify({"status": "error", "message": "Missing 'text' key"}), 400

        text = data['text']

        # Edge case: Empty string
        if not text.strip():
            logging.warning("Empty text received")
            return jsonify({"status": "error", "message": "Empty text"}), 400

        # Normal keyword extraction
        words = nltk.word_tokenize(text)
        keywords = [word for word in words if word.isalpha() and len(word) > 3]

        # Log success
        logging.info(f"Successfully extracted {len(keywords)} keywords")

        # Measure latency
        latency = round(time.time() - start_time, 3)

        return jsonify({
            "status": "success",
            "keywords": keywords,
            "latency_seconds": latency
        })

    except Exception as e:
        logging.exception("Error occurred in /keywords route")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


