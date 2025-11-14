from flask import Flask, render_template, request
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import random
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audio'
app.config['GRAPH_FOLDER'] = 'static/graphs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create folders if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)


# Function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


# Dummy prediction function (replace with ML model later)
def predict_from_features(features):
    emotions = ["Happy", "Sad", "Angry", "Neutral", "Fearful", "Surprised"]
    genders = ["Male", "Female"]
    energy_levels = ["Low", "Medium", "High"]
    stress_levels = ["Relaxed", "Moderate", "Stressed"]

    return {
        "emotion": random.choice(emotions),
        "gender": random.choice(genders),
        "energy": random.choice(energy_levels),
        "stress": random.choice(stress_levels)
    }


# Function to create bar charts
def create_graphs(predictions):
    graph_paths = {}

    # Emotion Graph
    emotions = ["Happy", "Sad", "Angry", "Neutral", "Fearful", "Surprised"]
    values = [1 if e == predictions["emotion"] else 0 for e in emotions]
    plt.bar(emotions, values, color='skyblue')
    plt.title("Emotion Prediction")
    path = os.path.join(app.config['GRAPH_FOLDER'], "emotion.png")
    plt.savefig(path)
    plt.close()
    graph_paths["emotion"] = "graphs/emotion.png"

    # Gender Graph
    genders = ["Male", "Female"]
    values = [1 if g == predictions["gender"] else 0 for g in genders]
    plt.bar(genders, values, color='orange')
    plt.title("Gender Prediction")
    path = os.path.join(app.config['GRAPH_FOLDER'], "gender.png")
    plt.savefig(path)
    plt.close()
    graph_paths["gender"] = "graphs/gender.png"

    # Energy Graph
    energy = ["Low", "Medium", "High"]
    values = [1 if e == predictions["energy"] else 0 for e in energy]
    plt.bar(energy, values, color='green')
    plt.title("Energy Level")
    path = os.path.join(app.config['GRAPH_FOLDER'], "energy.png")
    plt.savefig(path)
    plt.close()
    graph_paths["energy"] = "graphs/energy.png"

    # Stress Graph
    stress = ["Relaxed", "Moderate", "Stressed"]
    values = [1 if s == predictions["stress"] else 0 for s in stress]
    plt.bar(stress, values, color='red')
    plt.title("Stress Level")
    path = os.path.join(app.config['GRAPH_FOLDER'], "stress.png")
    plt.savefig(path)
    plt.close()
    graph_paths["stress"] = "graphs/stress.png"

    return graph_paths


@app.route("/", methods=["GET", "POST"])
def index():
    predictions, graphs = None, None
    if request.method == "POST":
        file = None

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
        elif "recorded_data" in request.files and request.files["recorded_data"].filename != "":
            file = request.files["recorded_data"]

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            features = extract_features(file_path)
            predictions = predict_from_features(features)
            graphs = create_graphs(predictions)

    return render_template("index.html", predictions=predictions, graphs=graphs)


if __name__ == "__main__":
    app.run(debug=True)
