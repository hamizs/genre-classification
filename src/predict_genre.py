# src/predict_genre.py

import os
import sys
import joblib
import pandas as pd

# Make sure we can import features.py if run as a module
here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from features import extract_clip_features  # same function used for GTZAN

# Path to the saved model bundle (you created this from the notebook)
PROJECT_ROOT = os.path.dirname(here)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gtzan_best_model.joblib")


def load_model_bundle(path=MODEL_PATH):
    """Load the trained model and feature column order."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Run the 01_features_and_baselines notebook and save the model first."
        )
    bundle = joblib.load(path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    return model, feature_columns


def predict_genre(audio_path, sr=22050, n_mfcc=20, n_chroma=12):
    """
    Predict the genre of a single audio file.

    Returns:
        predicted_label (str),
        proba_dict (dict or None): {class -> probability}
    """
    model, feature_columns = load_model_bundle()

    # Extract features for this clip
    feats = extract_clip_features(
        audio_path,
        sr=sr,
        duration=None,   # use full file; you can clip if you want
        n_mfcc=n_mfcc,
        n_chroma=n_chroma,
    )

    # Put into DataFrame and align columns with training features
    row = pd.DataFrame([feats])

    missing = set(feature_columns) - set(row.columns)
    if missing:
        raise ValueError(f"Missing features in extracted clip: {missing}")

    row = row[feature_columns]
    X = row.values

    # Predict
    pred = model.predict(X)[0]

    proba_dict = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        proba_dict = dict(zip(model.classes_, probs))

    return pred, proba_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict music genre for an audio file using the trained GTZAN model."
    )
    parser.add_argument("audio_path", help="Path to an audio file (wav/mp3/etc.)")
    args = parser.parse_args()

    audio_path = args.audio_path

    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    print(f"Classifying: {audio_path}")
    try:
        genre, proba = predict_genre(audio_path)
    except Exception as e:
        print("ERROR during prediction:", e)
        sys.exit(1)

    print(f"\nPredicted genre: {genre}")
    if proba is not None:
        print("\nClass probabilities (sorted):")
        for g, p in sorted(proba.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {g:10s} : {p:.3f}")


if __name__ == "__main__":
    main()
