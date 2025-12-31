#!/usr/bin/env python3
"""
SMS Spam Classifier CLI - 97.2% accurate model
Usage: python predict.py "your message here"
"""

import joblib
import sys
import os

# Load model
model_path = 'models/spam_classifier.pkl'
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Run notebooks/01_eda_and_preprocessing.ipynb first!")
    sys.exit(1)

model = joblib.load(model_path)

def predict_spam(text):
    """Predict if message is spam (1) or ham (0)"""
    pred = model.predict([text])[0]
    prob_spam = model.predict_proba([text])[0][1]
    return 'SPAM' if pred == 1 else 'HAM', prob_spam

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Free money click here!\"")
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    result, confidence = predict_spam(message)
    print(f"📧 '{message}'")
    print(f"🎯 Prediction: {result} (confidence: {confidence:.1%})")
