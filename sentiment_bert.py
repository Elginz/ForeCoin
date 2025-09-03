"""
As shown in tests, FinBERT was shown to be effective in sentimental analysis of news articles. 

This file uses the FinBERT ProsusAI model to give a determined expectation of the stock.

It returns:
- prediction value (Positive = +1, Negative = -1, Neutral = 0)
- confidence score (0.0 - 1.0)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load FinBERT tokenizer and model
model_name = "ProsusAI/finbert" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Map labels to numerical values
label_to_score = {
    "positive": 1,
    "negative": -1,
    "neutral": 0
}

def predict_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    # Get label names
    labels = model.config.id2label
    predicted_class_id = torch.argmax(probs, dim=1).item()
    predicted_label = labels[predicted_class_id]
    confidence_score = probs[0][predicted_class_id].item()

    # Map label to sentiment score
    prediction_value = label_to_score[predicted_label]

    return prediction_value, confidence_score

# Example texts to be used 
if __name__ == "__main__":
    sample_text = "SEC Delays Decision on Trump-Linked Truth Social Bitcoin ETF Until September - CoinDesk."
    prediction, confidence = predict_sentiment(sample_text)
    print(f"Sentence: {sample_text}")
    print(f"Prediction: {prediction} ({'Positive' if prediction == 1 else 'Negative' if prediction == -1 else 'Neutral'})")
    print(f"Confidence: {confidence:.4f}")
