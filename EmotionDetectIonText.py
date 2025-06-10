# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load a pretrained emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)  #This model detects emotions in English text
 
# Sample texts
texts = [
    "I am so happy today!",
    "This is the saddest movie I've ever seen.",
    "I feel angry about the situation.",
    "Surprise! I didn't expect that.",
    "I am feeling a bit anxious about the exam tomorrow.",
    "It's a beautiful day, and I feel peaceful.",
    "I love spending time with my friends and family.",
    "I am frustrated with the traffic today."
]
 
print("🧠 Emotion Detection Results:\n")

# Iterate through each text and classify emotions
for text in texts:
    result = emotion_classifier(text)[0]  # Get top emotion scores
    top_emotion = max(result, key=lambda x: x['score'])
    print(f"💬 Text: {text}")
    print(f"🔍 Detected Emotion: {top_emotion['label']} (Confidence: {top_emotion['score']:.2f})\n")