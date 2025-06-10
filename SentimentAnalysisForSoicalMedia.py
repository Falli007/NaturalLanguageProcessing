#Sentiment analysis on social media helps businesses and analysts understand public opinions in real time. 
#In this project, we analyze sentiment in short, informal text like tweets or posts using a pretrained sentiment model.
# I used VADER (Valence Aware Dictionary for Sentiment Reasoning), which is tailored for social media language.

# Install if not already: pip install vaderSentiment

import nltk        # Natural Language Toolkit for text processing
from nltk.sentiment import SentimentIntensityAnalyzer     # For sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner)
 
nltk.download('vader_lexicon')    # Download the VADER lexicon, which is a pre-built sentiment analysis model specifically designed for social media text.
 
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()     # Create an instance of the SentimentIntensityAnalyzer
 
# Example social media posts
tweets = [
    "I'm so happy with the new update! 🚀🔥",
    "Trump is the best president ever! 🇺🇸",
    "Elon Musk is ruining Twitter. 😡",
    "Brexit was a terrible decision for the UK. #BrexitFail",
    "This is the worst experience I've ever had. #fail",
    "Tech lay offs are hitting hard. 😢",
    "S&P500 hits record high! 📈💰",
]
 
print("🧠 Sentiment Analysis on Social Media Posts:\n")


for tweet in tweets:
    scores = sia.polarity_scores(tweet)
    sentiment = "positive" if scores['compound'] > 0.05 else "negative" if scores['compound'] < -0.05 else "neutral"
    
    print(f"💬 Tweet: {tweet}")
    print(f"🔎 Sentiment: {sentiment} (Score: {scores['compound']})\n")