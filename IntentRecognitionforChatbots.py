#Intent recognition is a core component of chatbots that classifies a user’s input into a predefined intent (e.g., greeting, order_status, cancel_reservation). 
# In this project, I'll demonstrate how to build a simple intent recognition model using Python's scikit-learn library.
# Install if not already: pip install scikit-learn
 
from sklearn.feature_extraction.text import TfidfVectorizer   # For text vectorization 
from sklearn.linear_model import LogisticRegression           # For classification using logistic regression
from sklearn.pipeline import Pipeline                         # For creating a machine learning pipeline
from sklearn.model_selection import train_test_split          # For splitting the dataset into training and testing sets
from sklearn.metrics import classification_report             # For evaluating the model's performance
 
#To sample training data (intent-labeled)
training_data = [
    ("hi there", "greeting"),
    ("hello", "greeting"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("track my order", "order_status"),
    ("where is my package", "order_status"),
    ("cancel my order", "cancel_order"),
    ("i want to cancel the reservation", "cancel_order"),
    ("what's the weather like", "weather"),
    ("is it going to rain today", "weather"),
]
 
#To split into features and labels
texts, labels = zip(*training_data)
 
#To Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
 
#To create a pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
 
#To train the model
model.fit(X_train, y_train)
 
#To evaluate the model
y_pred = model.predict(X_test)
print("🧠 Intent Recognition Results:\n")
print(classification_report(y_test, y_pred))
 
#To try with new inputs
test_inputs = [
    "hey there",
    "hello world",
    "i need to track my shipment",
    "i want to cancel my order",
    "goodbye for now"
]

# To predict intents for new inputs
print("🔍 Predicted Intents:")
for inp in test_inputs:
    intent = model.predict([inp])[0]
    print(f"💬 \"{inp}\" → Intent: {intent}")