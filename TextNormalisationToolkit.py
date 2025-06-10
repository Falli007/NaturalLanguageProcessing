#Text normalization transforms raw text into a consistent and standard format for NLP tasks. 
#It typically includes lowercasing, punctuation removal, stopword removal, stemming or lemmatization, and more.
#In this project, I create a customizable Python toolkit to normalize text—an essential preprocessing step for all language models and pipelines.

# Install if not already: pip install nltk
 
import re
import nltk   # Natural Language Toolkit for text processing
from nltk.corpus import stopwords   # For stopword removal
from nltk.stem import PorterStemmer, WordNetLemmatizer   # For stemming and lemmatization. Stemming reduces words to their root form, while lemmatization converts words to their base or dictionary form.
from nltk.tokenize import word_tokenize   # For tokenization, which splits text into individual words or tokens.
 
nltk.download('punkt')       # Download the Punkt tokenizer models. Punkt is a pre-trained tokenizer that can handle various languages and punctuation.
nltk.download('stopwords')   # Download the list of stopwords. Stopwords are common words that are often removed from text during preprocessing, such as "the", "is", "in", etc.
nltk.download('wordnet')     # Download the WordNet corpus. WordNet is a lexical database for the English language that provides synonyms, antonyms, and definitions of words.
 
# Initialize tools
stop_words = set(stopwords.words('english'))     # Set of English stopwords
stemmer = PorterStemmer()                        # Initialize the Porter Stemmer for stemming
lemmatizer = WordNetLemmatizer()                 # Initialize the WordNet Lemmatizer for lemmatization
 
# Normalization function
def normalize_text(text, use_stemming=False, use_lemmatization=True): # Function to normalize text
    # 1. Lowercase
    text = text.lower()                                    # Convert text to lowercase to ensure uniformity
    
    # 2. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)              # Remove all characters that are not lowercase letters or whitespace. This includes punctuation and numbers.
    
    # 3. Tokenize
    tokens = word_tokenize(text)                    # Split the text into individual words or tokens using the NLTK word tokenizer.
    
    # 4. Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]           # Filter out stopwords from the tokenized words. This step removes common words that do not contribute much meaning to the text, such as "the", "is", "and", etc.
    
    # 5. Apply stemming or lemmatization
    if use_stemming:                                                        
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif use_lemmatization:
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        processed_tokens = filtered_tokens
 
    return " ".join(processed_tokens)
 
# Example input
sample_text = "Hello, world! This is a sample text for normalization. It includes punctuation, numbers like 123, and some stopwords."
 
# Normalize
print("🧠 Original Text:\n", sample_text)
normalized = normalize_text(sample_text)
print("\n✅ Normalized Text:\n", normalized)