"""
Text Preprocessing Module for Emotion Recognition.
Provides clean, deterministic text normalization matching the training pipeline.
"""

import re
import string

# Standard English stopwords fallback list to ensure offline reliability
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

_stopwords = set(DEFAULT_STOPWORDS)

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        _stopwords = set(stopwords.words('english'))
except Exception:
    pass


def clean_text(text: str) -> str:
    """
    Cleans raw text input:
    1. Lowercases the text
    2. Removes URLs
    3. Removes HTML tags
    4. Removes emojis / non-ASCII characters
    5. Removes punctuation
    6. Removes digits
    7. Removes stopwords and extra whitespaces
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Remove emojis / non-ASCII
    text = ''.join(c for c in text if c.isascii())

    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 6. Remove digits
    text = ''.join(c for c in text if not c.isdigit())

    # 7. Tokenize and remove stopwords
    tokens = text.split()
    cleaned_tokens = [w for w in tokens if w not in _stopwords and len(w) > 1]

    return ' '.join(cleaned_tokens)
