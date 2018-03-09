from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob


def preprocess(comment):
    """
        Return str with lemmatized words without stop words
    """
    lemmatizer = WordNetLemmatizer()
    tokens = TextBlob(comment).tokens
    stop = set(stopwords.words('english'))
    new_comment = ' '
    normalized_tokens = []
    for token in tokens:
        normalized = lemmatizer.lemmatize(token[0])
        if normalized not in stop:
            normalized_tokens.append(normalized)
    return ' '.join(normalized)
