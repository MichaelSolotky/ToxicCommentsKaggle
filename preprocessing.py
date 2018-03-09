from nltk import WordNetLemmatizer, stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob


def raw_comment_to_tokens_list(comment):
    lemmatizer = WordNetLemmatizer()
    tokens = TextBlob(train['comment_text'][0]).tags
    stop = set(stopwords.words('english'))
    new_comment = []
    for token in tokens:
        normalized = lemmatizer.lemmatize(token[0])
        if normalized not in stop:
            new_comment.append((normalized, token[1]))
    return new_comment