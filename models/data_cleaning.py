import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Stopwords for cleaning
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

stop_words = set(stopwords.words('german'))


def remove_urls(text):
    return re.sub(r'(http|https|www)\S+', '', text)


def remove_phone_numbers(text):
    """ Only digit sequences with min. length of 10 will be removed.
    Important to avoid accidental address removal"""

    return re.sub(r'(\(?([\d \-\)\–\+\/\(]+){10,}\)?([ .\-–\/]?)([\d]+))', '', text)


def remove_emails(text):
    return re.sub(r'[\w\.-]+@[\w\.-]+', '', text)


def remove_characters(text):
    return re.sub(r'[^a-zA-Z0-9 \n\.]', '', text)


def to_lower(text):
    return text.lower()


def replace_umlaute(text):
    return text.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')


def to_word_tokens(text):
    return word_tokenize(text)


def remove_stopwords(text):
    pure_text = []
    for i in text:
        if i.lower().strip() not in stop_words and i.strip().lower().isalpha():
            pure_text.append(i.lower())
    return " ".join(pure_text)


def remove_empty_cells(text):
    text = text.replace('', np.nan, inplace=True)
    text = text.dropna(inplace=True)
    return text


def text_cleaning(text):
    text = remove_urls(text)
    text = remove_phone_numbers(text)
    text = remove_emails(text)
    text = to_lower(text)
    text = replace_umlaute(text)
    text = remove_characters(text)
    text = to_word_tokens(text)
    text = remove_stopwords(text)

    return text


def metrics_eval(y_test=None, y_pred=None, model=None):
    """ Custom function for model evaluation to make code more reusable """

    print('--------------------{m} Model--------------------'.format(m=model), '\n')

    print('--------------Confusion Matrix--------------', '\n', confusion_matrix(y_test, y_pred), '\n')

    print('--------------Classification Report--------------', '\n', classification_report(y_test, y_pred), '\n')

    print('--------------Total accuracy--------------', '\n', accuracy_score(y_test, y_pred), '\n')


