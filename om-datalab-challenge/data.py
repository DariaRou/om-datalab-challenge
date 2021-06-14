
import pandas as pd
import numpy as np
import os
import spacy
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
stop = set(stopwords.words('french'))
nlp=spacy.load("fr_core_news_md")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def om_get_data(n):
    # define file path, data stored in "/om-datalab-challenge/om-datalab-challenge/data"
    filepath = 'data/french_tweets.csv'
    # get a sample data in a dataFrame
    french = pd.read_csv(filepath).sample(n).reset_index(drop=True)
    
    return df

# Preprocessing data 
def om_clean_text(df):
    
    #replace emojis:
    emojis=demoji.findall(text)
    if emojis != {}:
        for key,value in emojis.items(): 
            if key in text:
                try:
                    translated_text=ts.translate_html(value, translator=ts.google, to_language='fr', n_jobs=-1)
                    text=text.replace(key,translated_text)
                except TypeError:
                        pass
    
    # lower text
    text = text.lower()
    # remove puncutation
    for punctuation in string.punctuation.replace('#',''):
        text = text.replace(punctuation, ' ')
    # remove words that contain numbers
    text = ''.join(letter for letter in text if not letter.isdigit())
    #tokenization + remove stop words
    doc=nlp(text)
    lemmatized= [token.lemma_ for token in doc]
    # join all
    text = " ".join(lemmatized)
    return text


if __name__ == '__main__':
    n= 100
    df = om_get_data(n)
