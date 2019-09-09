# Find the best review from IMDb website

import pandas as pd
import csv
import requests
import numpy as np
import json
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from spacy.lang.en import English
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
from umap import UMAP

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

import numpy as np
from hdbscan import HDBSCAN

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

def getreviews(id):
    
    reviews_imdb = []
    
    testurl = "https://www.imdb.com/title/tt{}/reviews?ref_=tt_urv".format(id)
    driver = webdriver.Firefox()
    driver.get(testurl)
    soup = BeautifulSoup(driver.page_source, features="html.parser")
    content = soup.find_all('div', class_=['text','show-more__control'])
    list_content = [tag.get_text() for tag in content]
    reviews_imdb.append(list_content)

    driver.quit()
    
    return reviews_imdb

def only_first_paragraph(lista):
    print(len(lista))
    lista_primer_parrafo=[]
    
    for j in range(len(lista)):
        lista_primer_parrafo.append(lista[j][0][0])
        
    return lista_primer_parrafo

# Tokenizer (Natural Language Processing)

def spacy_tokenizer(sentence):
    tokens = parser(sentence)

    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()

        if lemma not in STOP_WORDS and re.search('^[a-zA-Z]+$', lemma):
            filtered_tokens.append(lemma)

    return filtered_tokens

# Getting titles from clusters

def get_titles_from_cluster(cluster):
    return pd.Series(df["Title"])[clustering==cluster]

def get_df_from_cluster(cluster):
    return tfidf_df[clustering==cluster]