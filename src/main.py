from functions import spacy_tokenizer, get_titles_from_cluster, get_df_from_cluster
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from umap import UMAP
import numpy as np
from hdbscan import HDBSCAN
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


# # Importing CSV

print("Reading our data...")

df = pd.read_csv("titles_and_imdb-id.csv")
df = df.drop(["Unnamed: 0"], axis = 1)
reviews = df["First Review"]
reviews_list = list(reviews)


# # Sentimental Analysis from reviews (IMDb)

print("Let's see how Natural Language Processing can help us!")

list_of_results = []
    
for i in range(len(reviews_list)):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    scores = sentiment_analyzer.polarity_scores(reviews_list[i])
        
    list_of_results.append(scores)

# Let's create a new dataframe with the sentimental analysis information

sentiment_analyzer = pd.DataFrame.from_dict(list_of_results)
sentiment_analyzer = sentiment_analyzer.rename(columns={"compound": "Compound", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})


# # Concat both DataFrames

streaming_platforms = pd.concat([df, sentiment_analyzer], axis=1, join='inner')

#streaming_platforms.hist(column='Compound', weights=np.ones(len(streaming_platforms["Compound"])) / len(streaming_platforms["Compound"]))
#plt.xlabel('Compound for every reviews')
#plt.ylabel('Percentaje of movies')
#plt.title(r'Compound for every review (IMDb website)')
#plt.subplots_adjust(left=0.15) 
#plt.show()

# # Creating csv from my results
df.to_csv('titles_imdb-id_sentiment-analyzer.csv')


# # Now we are going to study positive reviews specifically...

positive_reviews = df.loc[df['Compound'] >= 0.75]


# # Reviews cleaning

nlp = spacy.load('en')
parser = English()

print(spacy_tokenizer(positive_reviews["First Review"][1])[:15])


# # TD-IDF

tfidf_vectorizer = TfidfVectorizer(min_df=0.10, tokenizer=spacy_tokenizer, ngram_range=(1,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(positive_reviews["First Review"])

terms = tfidf_vectorizer.get_feature_names()

print(terms[:30])

dist = 1 - cosine_similarity(tfidf_matrix)


# # Clustering 

# # UMAP

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    umap = UMAP(random_state=42)
    embedding = umap.fit_transform(dist)

print(embedding[:5])

# # plt.scatter(embedding[:,0], embedding[:,1])


# # HDBSCAN

hdbscan = HDBSCAN(min_cluster_size=8)

clustering = hdbscan.fit_predict(embedding)

# # Three clusters !

# # plt.scatter(embedding[:,0], embedding[:,1], c=clustering);

# # Titles from the first cluster (Fun)

titles_cluster = get_titles_from_cluster(0)


# # Titles from the second cluster (Fan)

titles_cluster = get_titles_from_cluster(1)


# # Titles from the third cluster (cinemagoer)

titles_cluster = get_titles_from_cluster(2)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)

# # Reviews from the third cluster (Cinemagoer)

print(positive_reviews["First Review"][6])

print(positive_reviews["First Review"][10])

print(positive_reviews["First Review"][26])


# # Most common words on reviews from the third cluster (Cinemagoer)

top_words_cluster = get_df_from_cluster(2).T.sum(axis=1).sort_values(ascending=False)
keywords_cluster2 = top_words_cluster.keys()
print(keywords_cluster2)

#unique_string=(" ").join(keywords_cluster2)
#wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
#plt.figure(figsize=(15,8))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.savefig("top_words_cluster_2"+".png", bbox_inches='tight')
#plt.show()
#plt.close()


# # Reviews from the second cluster (Fan)

print(positive_reviews["First Review"][1])

print(positive_reviews["First Review"][11])

print(positive_reviews["First Review"][109])


# # Most common words on reviews from the third cluster (Fan)

top_words_cluster = get_df_from_cluster(1).T.sum(axis=1).sort_values(ascending=False)
print(top_words_cluster)

keywords_cluster1 = top_words_cluster.keys()
print(keywords_cluster1)

#unique_string=(" ").join(keywords_cluster1)
#wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
#plt.figure(figsize=(15,8))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.savefig("top_words_cluster_1"+".png", bbox_inches='tight')
#plt.show()
#plt.close()


# # Reviews from the first cluster (Fun)

print(positive_reviews["First Review"][126])

print(positive_reviews["First Review"][76])


# # Most common words on reviews from the first cluster (Fun)

top_words_cluster = get_df_from_cluster(0).T.sum(axis=1).sort_values(ascending=False)
print(top_words_cluster)

keywords_cluster0 = top_words_cluster.keys()
print(keywords_cluster0)

#unique_string=(" ").join(keywords_cluster0)
#wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
#plt.figure(figsize=(15,8))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.savefig("top_words_cluster_0"+".png", bbox_inches='tight')
#plt.show()
#plt.close()
