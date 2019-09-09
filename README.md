**Analyzing Movie reviews with Natural Processing Language (NLP)**


*Introduction*

We are going to analyze movie reviews on IMDb's website. 

My hypothesis is that firsts reviews (most voted ones) are from people that like very much those movies and don't have an interesting thesis.

I am going to use Natural Processing Language and I am going to focalize on movies that they are available on Netflix or HBO.

*Technologies*

Python 2.7 (for API GuideBox)
Python 3.6


*My conclusion*

The most part of the movies have the same characteristics: simple reviews and similar size.

Some outliers have some speficities like a special mention to an actor, or a more critical analysis of technical and thematic content.

*src*

- *data_cleaning.py* : Getting information and data cleaning.

I found information in GuideBox's API, IMDb's API and scrapying with Selenium IMDb's website.

- *main.py* : working with Natural Processing Language. I tried TfidfVectorizer (SKlearn), UMAP and HDBSCAN.

- *functions.py* : all the functions used for this project.

*notebooks*

Everything here was written in the Jupyter Notebook environment, previous to build the code at src.s

*documents*

All the dataframes in csv.

*top_word_clusters*

WordCloud for each Cluster.






