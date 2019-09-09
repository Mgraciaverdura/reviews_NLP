import pandas as pd
import csv
import requests
import numpy as np
import json
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from functions import getreviews, only_first_paragraph


# Data loading and pre-processing

print("Reading the first dataframe...")

hbo_original = pd.read_csv("Documents/hbo_content_definitivo.csv")

hbo = hbo_original.copy()

# Data cleaning the first dataframe

print("Let's see how Pandas deals with these!")

# Dropping useless information

hbo = hbo.drop(["Unnamed: 0", "alternate_titles", "artwork_208x117", "artwork_304x171", "artwork_448x252", "artwork_608x342", "container_show", "first_aired", "freebase", "id", "themoviedb", "tvdb", "tvrage", "wikipedia_id"], axis=1)

# Creating a new column about the streaming platform where we can find those movies

hbo['Streaming_platform']='HBO'

hbo["Title"] = hbo["title"]
hbo["IMDb_id"] = hbo["imdb_id"]

print("Running...")

hbo = hbo.drop(["title", "imdb_id"], axis=1)

print("Ok! The first dataframe it's ready !Let's see the second one!")

# Data loading and pre-processing

print("Let's see how Pandas deals with these!")

netflix_original = pd.read_csv("Documents/netflix_content_definitivo_1.csv")

netflix = netflix_original.copy()

# Data cleaning the second dataframe

# Dropping useless information

print("Running...")

netflix = netflix.drop(["Unnamed: 0", "alternate_titles", "artwork_208x117", "artwork_304x171", "artwork_448x252", "artwork_608x342", "container_show", "first_aired", "freebase", "id", "themoviedb", "tvdb", "tvrage", "wikipedia_id"], axis=1)

# Creating a new column about the streaming platform where we can find those movies

netflix['Streaming_platform']='Netflix'

netflix = netflix[['title', 'imdb_id', 'Streaming_platform']]

netflix["Title"] = netflix["title"]
netflix["IMDb_id"] = netflix["imdb_id"]

netflix = netflix.drop(["title", "imdb_id"], axis=1)

print("Ok! Our second dataframe is ready !")

# Concat the two dataframes

print("Now, let's concanate the two dataframes...")

df = hbo.append(netflix, ignore_index=True)

df["IMDb_id"] = df.IMDb_id.str.replace("tt","")

print("Running...")

df = df.dropna(subset=['IMDb_id'])
df.IMDb_id = df.IMDb_id.astype(int)

# Save our new dataframe into a csv

df.to_csv('titles_and_imdb-id.csv')

# Web scraping reviews for every movie in IMDb

prueba = getreviews("2575988")

print("Let's see if we can find a review from IMDb website...")
print("This is a review about Silicon Valley (2014)")
print(prueba)

list_all_reviews = df['IMDb_id'].map(getreviews)

print(len("The dataframe has",list_all_reviews, " rows..."))

# We create a new column with this information

df["First Review"] = list_all_reviews

# Dropping useless information

df = df[df["First Review"].notnull()]
df = df[df["First Review"] != "nan"]
df["First Review"] = only_first_paragraph(list_all_reviews)
df = df[df["First Review"] != 'error para borrar']

# Saving CSV

df.to_csv('titles_and_imdb-reviews.csv')

print("The End!")



