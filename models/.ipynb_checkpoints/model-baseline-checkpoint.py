# Import custom helper libraries
import os, sys, re, csv, codecs
import pickle

# Maths modules
import numpy as np
import pandas as pd
from numpy import exp
from numpy.core.fromnumeric import repeat, shape  # noqa: F401,W0611
from scipy.stats import f_oneway

# Viz modules
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
%matplotlib inline

# Render for export
import plotly.io as pio
pio.renderers.default = "notebook"
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff

#Sklearn modules
from sklearn import metrics
from sklearn.metrics import (ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay,)
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, classification_report)
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# System modules
import random
import contractions
import re
import time
from collections import Counter
from collections import defaultdict
from unidecode import unidecode
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import gc
from random import shuffle
import itertools

# ML modules
from tqdm import tqdm
tqdm.pandas()

# NLTK modules
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Keras modules
import keras
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, BatchNormalization, TimeDistributed, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import Constant
# from keras.layers import (LSTM, Embedding, BatchNormalization, Dense, TimeDistributed, Dropout, Bidirectional, Flatten, GlobalMaxPool1D)
# from keras.optimizers import Adam

# Tensoflow modules
from tensorflow.keras.callbacks import EarlyStopping

# Gensim
import gensim.models.keyedvectors as word2vec

# Load data from CSV
df = pd.read_csv(r"C:\\Users\\ezequ\\proyectos\\openclassrooms\\Projet_7\\data\\raw\\sentiment140_16000_tweets.csv",
                 names=["target", "text"], encoding='latin-1')

# Drop useless raw
df = df.iloc[1: , :]

#TEXT PREPROCESSING
def text_cleaning(text, ponct, only_letters, numbers):
    text = text.lower()
    text = unidecode(text)
    ponctuation = "[^!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
    number = "[^0-9]"
    letters = "[^a-zA-Z ]"
    if ponct == 1:
        text = re.sub(ponctuation, '', text)
    if only_letters == 1:
        text = re.sub(letters, '', text)
    if numbers == 1:
        text = re.sub(number, '', text)
    return text

# Let's put the text in lower case.
df["new_text"] = df["text"].str.lower()

# Let's remove the punctuation.
df['new_text'] = df.progress_apply(lambda x: text_cleaning(x['text'], 0, 1, 0),axis=1)

# We can separate the text into word lists => each word unit is a tokens
df['words'] = df.progress_apply(lambda x: word_tokenize(x['new_text']),axis=1)

# Let's count the number of words per comment
df['nb_words'] = df.progress_apply(lambda x: len(x['words']),axis=1)

nltk.download('stopwords')
sw_nltk = stopwords.words('english')
keep_words = []
new_sw_nltk = [word for word in sw_nltk if word not in keep_words]
new_sw_nltk.extend(['th','pm', 's', 'er', 'paris', 'rst', 'st', 'am', 'us'])
pat = r'\b(?:{})\b'.format('|'.join(new_sw_nltk))
cleaning = df['new_text'].str.replace(pat, '')
df['new_words'] = cleaning.progress_apply(lambda x: nltk.word_tokenize(x))
df['new_text'] = cleaning

# The process of classifying words into their parts of speech and labeling 
# them accordingly is known as part-of-speech tagging, POS-tagging, or simply tagging. 

def word_pos_tagger(list_words):
    pos_tagged_text = nltk.pos_tag(list_words)
    return pos_tagged_text

all_reviews = df["new_text"].str.cat(sep=' ')
description_words = word_pos_tagger(nltk.word_tokenize(all_reviews))
list_keep = []
list_excl = ['IN', 'DT', 'CD', 'CC', 'RP', 'WDT', 'EX', 'MD', 'NNP', 'WDT', 'UH', 'WRB', 
'WP', 'WP$', 'PDT', 'PRP$', 'EX', 'POS', 'SYM', 'TO', 'NNPS']
for word, tag in description_words:
    if tag not in list_excl:
        list_keep.append(tag)
        
df["text_tokens_pos_tagged"] =  df["new_text"].progress_apply(lambda x: nltk.word_tokenize(x))
df["text_tokens_pos_tagged"] =  df["text_tokens_pos_tagged"].progress_apply(lambda x: nltk.pos_tag(x))

list_nouns = ["NN", "NNS"]
df["words_subjects"] =  df["text_tokens_pos_tagged"].progress_apply(lambda x: [y for y, tag in x if tag in list_nouns])

# The join() method takes all items in an iterable and joins them into one string.
df["words_subjects"] =  df["words_subjects"].progress_apply(lambda x: " ".join(x))

def lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return lemmatizer.lemmatize(text)

df["words_subjects_lem"] = df["words_subjects"].progress_apply(lambda x: lemmatize_text(x))

#label enconder
le = LabelEncoder()
le.fit(df['target'])
df['target_encoded'] = le.transform(df['target'])

# VECTORIZATION
# Processed data path
processed_data_path = os.path.join("..", "data", "processed")
vectorized_dataset_file_path = os.path.join(
    processed_data_path, "tfidf_spacy_dataset.pkl"
)
vocabulary_file_path = os.path.join(processed_data_path, "tfidf_spacy_vocabulary.pkl")

corpus = df["words_subjects_lem"]
# Define vectorizer
vectorizer = TfidfVectorizer()

# Vectorize text
X = vectorizer.fit_transform(corpus)

# Get vocabulary
vocabulary = vectorizer.get_feature_names()

# Train LSA model
n_components = 50
lsa = TruncatedSVD(n_components=n_components, random_state=42).fit(X)

# Reduce dimensionality
X_lsa = lsa.transform(X)

# Split data into train and test sets
# set aside 20% of train and test data for evaluation

X_train, X_test, y_train, y_test = train_test_split(X_lsa, df["target_encoded"], #df.target,
    test_size=0.2,  stratify=df.target,shuffle = shuffle, random_state = 42)

# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.1, random_state= 42) # 0.25 x 0.8 = 0.2

# Define model
model = LogisticRegressionCV(random_state=42)

# Train model
model.fit(X_train, y_train)

# Save the model to disk
pickle.dump(model, open('model.pkl', 'wb'))

# # Loading the model to compare results
# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([[2, 9, 6]]))
