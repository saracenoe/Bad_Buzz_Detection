import streamlit as st
import pandas as pd
import pandas as pd
#import  tweepy
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import sys
import re
import streamlit as st
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras.preprocessing.text import Tokenizer
import gensim.models.keyedvectors as word2vec
import gc
import tensorflow as tf


st.write("""
# Predict Sentiment from Tweeters

An interactive Web app to perform Sentiment Analysis on Tweets, based on machine learning algorithm.

""")

#function that would clean the data
@st.cache
def review_to_words(raw_text):
    """
    Function to convert a raw review to a string of words
    The input is a single string (a raw text), and 
    the output is a single string (a preprocessed text)
    """
    # 1. Remove HTML
    raw_text = BeautifulSoup(raw_text,features="html.parser").get_text() 
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_text))
    
    # puncuation
    text  = "".join([char for char in letters_only if char not in string.punctuation])
    
    # 3. Convert to lower case
    words = text.lower().split() 
    
    # 4. Tokens
    # tokenizer_words = TweetTokenizer()
    # tokens_sentences = [tokenizer_words.tokenize(t) for t in nltk.sent_tokenize(words)]
    
    # 5. Remove stop words
    # In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    
    stops = set(stopwords.words("english")) 
    meaningful_words = [w for w in words if not w in stops]
    
    # 4. Tokens
    # word_tokenize(meaningful_words, language='english')
    
    # 6. Join the words back into one string separatSed by space, and return the result.
    return( " ".join(meaningful_words))
    
def tokenize_and_stem(text):

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    stemmer = SnowballStemmer(language='english')

    stems = [stemmer.stem(t) for t in filtered_tokens if len(t) > 0]

    return stems


def last_processing (user_input):
    list_sentences_train = user_input

    max_features = 20000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list_sentences_train)
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(list_sentences_train)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index)  + 1
    
    longest_train = max(list_sentences_train, key=lambda sentence: len(user_input))
    length_long_sentence = len(word_tokenize(longest_train))
    padded_sentences = pad_sequences(word_tokenizer.texts_to_sequences(list_sentences_train), length_long_sentence, padding='post')

    maxlen = 300
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    
    return X_t

def lemmatize_text(meaningful_words):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return lemmatizer.lemmatize(meaningful_words)

st.write('FOR DEMO')
user_input = st.text_input("Write any tweet to check its sentiment","EXAMPLE : this is my pet project and i love it.")

#cleansing and NLP part
user_input = review_to_words(user_input)
user_input = lemmatize_text(user_input)

user_input = [user_input]

user_input = last_processing (user_input)

#Load model
path = 'C://Users//ezequ//proyectos//openclassrooms//Projet_7//streamlit//model.h5'
loaded_model= tf.keras.models.load_model(path )

# Prediction function
def sentiment_analysis(user_input):
        
#     # changing the input_data to numpy array
#     user_imput_as_numpy_array = np.asarray(user_input)
    
#     # reshape the array as we are predicting for one instance
#     input_data_reshaped = user_imput_as_numpy_array.reshape(-1,1)
    
    
    predictions = loaded_model.predict(user_input) 
    return predictions


if st.button("Predict"):
    with st.spinner("Analyzing the text …"):
        prediction = sentiment_analysis(user_input)
        if prediction[0][0] >0.4: #== 1:
            st.success("Positive sentiment")# with {:.2f} confidence".format(prediction))
            st.balloons()
        elif prediction[0][0] <0.4: #== 0:
            st.error("Negative sentiment")# with {:.2f} confidence".format(1-prediction))
            st.balloons()
        else:
            st.warning("Not sure! Try to add some more words")

    
    