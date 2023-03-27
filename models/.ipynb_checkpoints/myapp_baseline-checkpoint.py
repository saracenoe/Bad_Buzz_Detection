import streamlit as st
import pandas as pd
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import sys
import re
import streamlit as st
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st
st.write("""
# Predict Sentiment from Tweeters

An interactive Web app to perform Sentiment Analysis on Tweets, based on machine learning algorithm.

""")

#sid = SentimentIntensityAnalyzer()

#function that would clean the data
@st.cache
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw text), and 
    # the output is a single string (a preprocessed text)
    
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,features="html.parser").get_text() 
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    #puncuation
    text  = "".join([char for char in  letters_only if char not in string.punctuation])
    
    
    # 3. Convert to lower case, split into individual words
    words = text.lower().split()                             
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
    
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(meaningful_words )) 

# Lemmatization 
def lemmatize_text(meaningful_words):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return lemmatizer.lemmatize(meaningful_words)

st.write('FOR DEMO')
user_input = st.text_input("Write any tweet to check its sentiment","EXAMPLE : this is my pet project and i love it.")

#Loading model
model = pickle.load(open("model_LSTM.pkl", 'rb'))

#cleansing and NLP part
user_input = review_to_words(user_input)
user_input = lemmatize_text(user_input)

user_input = [user_input]

# Define vectorizer
vectorizer = TfidfVectorizer()

# Vectorize text
user_input = vectorizer.fit_transform(user_input)

# Prediction function
def sentiment_analysis(user_input):
    
    
    
    # changing the input_data to numpy array
    user_imput_as_numpy_array = np.asarray(user_input)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = user_imput_as_numpy_array.reshape(-1,1)
    
    
    prediction = model.predict(input_data_reshaped) 
    print(prediction)


#t = sid.polarity_scores(demo)
if st.button("Predict"):
    sentiment_analysis(user_input)
    # st.success(f'The predicted value is R$ {round(predicted_value[0], 2)}')
    
#     st.write("Neutral ,Positive and Negative value of text is :")
#     st.write(t['neu']   ,t['pos'],t['neg'])
#     if t['neu']>t['pos'] and t['neu'] > t['neg'] and t['neu']>0.85:
#         st.markdown('Text is classified as **Neutral**. :confused::neutral_face:')
#     elif t['pos'] > t['neg']:
#         st.markdown('Text is classified as **Positive**. :smiley:')
#         st.balloons()
#     elif t['neg'] > t['pos']:
#         st.markdown('Text is classified as **Negative**. :disappointed: ')
#     else:
#         st.markdown('Text is classified as **Neutral**. :confused::neutral_face:')

#     st.write(' ')
#     st.write(' ')