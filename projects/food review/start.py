# Add environment Packages paths to conda
import os, sys
# env_name = "food_review"
# sys.path.append(f"C:\\Environments\\{env_name}\\lib\\site-packages\\")

import pandas as pd
import numpy as np

# Text preprocessing packages
import nltk # Text libarary
# nltk.download('stopwords')
import string # Removing special characters {#, @, ...}
import re # Regex Package
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer
from gensim.utils import simple_preprocess  # Text ==> List of Tokens

# Text Embedding
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Saving Model
import pickle

# Visualization Packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.3)
import streamlit as st

@st.cache(suppress_st_warning=True)
def looding1():
    pickle_in = open('rf_model.pk', "rb")
    loaded_model = pickle.load(pickle_in)
    return loaded_model 

@st.cache(suppress_st_warning=True)
def looding2():
    pickle_inn = open('tfidf_vectorizer.pk', "rb")
    loaded_vect = pickle.load(pickle_inn)
    return loaded_vect

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
x=stopwords.words('english')
stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()
not_words={'no','not','nor',"don't", "aren't", "couldn't", "didn't", "doesn't","hadn't", "hasn't", "haven't",
 "isn't", "mightn't", "mustn't", "needn't", "shouldn't", "wasn't", "weren't" ,"won't", "wouldn't"}


stop_words=set(x).difference(not_words)
stop_words=list(stop_words)
punctuation=list(string.punctuation)


@st.cache(suppress_st_warning=True)
def cleaning(sentence):
    #st.write("Cache miss : expensive_computation(", sentence,")ran")
    word_list = nltk.word_tokenize(sentence.lower())
    word_list=set(word_list).difference(set(stop_words))
    word_list=set(word_list).difference(set(punctuation))
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w.lower(),"v") for w in list(word_list) if not w.isdigit()])
    return lemmatized_output.lower()
    
@st.cache(suppress_st_warning=True)
def raw_test(review, model, vectorizer):
    #st.write("Cache miss : expensive_computation(", review,",", model,",",vectorizer,")ran")
    review=cleaning(review)
    embedding =vectorizer.transform([review])
    prediction=model.predict(embedding)
    if prediction==1:
        return "Positive"
    else:
        return 'Negative'

	

def welcome():
    return 'welcome all'

# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title('Amazon Food Review')
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:green;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Amazon Food Review Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    your_review = st.text_input("Enter your review", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = raw_test(your_review,looding1(),looding2())
    st.success('The output is {}'.format(result))


if __name__ =="__main__":
    main()