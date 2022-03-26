import streamlit as st
import pickle
import time 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

###############################################################################
######################### IMPORT GBC MODEL ####################################
###############################################################################
modelGBC = pickle.load(open("model_gbc.sav", "rb"))

###############################################################################
######################### FUNCTION DECLARATION ################################
###############################################################################

def normalization(text):
    # transform text into lowercase
    text = text.lower()
    
    # remove the punctuation and special characters 
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # remove digits
    text = re.sub(r'[\d+]', '', text)
    
    # remove greeks
    text = (re.sub('(?![ -~]).', '', text))
    
    return text
    
def tokenization(text):
    # split the text into a series of tokens
    text = word_tokenize(text)
    
    return text

def removedStopwords(text):
    
    # initialize an empty list of tokens with no stopwords
    tokens_with_no_stopwords = list()
    
    # extract tokens not found in a list of stopwords 
    for word in text:
        if word not in stopwords.words('english'):
            tokens_with_no_stopwords.append(word)
    
    return tokens_with_no_stopwords
    
def removed_words_less_than_4_characters(text):
    # initialize an empty list for tokens with more than 4 characters
    tokens_with_more_than_4_characters = list()
    
    # extract tokens with more than 4 characters
    for word in text:
        if len(word) >= 4:
            tokens_with_more_than_4_characters.append(word)
            
    return tokens_with_more_than_4_characters
    
def wordsLemmatization(text):
    # define object for lemmatizer 
    lemmatizer = WordNetLemmatizer()
    
    # initialize an empty list for lemmatized words
    wordsLemmatize = list()
    
    # lemmatize the words
    for word in text:
        wordsLemmatize.append(lemmatizer.lemmatize(word))
    
    return wordsLemmatize

def sentence_reconstruction(text):
    # initialize an emppty string 
    sentence_reconstruction = ""
    
    # combine each token to form a string 
    for word in text:
        sentence_reconstruction = sentence_reconstruction + word + " "
    return sentence_reconstruction 

def sentiment_prediction(text):
    predict_sentiment = modelGBC.predict([text])
    return predict_sentiment

def main():
    
    ###############################################################################
    ################################### MAIN  #####################################
    ###############################################################################
    
    st.title("Sentiment Prediction on Coronavirus Tweets")
    st.markdown("**Using the state-of-the-art machine learning algorithm (Gradient Boosting Classifier) to predict sentiment outcome on coronavirus tweets.**")
    st.image("sentiment.png")
    # instructions to the user
    st.write("**Instructions:**")
    st.info("Enter a series of text or tweets in the text box and then click on the **Click Here to Predict** button to predict sentiment. ")
    # prompt the user to enter a text 
    text = st.text_area("Enter a series of text or tweets: ", value = "", placeholder = "Enter your coronavirus text/ tweets here (e.g. I love coronavirus.).")
    
    # prompt the user to click on the button to submit the text for analysis 
    submit_button = st.button("Click Here To Predict")
    
    if submit_button: 
        if text == "":
            st.error("Error! Input text or tweet cannot be empty.")
        
        else:
            
            # display the raw text
            st.markdown("**Raw Text:**")
            st.warning(text)
            
            with st.spinner("TEXT CLEANING IN PROGRESS!"):
                time.sleep(5)
            
            # call the function to normalized the text
            text = normalization(text)
            
            # call the function to tokenized the text
            text = tokenization(text)
            
            # call the function to removed stopwords
            text = removedStopwords(text)
            
            # call the function to removed words less than 4 characters
            text = removed_words_less_than_4_characters(text)
            
            # call the function to lemmatized the text
            text = wordsLemmatization(text)
            
            # call the function to rejoined the text after cleaning
            text = sentence_reconstruction(text)
            
            # display the cleaned text
            st.write("**Cleaned Text:** ")
            st.info(text)
            
            with st.spinner("SENTIMENT PREDICTION IN PROGRESS! "):
                time.sleep(5)
            
            # call the function to predict the sentiment text 
            predicted_sentiment = sentiment_prediction(text)
            
            st.markdown("**Prediction Outcome:** ")
            if predicted_sentiment == 'Positive':
                st.success("The sentiment is " + predicted_sentiment[0].lower() + ".")
            elif predicted_sentiment == 'Neutral':
                st.info("The sentiment is " + predicted_sentiment[0].lower() + ".")
            else:
                st.error("The sentiment is " + predicted_sentiment[0].lower() + ".")

if __name__ == '__main__':
    main()
