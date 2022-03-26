# Sentiment Prediction on Coronavirus Tweets

This app uses state-of-the-art machine learning algorithm (i.e. Gradient Boosting Classifier or GBC) to predict sentiment outcome on coronavirus tweets. 

What's actually happened behind the scene of this sentiment prediction app? 
1. The program receives input text or tweets from the user. 
2. The program performs pre-processing (i.e. text cleaning) on the input text or tweets such as normalization, removing stopwords, word tokenization, removing short-words, and word lemmatization. 
3. After pre-processing, the program then re-joined the cleaned tokenized text into a string of sentences. 
5. Prior to prediction analysis, the program uses the TFIDF vectorize library to transform the pre-processed input text or tweets into feature vectors.
6. The program then uses the pre-trained Gradient Boosting Classifier algorithm to predict the sentiment based on the transformed feature vectors. 

- GBC achieved 86% accuracy on testing set. 

You may click on this link to view/use the app: https://share.streamlit.io/jeffwongqy/sentiment-prediction-on-coronavirus-tweet/main/app.py
