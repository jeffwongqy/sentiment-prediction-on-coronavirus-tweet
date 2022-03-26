# import relevant libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek 
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
import pickle 

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


def text_subjectivity(text):
    text_ = TextBlob(text)
    return text_.sentiment.subjectivity

def text_polarity(text):
    text_ = TextBlob(text)
    return text_.sentiment.polarity

def sentiment_analysis_outcome(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"
    


###############################################################################
######################### LOADING DATA FILE ###################################
###############################################################################

# read the csv file
filepath = r"C:\Users\jeffr\Desktop\c19\coronavirus_tweets.csv"
coronavirus_df = pd.read_csv(filepath)

###############################################################################
########################## DATA INFORMATION ###################################
###############################################################################

# check the data information 
print(coronavirus_df.info())

# check for missing data 
print(coronavirus_df.isnull().sum())

# use heatmap to visualize the missing row data
sns.heatmap(coronavirus_df.isnull(), annot = False, cmap = 'coolwarm')

# remove the row with null value 
coronavirus_df.dropna(axis = 0, how = 'any', inplace = True)

# re-check for missing data
print(coronavirus_df.isnull().sum())

# use heatmap to visualize the missing row data
sns.heatmap(coronavirus_df.isnull(), annot = False, cmap = 'coolwarm')

###############################################################################
############################# NORMALIZATION ###################################
###############################################################################
# call the function to perform text cleaning 
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(normalization)

###############################################################################
############################# TOKENIZATION ####################################
###############################################################################
# call the function to split the text into a list of tokens
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(tokenization)

###############################################################################
############################# REMOVED STOPWORDS ###############################
###############################################################################
# call the function to removed stopwords
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(removedStopwords)

###############################################################################
#################### REMOVED WORDS LESS THAN 4 CHARACTERS #####################
###############################################################################
# call the function to removed words less than 4 characters
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(removed_words_less_than_4_characters)

###############################################################################
################################ LEMMATIZATION  ###############################
###############################################################################
# call the function to lemmatize the words 
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(wordsLemmatization)

###############################################################################
########################### SENTIMENT ANALYSIS  ###############################
###############################################################################
# call the function to combine each token into a string 
coronavirus_df['Tweet'] = coronavirus_df['Tweet'].apply(sentence_reconstruction)

# call the function to determine the subjectivity of each tweet
coronavirus_df_sujectivity = coronavirus_df['Tweet'].apply(text_subjectivity)
# create a new column in coronavirus dataframe for subjectivity
coronavirus_df['subjectivity'] = coronavirus_df_sujectivity

# call the function to determine the polarity of each tweet
coronavirus_df_polarity = coronavirus_df['Tweet'].apply(text_polarity)
# create a new column in coronavirus dataframe for polarity
coronavirus_df['polarity'] = coronavirus_df_polarity

# create a new column in coronavirus dataframe for sentiment analysis outcome 
coronavirus_df['sentiment_analysis'] = coronavirus_df_polarity.apply(sentiment_analysis_outcome)

# create a scatter plot to show the relationship between polarity vs subjectivity
sns.set_style("darkgrid")
sns.scatterplot(x = coronavirus_df_polarity, y = coronavirus_df_sujectivity)
plt.title("Sentiment Analysis of Polarity vs Subjectivity", fontweight = 'bold', fontsize = 15)
plt.xlabel("Polarity", fontweight = 'bold', fontsize = 12)
plt.ylabel("Subjectivity", fontweight = 'bold', fontsize = 12)
plt.show()

# create a bar chart to show the distribution of sentiment analysis outcome
sns.set_style("darkgrid")
coronavirus_df['sentiment_analysis'].value_counts().plot(kind = "bar", color = ["crimson", "navy", "chocolate"])
plt.title("Distribution of Sentiment Analysis Outcome", fontsize = 15, fontweight = 'bold')
plt.xlabel("Sentiment Analysis Outcome", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Observations", fontsize = 12, fontweight = 'bold')
plt.show()


###############################################################################
########################### TRAIN-TEST SPLITS  ################################
###############################################################################


X_train, X_test, y_train, y_test = train_test_split(coronavirus_df['Tweet'],
                                                    coronavirus_df['sentiment_analysis'],
                                                    test_size = 0.25, 
                                                    random_state = 42)


###############################################################################
############# MODEL BUILDING (BASELINE) & PERFORMANCE EVALUATION  #############
###############################################################################
# define model object listing
model_object_list = [MultinomialNB(), 
              RandomForestClassifier(n_estimators = 250,  max_depth = 8, random_state = 42), 
              GradientBoostingClassifier(n_estimators = 350, learning_rate = 0.1, max_depth = 10, random_state = 42), 
              DecisionTreeClassifier(max_depth = 4, random_state = 42),
              SVC(kernel = "linear", random_state = 42)]

# model name listing
model_name_list = ['Multinomial NB',
                   'Random Forest Classifier',
                   'Gradient Boosting Classifier', 
                   'Decision Tree Classifier', 
                   'SVC']

# model building
for i in range(len(model_object_list)):
    # create a pipeline steps
    steps = [('tfidfVectorizer', TfidfVectorizer(analyzer = 'word')), 
                 ('smt', SMOTETomek(random_state = 42)),
                 ('model',model_object_list[i])]
    # model pipeline
    model = Pipeline(steps = steps)
    # model fitting
    model.fit(X_train, y_train)
    
    # predict training set
    y_pred_train = model.predict(X_train)
    # predict testing set
    y_pred_test = model.predict(X_test)
    
    # display the classification report for respective model 
    print("Classification Report for",  model_name_list[i], "with Training Set: ")
    print(classification_report(y_train, y_pred_train))
    print()
    print("Classification Report for", model_name_list[i], "with Testing Set:")
    print(classification_report(y_test, y_pred_test))




###############################################################################
######################## REBUILD OF SELECTED ML (GBC)  ########################
###############################################################################
steps = [('tfidfVectorizer', TfidfVectorizer(analyzer = 'word')), 
             ('smt', SMOTETomek(random_state = 42)),
             ('gbc', GradientBoostingClassifier(n_estimators = 350, 
                                                learning_rate = 0.1, 
                                                max_depth = 10, 
                                                random_state = 42))]

# build gradient boosting model
model_gbc = Pipeline(steps = steps)

# fit the model
model_gbc.fit(X_train, y_train)



###############################################################################
######################## MODEL EVALUATION ON GBC  #############################
###############################################################################
# predict the values of training set and testing set
y_pred_train_gbc = model_gbc.predict(X_train)
y_pred_test_gbc = model_gbc.predict(X_test)

# display the classification report
print("Classification Report for Gradient Boosting Classifier with Training Set: ")
print(classification_report(y_train, y_pred_train_gbc))
print()
print("Classification Report for Gradient Boosting Classifier with Testing Set:")
print(classification_report(y_test, y_pred_test_gbc))

# display the confusion matrix
conf_matrix = confusion_matrix(y_train, y_pred_train_gbc, labels = ['Negative', 'Neutral', 'Positive'])
sns.heatmap(conf_matrix, annot = True, cmap = 'viridis')
plt.title("Confusion Matrix for Gradient Boosting Classifier (Training Set)", fontweight = 'bold', fontsize = 12)
plt.xlabel("Actual", fontweight = 'bold', fontsize = 12)
plt.ylabel("Predicted", fontweight = 'bold', fontsize = 12)

conf_matrix = confusion_matrix(y_test, y_pred_test_gbc, labels = ['Negative', 'Neutral', 'Positive'])
sns.heatmap(conf_matrix, annot = True, cmap = 'viridis')
plt.title("Confusion Matrix for Gradient Boosting Classifier (Testing Set)", fontweight = 'bold', fontsize = 12)
plt.xlabel("Actual", fontweight = 'bold', fontsize = 12)
plt.ylabel("Predicted", fontweight = 'bold', fontsize = 12)

# display the evaluation metrics
print("Performance Metrics for Gradient Boosting Classifier (Training Set):")
print("Accuracy Score: ", accuracy_score(y_train, y_pred_train_gbc))
print("Precision Score: ", precision_score(y_train, y_pred_train_gbc, average = 'weighted'))
print("F1 Score: ", f1_score(y_train, y_pred_train_gbc, average = "weighted"))
print("Recall Score: ", recall_score(y_train, y_pred_train_gbc, average = "weighted"))
print()
print("Performance Metrics for Gradient Boosting Classifier (Testing Set):")
print("Accuracy Score: ", accuracy_score(y_test, y_pred_test_gbc))
print("Precision Score: ", precision_score(y_test, y_pred_test_gbc, average = 'weighted'))
print("F1 Score: ", f1_score(y_test, y_pred_test_gbc, average = "weighted"))
print("Recall Score: ", recall_score(y_test, y_pred_test_gbc, average = "weighted"))

###############################################################################
################################### SAVE MODEL  ###############################
###############################################################################
filename = "model_gbc.sav"
pickle.dump(model_gbc, open(filename, 'wb'))

