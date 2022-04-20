from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import random
import re
import string

stopwords = set(stopwords.words('english'))

mydata_train = fetch_20newsgroups(subset='train', shuffle=True)
mydata_test = fetch_20newsgroups(subset='test', shuffle=True)
mydata_train_df = pd.DataFrame({'data': mydata_train.data, 'target': mydata_train.target})
mydata_test_df = pd.DataFrame({'data': mydata_test.data, 'target': mydata_test.target})

tfidfV = TfidfVectorizer(stop_words=stopwords) 
X_train_tfidfV = tfidfV.fit_transform(mydata_train_df.data) # fit_transform learns the vocab and one-hot encodes 
X_test_tfidfV = tfidfV.transform(mydata_test_df.data) # transform uses the same vocab and one-hot encodes 

X_train_tfidfV_df = pd.DataFrame(X_train_tfidfV.todense())
X_train_tfidfV_df.columns = sorted(tfidfV.vocabulary_)

# Performance of LR Classifier with No Stemming & Lemmatization
text_clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words=stopwords)),
        ('clf', LogisticRegression())])

text_clf.fit(mydata_train_df.data, mydata_train_df.target)
print(text_clf.score(mydata_test_df.data, mydata_test_df.target))
