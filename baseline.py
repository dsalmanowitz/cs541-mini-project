from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import pandas as pd
import time

start_time = time.time()

stopwords = set(stopwords.words('english'))

training = fetch_20newsgroups(subset='train', shuffle=True)
test = fetch_20newsgroups(subset='test', shuffle=True)
training_df = pd.DataFrame({'data': training.data, 'target': training.target})
test_df = pd.DataFrame({'data': test.data, 'target': test.target})

text_clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words=stopwords)),
        ('clf', LogisticRegression())])

text_clf.fit(training_df.data, training_df.target)
print(text_clf.score(test_df.data, test_df.target))
print(f'Time: {time.time()-start_time}')
