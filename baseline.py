from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import pandas as pd
import time

sum_time = 0
sum_acc = 0
stop_words = set(stopwords.words('english'))

for i in range(10):

  start_time = time.time()

  training = fetch_20newsgroups(subset='train', shuffle=True)
  test = fetch_20newsgroups(subset='test', shuffle=True)
  training_df = pd.DataFrame({'data': training.data, 'target': training.target})
  test_df = pd.DataFrame({'data': test.data, 'target': test.target})

  text_clf = Pipeline([
          ('vect', TfidfVectorizer(stop_words=stop_words)),
          ('clf', LogisticRegression())])

  text_clf.fit(training_df.data, training_df.target)
  a = text_clf.score(test_df.data, test_df.target)
  t = time.time()-start_time
  print(f'Acc: {a}')
  print(f'Time: {t}')
  sum_time += t
  sum_acc += a
print(sum_acc/10)
print(sum_time/10)