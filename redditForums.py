# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as metrics

crypto = pd.read_csv('Data/Crypto.csv', usecols=['body'])
crypto['forum'] = "r/Crypto"
wsb = pd.read_csv('Data/WallStreetBets.csv', usecols=['body'])
wsb['forum'] = "r/WallStreetBets"
vm = pd.read_csv('Data/VaccineMyths.csv', usecols=['body'])
vm['forum'] = "r/VaccineMyths"
#data = crypto.append(wsb)
#data.append(vm)
data = pd.DataFrame().append([crypto,wsb,vm])
df = data.dropna()
print(df['forum'].value_counts())
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
#print(train['forum'].value_counts())
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts = count_vect.fit_transform(train.body)
clf = MultinomialNB().fit(X_train_counts, train.forum)
X_new_counts = count_vect.transform(test.body)
#print(X_new_counts)

predicted = clf.predict(X_new_counts)
for doc, category in zip(test.body, predicted):
    print(f'{doc} => {category}')
