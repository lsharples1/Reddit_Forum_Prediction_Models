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

count_vect = CountVectorizer()
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_train = count_vect.fit_transform(train.body)
X_test = count_vect.transform(test.body)
Y_train = train.forum
Y_test = test.forum


#print(train['forum'].value_counts())


#X_train_counts = count_vect.fit_transform(twenty_train.data)

clfM = MultinomialNB().fit(X_train, Y_train)
clfB = BernoulliNB().fit(X_train, Y_train)

Y_predM = clfM.predict(X_test)
Y_predB = clfB.predict(X_test)

accuracyM = metrics.accuracy_score(Y_test, Y_predM)
accuracyB = metrics.accuracy_score(Y_test, Y_predB)
#Multinomial NB with no adjustments
#plot(clfM, X_test, Y_test, "Multinomial NB")
print('Multinomial Accuracy (OG data) = ' +str(accuracyM))

#Bernoulli NB with no adjustments
#plot(clfB, X_test, Y_test, "Bernoulli NB")
print('Bernoulli Accuracy (OG data) = ' + str(accuracyB))

#X_new_counts = count_vect.transform(test.body)
#print(X_new_counts)

#predicted = clfM.predict(X_new_counts)
#for doc, category in zip(test.body, predicted):
   # print(f'{doc} => {category}')
