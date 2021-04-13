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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

crypto = pd.read_csv('Data/Crypto.csv', usecols=['body'])
crypto['forum'] = "r/Crypto"

wsb = pd.read_csv('Data/WallStreetBets.csv', usecols=['title'])
wsb.rename(columns={'title':'body'}, inplace=True)
wsb['forum'] = "r/WallStreetBets"

#combined android datasets from 2018-2019 and 2019-2020
android = pd.read_csv('https://raw.githubusercontent.com/lsharples1/Mini_Project_2/main/data/r_androiddev/androiddev_2019_2020.csv?token=AO4LMYPMCJ3WB6Q3BEE576TAO4ECE', usecols=['text'])
android.rename(columns={'text':'body'}, inplace=True)
android2 = pd.read_csv('https://raw.githubusercontent.com/lsharples1/Mini_Project_2/main/data/r_androiddev/androiddev_2018_2019.csv?token=AO4LMYOSRHE2ZXWQYYJNYETAO5IOS', usecols=['text'])
android2.rename(columns={'text':'body'}, inplace=True)
android = pd.concat([android, android2])
android['forum'] = 'r/AndroidDev'

news = pd.read_csv('https://raw.githubusercontent.com/lsharples1/Mini_Project_2/main/data/reddit_worldnews_start_to_2016-11-22.csv?token=AO4LMYIOO3E7K25BLB7YZVTAO5I44', usecols=['title'])
news.rename(columns={'title':'body'}, inplace=True)
news['forum'] = 'r/WorldNews'

vm = pd.read_csv('Data/VaccineMyths.csv', usecols=['body'])
vm['forum'] = "r/VaccineMyths"

cv = pd.read_csv('https://raw.githubusercontent.com/lsharples1/Mini_Project_2/main/data/Coronavirus_1k.csv?token=AO4LMYKDHGLUO2LD2THDWGTAO4CRI', usecols=['title'])
cv.rename(columns={'title':'body'}, inplace=True)
cv['forum'] = "r/Coronavirus"


data = pd.DataFrame().append([crypto,wsb,vm,cv,android,news])
df = data.dropna()
count_vect = CountVectorizer()

#data default split .25/.75
train, test = train_test_split(df, random_state=5, shuffle=True)

X_train = count_vect.fit_transform(train.body)
X_test = count_vect.transform(test.body)

Y_train = train.forum
Y_test = test.forum

tfidf_count_vect = TfidfVectorizer()
tfidf_X_train = tfidf_count_vect.fit_transform(train.body)
tfidf_X_test = tfidf_count_vect.transform(test.body)

#MNB .965, BNM .860, RS =  5, default train test split
clfM = MultinomialNB().fit(X_train, Y_train)
clfB = BernoulliNB().fit(X_train, Y_train)

Y_predM = clfM.predict(X_test)
Y_predB = clfB.predict(X_test)


# tfidf model
clfTB = BernoulliNB().fit(tfidf_X_train, Y_train)
clfTM = MultinomialNB().fit(tfidf_X_train, Y_train)

# predictions and accuracies

Y_predM = clfM.predict(X_test)
Y_predB = clfB.predict(X_test)
Y_predTB = clfTB.predict(tfidf_X_test)
Y_predTM = clfTB.predict(tfidf_X_test)

accuracyM = metrics.accuracy_score(Y_test, Y_predM)
accuracyB = metrics.accuracy_score(Y_test, Y_predB)
accuracyTB = metrics.accuracy_score(Y_test, Y_predTB)
accuracyTM = metrics.accuracy_score(Y_test, Y_predTM)



accuracyM = metrics.accuracy_score(Y_test, Y_predM)
accuracyB = metrics.accuracy_score(Y_test, Y_predB)


def plot(classifier, x, y, title): 
    class_names = ['Crypto','WSB','VM','Corona','Android','WN']
    disp = plot_confusion_matrix(classifier, x, y,
                                 display_labels=class_names,
                                 cmap=plt.cm.PuRd,values_format = ''
                                   )
    disp.ax_.set_title(title)
    plt.show() 
    
#Multinomial NB with no adjustments
plot(clfM, X_test, Y_predM, "Multinomial NB")
print('Multinomial Accuracy = ' +str(accuracyM))

#Bernoulli NB with no adjustments
plot(clfB, X_test, Y_predM, "Bernoulli NB")
print('Bernoulli Accuracy = ' + str(accuracyB))

#TFIDF NB with no adjustments
plot(clfTB, X_test, Y_predM, "TFIDF Bernoulli NB")
print('TFIDF Bernoulli Accuracy = ' + str(accuracyTB))

#TFIDF NB with no adjustments
plot(clfTM, X_test, Y_predM, "TFIDF Multinomial NB")
print('TFIDF Multinomial Accuracy = ' + str(accuracyTM))

#X_new_counts = count_vect.transform(test.body)
#print(X_new_counts)

#predicted = clfM.predict(X_new_counts)
#for doc, category in zip(test.body, predicted):
   # print(f'{doc} => {category}')
