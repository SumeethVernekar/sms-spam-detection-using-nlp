import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

doc= pd.read_table("data",header=None,names=['class','sms'])

hs=doc['class'].value_counts()

doc['label'] = doc['class'].map({'ham':0,'spam':1})
doc=doc.drop(['class'],axis=1)
print(doc.head())

x=doc['sms']
y=doc['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.head())
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words='english',max_features=1000,lowercase=True,ngram_range=(1,1),analyzer='word',max_df=0.8,min_df=1)
vect.fit(x_train)
x_train_transformed = vect.transform(x_train)
x_test_transformed=vect.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_transformed,y_train)
y_pred = model.predict(x_test_transformed)
y_pred_prob=model.predict_proba(x_test_transformed)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
precision=metrics.precision_score(y_test,y_pred)
recall=metrics.recall_score(y_test,y_pred)
f1=metrics.f1_score(y_test,y_pred)
print("Accuracy",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("f1 score",f1)