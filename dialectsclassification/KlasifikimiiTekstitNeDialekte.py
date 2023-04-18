#!/usr/bin/env python
# coding: utf-8

# In[13]:


import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns


# prej CSV file
X_train = []
y_train = []
with open(r'C:\Users\admin\Downloads\tekstshqip.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row
    for row in reader:
        label = row[1]
        message = row[0]
        X_train.append(message)
        if label == 'gegë':
            y_train.append(1)
        else:
            y_train.append(0)


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)


clf = MultinomialNB()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

text_to_test = "tue pa e tue ba"
X_test = vectorizer.transform([text_to_test])
prediction = clf.predict(X_test)
if prediction == 1:
    print("Teksti u klasifikua  ne dialektin Gegë ")
else:
    print("Teksti u klasifikua ne dialektin Toske")



conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", conf_mat)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




