#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


ds= pd.read_csv('UpdatedResumeDataSet.csv')


# In[3]:


ds.head()


# In[4]:


ds.isnull().sum()


# In[33]:


ds['Category'].value_counts()


# In[5]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
corpus= []


# In[6]:


for i in range(0,962):
    review = re.sub('[^a-zA-Z]', ' ',str(ds['Resume'][i]))
    review= review.lower()
    review =review.split()
    review = [ps.stem(w) for w in review if not w in set(stopwords.words('English'))]
    review= ' '.join(review)
    corpus.append(review)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[12]:


ds['Category'] = encoder.fit_transform(ds['Category'])


# In[13]:


y = ds['Category'].values


# In[16]:


x_train, x_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state =32)


# In[18]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# In[19]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

