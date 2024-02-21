#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n


# In[7]:


df=pd.read_csv("C://Users//mvish//Desktop//hiring.csv")
df


# In[10]:


df.experience=df.experience.fillna("zero")
df


# In[14]:


df.experience=df.experience.apply(w2n.word_to_num)
df


# In[21]:


import math
median_test_score=math.floor(df["test_score(out of 10)"].mean())
median_test_score


# In[22]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
df


# In[29]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[31]:


reg.predict([[2,5,6]])


# In[ ]:




