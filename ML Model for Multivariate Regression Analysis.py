#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df=pd.read_csv("myhomeprices.csv")
df


# In[3]:


#Handling the missing values
df.bedrooms.median()


# In[4]:


df.bedrooms=df.bedrooms.fillna(df.bedrooms.median()) #Fill any :fillna apply kore replace korbo 
df


# In[5]:


regmultiple=linear_model.LinearRegression()
regmultiple.fit(df[['area','bedrooms','age']],df.price)


# In[6]:


regmultiple.predict([[2800,7,26]])


# In[7]:


regmultiple.intercept_


# In[8]:


regmultiple.coef_


# In[ ]:




