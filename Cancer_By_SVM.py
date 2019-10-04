#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# 

# In[4]:


df=pd.read_csv("cell_samples.csv")
df[0:5]


# In[8]:


ax=df[df['Class']==4][0:5].plot(kind='scatter',x='Clump',y='UnifSize',color='Red',label='Malignant')
df[df['Class']==2][0:5].plot(kind='scatter',x='Clump',y='UnifSize',color='DarkBlue',label='Benign',ax=ax)
plt.show()


# In[5]:


df.dtypes


# In[9]:


df=df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()]
df['BareNuc']=df['BareNuc'].astype('int')


# In[12]:


feature_df =df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x = np.asarray(feature_df)
x[0:5]


# In[11]:


y=df['Class'].values
y[0:5]


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[31]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train) 


# In[32]:


yhat = clf.predict(x_test)
yhat [0:5]


# In[33]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,yhat)


# In[ ]:




