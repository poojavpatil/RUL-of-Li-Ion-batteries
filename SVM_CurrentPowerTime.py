#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Load data from CSV file

import pandas as pd


# In[2]:


CurrentPowerTime = pd.read_csv('current,power vs time.csv')
CurrentPowerTime


# In[3]:


CurrentPowerTime.shape


# In[4]:


CurrentPowerTime.size


# In[5]:


CurrentPowerTime.count()


# In[6]:


CPT = CurrentPowerTime[1:200]
CPT.plot(x='Current',y='Power',color='blue',label='Current vs Power')


# In[7]:


CurrentPowerTime.dtypes


# In[8]:


###Identifying Unwanted Rows

CurrentPowerTime.dtypes


# In[9]:


CurrentPowerTime.columns


# In[10]:


import numpy as np
CurrentPowerTime.columns
features = CurrentPowerTime[['Category', 'Current', 'Power']]
x=np.asarray(features)
y=np.array(CurrentPowerTime['Power'])
x[1:10]


# In[11]:


###Divide Data into train/test data set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
x_train.shape
x_test.shape


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
y_train.shape
y_test.shape


# In[13]:


###Modeling


from sklearn import svm
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
classifier = svm.SVC(kernel='poly',gamma='auto',C=0)
classifier


# In[17]:


#classifier.fit(x_train,y_train)
#y_predict = classifier.predict(x_test)


# In[ ]:





# In[ ]:





# In[ ]:




