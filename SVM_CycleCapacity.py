#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Load data from CSV file

import pandas as pd


# In[2]:


CycleCapacity=pd.read_csv('Cycle n Capacity.csv')


# In[3]:


CycleCapacity.shape


# In[4]:


CycleCapacity.size


# In[5]:


CycleCapacity.count()


# In[6]:


###Distribution of the classes

CycleC = CycleCapacity[1:200]
CycleC.plot( x='Cycle', y='Capacity(Ah)', color='blue', label='Cycle capacity' )


# In[7]:


CycleCapacity.dtypes


# In[8]:


###Identifying Unwanted Rows

CycleCapacity.dtypes
CycleCapacity=CycleCapacity[pd.to_numeric(CycleCapacity['Cycle'],errors='coerce').notnull()]
CycleCapacity['Cycle'] = CycleCapacity['Cycle'].astype('int')
CycleCapacity.dtypes


# In[9]:


CycleCapacity.columns


# In[10]:


import numpy as np
CycleCapacity.columns
features = CycleCapacity[['Cycle', 'Capacity(Ah)', 'Voltage Measured(V)', 'Current Measured','Temperature Measured', 'Time Measured(Sec)', 'SampleId']]
x=np.asarray(features)
y=np.array(CycleCapacity['Capacity(Ah)'])
x[1:10]


# In[11]:


###Divide Data into train/test data set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
x_train.shape #508 * 7
x_test.shape #128 *7


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
y_train.shape # 508
y_test.shape #128


# In[13]:


###Modeling


from sklearn import svm
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
classifier = svm.SVC(kernel='linear',gamma='auto',C=0)
classifier
#classifier.fit(x_train,y_train)
#y_predict = classifier.predict(x_test)


# In[14]:


from sklearn.metrics import classification_report
#print(classification_report(y_test, y_predict))


# In[ ]:





# In[ ]:





# In[ ]:




