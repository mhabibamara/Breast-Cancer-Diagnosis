#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer(return_X_y = True, as_frame = True)
a = breast_cancer[0]
b = breast_cancer[1]
a['typeofcancer'] = b
a.shape


# In[7]:


df = a.iloc[0:569, [0,6,7,8,30]]
df.head(2)


# In[8]:


df.iloc[[17,18,19,20,21],0:4]


# In[11]:


dfsorted = df.sort_values ('typeofcancer', ignore_index=True)
f0, f1 = dfsorted.typeofcancer.value_counts()

fig, axs = plt.subplots( figsize = (10,2.5))

axs1 = plt.subplot2grid ( shape = (1, 4), loc = (0,0))
axs2 = plt.subplot2grid ( shape = (1, 4), loc = (0,1))
axs3 = plt.subplot2grid ( shape = (1, 4), loc = (0,2))
axs4 = plt.subplot2grid ( shape = (1, 4), loc = (0,3))
plt.tight_layout()

axs1.hist (dfsorted.iloc[0:f1, 0], edgecolor = 'b', fc = 'none', label = 'c0(M)')
axs1.hist (dfsorted.iloc[f1:f0+f1, 0], edgecolor = 'r', fc = 'none', label = 'c1(B)')
axs1.set_xlabel ('mean radius')
axs1.set_ylabel ('Frequency')
axs1.legend()

axs2.scatter (dfsorted.iloc[0:f0, 1], dfsorted.iloc[0:f0, 0], label = 'c0(M)')
axs2.scatter (dfsorted.iloc[f0:f0+f1, 1], dfsorted.iloc[f0:f0+f1, 0], label = 'c1(B)')
axs2.set_xlabel ('mean concavity')
axs2.set_ylabel ('mean radius')
axs2.legend()

axs3.scatter (dfsorted.iloc[0:f0, 2], dfsorted.iloc[0:f0, 0], label = 'c0(M)')
axs3.scatter (dfsorted.iloc[f0:f0+f1, 2], dfsorted.iloc[f0:f0+f1, 0], label = 'c1(B)')
axs3.set_xlabel ('mean concave points')
axs3.set_ylabel ('mean radius')
axs3.legend()

axs4.scatter (dfsorted.iloc[0:f0, 3], dfsorted.iloc[0:f0, 0], label = 'c0(M)')
axs4.scatter (dfsorted.iloc[f0:f0+f1, 3], dfsorted.iloc[f0:f0+f1, 0], label = 'c1(B)')
axs4.set_xlabel ('mean symmetry')
axs4.set_ylabel ('mean radius')
axs4.legend()


# In[ ]:




