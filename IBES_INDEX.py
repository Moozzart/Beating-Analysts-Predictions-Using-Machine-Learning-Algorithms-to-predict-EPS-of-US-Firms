#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary packages

import pandas as pd
import numpy as np
import os
import datetime as dt


# In[2]:


os.chdir(r"C:\Users\Shivangi Srivastava\Desktop\NewAP")


# In[3]:


ibes = pd.read_csv('IBES_withTicker_new.csv') #reading the file containing EPS details along with IBES Ticker and ANNDATS too


# In[4]:


ibes.head(2)


# In[5]:


ibes['STATPERS']=pd.to_datetime(ibes['STATPERS'], format="%d/%m/%Y") # converting the date information to pandas datetime format
ibes['FPEDATS']=pd.to_datetime(ibes['FPEDATS'], format="%d/%m/%Y")


# In[6]:


ibes=ibes.sort_values(by=['TICKER','FPEDATS','STATPERS'],ascending=[True,True,False]) 
ibes.head(2)


# In[7]:


# Extracting the day, month and year data from date and saving in new columns
ibes['Day'] = ibes['FPEDATS'].dt.day
ibes['Month'] = ibes['FPEDATS'].dt.month
ibes['Year'] = ibes['FPEDATS'].dt.year
ibes.head(2)


# In[8]:


def align(x): # changing 1,2 to 12, 4,5 to 3, 7,8 to 6 and 10,11 to 9
    a = int(x/3)
    if a == 0 :
        return 12
    else:
        return a*3


# In[9]:


sub_year=((ibes['Month'].values/3).astype(int)==0)*1
ibes['Year'] = (ibes['Year'].values - sub_year)
ibes['Month'] = ibes['Month'].apply(align)
ibes['Day'] = 28


# In[10]:


# function to create the four digit format for quarter
def index(x):
    if int(x/1000)==0:
        return ('0'+str(x))
    else:
        return str(x)


# In[11]:


ibes['Quarter'] = ibes['Month']*100 + ibes['Year']%100 # Joining month and year information to create quarter
ibes['Index'] = ibes['Quarter'].apply(index) # Getting a 4 digit format for quarter in case the month is a single digit
ibes['Index'] = ibes['TICKER'] + ibes['Index'] # Joining Ticker with quarter to get index


# In[12]:


IBES=ibes.drop(['Day','Month','Year','Quarter'],axis=1)
IBES.head()


# In[13]:


# Selecting the STATPERS date that is closest to FPEDATS date and smaller than it as multiple dates
# are present in STATPERS for same FPEDATS
IBES = IBES[IBES['STATPERS'] < IBES['FPEDATS']] # retaining only the smaller dates in STATPERS
IBES = IBES.groupby(['TICKER','FPEDATS']).first() # picking the closest STATPERS
IBES = IBES.reset_index()


# In[14]:


IBES.drop(['MEASURE'], axis = 1, inplace = True) # MEASURE not required so dropping it
IBES


# In[15]:


IBES.to_csv('IBES_withTicker_withIndex_new.csv') # final file with index formed along with other EPS data


# In[17]:


IBES


# In[ ]:




