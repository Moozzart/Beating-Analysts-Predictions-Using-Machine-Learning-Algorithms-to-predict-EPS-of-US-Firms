#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.chdir(r"C:\Users\Shivangi Srivastava\Desktop\NewAP")


# In[3]:


fund = pd.read_csv('Fundamentals_Gvkey.csv') # reading Fundamentals Data with only GVKEY 
ibes = pd.read_csv('IBES_withTicker_withIndex.csv') # EPS data with Index information


# In[4]:


map_dat = pd.read_csv('FinalListCSV.csv') # file that contains IBES Ticker for GVKEY


# In[5]:


fund


# In[6]:


map_dat


# In[7]:


#getting Fundamentals Data with IBES TICKER by merging the previous fundamentals data containing only GVKEY with data that contains map of GVKEY to IBES Ticker
fund_index = pd.merge(map_dat, fund, left_on = "GVKEY", right_on = "gvkey", how = "right") 


# In[8]:


fund_index


# In[ ]:


fund_index['datadate']=pd.to_datetime(fund_index['datadate'], format="%d/%m/%Y") # converting the date information to pandas datetime format


# In[ ]:


fund_index = fund_index.sort_values(by=['IBES Ticker','datadate'],ascending=[True,True])


# In[ ]:


# Extracting the day, month and year data from date and saving in new columns
fund_index['Day'] = fund_index['datadate'].dt.day
fund_index['Month'] = fund_index['datadate'].dt.month
fund_index['Year'] = fund_index['datadate'].dt.year


# In[ ]:


def align(x): # changing 1,2 to 12, 4,5 to 3, 7,8 to 6 and 10,11 to 9
    a = int(x/3)
    if a == 0 :
        return 12
    else:
        return a*3


# In[ ]:


sub_year=((fund_index['Month'].values/3).astype(int)==0)*1
fund_index['Year'] = (fund_index['Year'].values - sub_year)
fund_index['Month'] = fund_index['Month'].apply(align)
fund_index['Day'] = 28


# In[ ]:


# function to create the four digit format for quarter
def index(x):
    if int(x/1000)==0:
        return ('0'+str(x))
    else:
        return str(x)


# In[ ]:


fund_index['Quarter'] = fund_index['Month']*100 + fund_index['Year']%100 # Joining month and year information to create quarter
fund_index['Index'] = fund_index['Quarter'].apply(index) # Getting a 4 digit format for quarter in case the month is a single digit
fund_index['Index'] = fund_index['IBES Ticker'] + fund_index['Index'] # Joining Ticker with quarter to get index
fund_index=fund_index.drop(['Day','Month','Year','Quarter'],axis=1)


# In[ ]:


fund_index


# In[ ]:


fund_index.to_csv('Fundamentals_Ticker.csv')


# In[ ]:


df3=pd.merge(fund_index,ibes,how='inner',on='Index') # Merging Fundamentals data with IBES Data


# In[ ]:


df3


# Making Indexes of previous quarter and previous year fo reach row

# In[ ]:


df3['q_Index'] = df3['Index'].str[-4:-2]
df3['y_Index'] = df3['Index'].str[-2:]


# In[ ]:


# Getting month and year of the previous quarter and year
temp = df3['q_Index'].astype(int) - 3 

temp2 = ((temp == 0)*1)


temp3 = temp2*(12)

temp = temp + temp3

df3['q_Month'] = temp
df3['q_Year'] = df3['y_Index'].astype(int) - temp2

df3['y_Month'] = df3['q_Index'].astype(int)
df3['y_Year'] = df3['y_Index'].astype(int) - 1


# In[ ]:


df3['q_Month'] = 100 + df3['q_Month']
df3['q_Year'] = 100 + df3['q_Year']
df3['y_Month'] = 100 + df3['y_Month']
df3['y_Year'] = 100 + df3['y_Year']


# In[ ]:


# Creating Quarter
df3['q_Quarter'] = df3['q_Month'].astype(str).str[-2:] + df3['q_Year'].astype(str).str[-2:]
df3['y_Quarter'] = df3['y_Month'].astype(str).str[-2:] + df3['y_Year'].astype(str).str[-2:]


# In[ ]:


# Creating Index
df3['q_Index']=df3['IBES Ticker']+df3['q_Quarter'] 
df3['y_Index']=df3['IBES Ticker']+df3['y_Quarter']


# In[ ]:


df3 = df3.drop(['q_Month', 'q_Year', 'y_Month', 'y_Year', 'q_Quarter', 'y_Quarter'], axis = 1)


# In[ ]:


df3.to_csv('FundamentalsIBES.csv')

