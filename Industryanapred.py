#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[72]:


#defining index for dataframe
def index_func(data_frame):
    data_frame['Index'] = data_frame['FPEDATS'].str[3:5]+data_frame['FPEDATS'].str[-2:]
#shifting data for period t for period t-1 to use it for EPS prediction of time period t      
def go_prev(data_frame,series,l):
    data_frame['q_Index-' +str(l)] = series.str[-4:-2]
    data_frame['y_Index'] = series.str[-2:]
    temp = data_frame['q_Index-'+str(l)].astype(int) - 3
    temp2 = ((temp == 0)*1)
    temp3 = temp2*(12)
    temp = temp + temp3
    data_frame['q_Month'] = temp
    data_frame['q_Year'] = data_frame['y_Index'].astype(int) - temp2
    data_frame['q_Month'] = 100 + data_frame['q_Month']
    data_frame['q_Year'] = 100 + data_frame['q_Year']
    data_frame['q_Quarter'] = data_frame['q_Month'].astype(str).str[-2:] + data_frame['q_Year'].astype(str).str[-2:]
    data_frame['q_Index-'+str(l)]=data_frame['TICKER']+data_frame['q_Quarter']
    data_frame = data_frame.drop(['q_Month','q_Year','q_Quarter'], axis = 1)
    return(data_frame)


# In[73]:


#setting up data for our use
actual_with_index = pd.read_csv('IBES_withTicker_withIndex.csv')

# Indexing 

#getting previous quater
actual_with_index = go_prev(actual_with_index,actual_with_index['Index'],1)
#getting 2 quaters before
actual_with_index = go_prev(actual_with_index,actual_with_index['q_Index-1'],2)
#getting 3 quaters before
actual_with_index = go_prev(actual_with_index,actual_with_index['q_Index-2'],3)
#getting 4 quaters before
actual_with_index = go_prev(actual_with_index,actual_with_index['q_Index-3'],4)
#finding the prices for previous periods
actual_with_index.drop('Unnamed: 0',axis=1,inplace=True)

#getting lagged values for our input variables
for i in range(1,5):
    actual_with_index.index = actual_with_index['Index']
    temp = actual_with_index.loc[actual_with_index['q_Index-'+str(i)]]
    actual_with_index.index = np.arange(actual_with_index.shape[0])
    temp = temp.reset_index(drop=True)
    actual_with_index['ACTUALt-'+str(i)] = temp['ACTUAL']
    
#calculating shock = EPS - E(EPS)
#E(EPS) = (EPSt-1+EPSt-2+EPSt-3)/3

actual_with_index['Avg'] = (actual_with_index['ACTUALt-1']+actual_with_index['ACTUALt-2']+actual_with_index['ACTUALt-3'])/3
actual_with_index['shockt'] = actual_with_index['ACTUAL']-actual_with_index['Avg']

#finding last period's shock

actual_with_index.index = actual_with_index['Index']
temp = actual_with_index.loc[actual_with_index['q_Index-1']]
actual_with_index.index = np.arange(actual_with_index.shape[0])
temp.index = np.arange(temp.shape[0])
actual_with_index['shockt-1'] = temp['shockt']

##Modelling Begins

modelling_data = actual_with_index.dropna()


# In[74]:


from sklearn.model_selection import RandomizedSearchCV 


# In[38]:


#FINDING RANKS   
print("Start!")
print(i)
analyst_est = pd.read_csv('modeldata1.csv')
analyst_est.drop('Unnamed: 0',axis=1,inplace=True)
anal_group = analyst_est.groupby('Index')
l=analyst_est['Index'].unique()
l = pd.DataFrame(l,columns=['Index'])
data_with_common_index=pd.merge(modelling_data,l,on='Index',how='inner')
X = data_with_common_index[['ACTUALt-4','shockt-1','ACTUALt-1','MEDEST','MEANEST','Index','ACTUAL','STDEV','10.0', '15.0', '20.0', '25.0', '30.0', '35.0',
   '40.0', '45.0', '50.0', '55.0', '60.0']]
y = data_with_common_index[['ACTUAL']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#params={'bootstrap': [True, False],
#'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#'max_features': ['auto', 'sqrt'],
#'min_samples_leaf': [1, 2, 4],
#'min_samples_split': [2, 5, 10],
#'n_estimators': [80, 100, 120, 140, 160, 180, 200]}
search=RandomizedSearchCV(rf,param_distributions=params,random_state=23,n_iter=50,n_jobs=1)
search.fit(X_train.drop(['MEDEST','MEANEST','Index','ACTUAL','STDEV'], axis = 1), y_train)
print(search.best_params_)
print(search.best_score_)

rf = RandomForestRegressor(n_estimators= 200,
 min_samples_split= 5,
 min_samples_leaf= 2,
 max_features= 'auto',
 max_depth= 60,
 bootstrap= False)
rf.fit(X_train.drop(['MEDEST','MEANEST','Index','ACTUAL','STDEV'],axis=1),y_train)
pred = rf.predict(X_test.drop(['MEDEST','MEANEST','Index','ACTUAL','STDEV'],axis=1))
print('R2 score =',r2_score(y_test,pred))
pred = pd.DataFrame(pred,columns=['Prediction'])
#X_test = X_test.reset_index(drop=True)
#X_test = pd.concat([X_test,pred],axis=1)

#best params{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 60, 'bootstrap': False}


# In[48]:


#resetting index so that they concate easily
X_test = X_test.reset_index()
y_test1=y_test.reset_index()
X_test = pd.concat([X_test,y_test1,pred],axis=1)


# In[53]:


X_test.drop(['Prediction','ACTUAL'],axis=1,inplace=True)
X_test


# In[54]:


#sorting X_test by standard deviation of analysts prediction
X_test.sort_values(by=['STDEV'],ascending=True,inplace=True)


# In[67]:


#finding the bottom of tercile
X_test_3Bot = X_test.iloc[0:int(len(X_test)*0.33),:]
#finding the top of tercile
X_test_3Top = X_test.iloc[int(len(X_test)*0.66):int(len(X_test)),:]
y_test_3Bot = X_test_3Bot['ACTUAL']
pred_3Bot = X_test_3Bot['Prediction']
y_test_3Top = X_test_3Top['ACTUAL']
pred_3Top = X_test_3Top['Prediction']


# In[ ]:


#finding the bottom of quintiles
X_test_3Bot = X_test.iloc[0:int(len(X_test)*0.2),:]
#finding the top of quintiles
X_test_3Top = X_test.iloc[int(len(X_test)*0.8):int(len(X_test)),:]
y_test_3Bot = X_test_3Bot['ACTUAL']
pred_3Bot = X_test_3Bot['Prediction']
y_test_3Top = X_test_3Top['ACTUAL']
pred_3Top = X_test_3Top['Prediction']


# In[ ]:


#finding the bottom of deciles
X_test_3Bot = X_test.iloc[0:int(len(X_test)*0.1),:]
#finding the top of deciles
X_test_3Top = X_test.iloc[int(len(X_test)*0.9):int(len(X_test)),:]
y_test_3Bot = X_test_3Bot['ACTUAL']
pred_3Bot = X_test_3Bot['Prediction']
y_test_3Top = X_test_3Top['ACTUAL']
pred_3Top = X_test_3Top['Prediction']


# In[69]:


#FINDING METRICS
print('Various Metrics')
print(X_test_3Top[['MEANEST','MEDEST', 'ACTUAL', 'Prediction']].corr())
l = ['Prediction','MEDEST','MEANEST']
for i in l:
    X_test_3Top['diff'+str(i)] = abs((X_test_3Top[i] - X_test_3Top['ACTUAL']))

#print((X_test_3Top['diffPrediction']<X_test_3Top['diffMEANEST']).mean())
X_test_3Top_2=X_test_3Top
l = ['Prediction','MEDEST','MEANEST']
for i in l:
    X_test_3Top_2['diff'+str(i)] = abs((X_test_3Top_2[i] - X_test_3Top_2['ACTUAL']))
print('Percentage Mean Analyst Prediction Beaten:',(X_test_3Top_2['diffPrediction']<X_test_3Top_2['diffMEANEST']).mean())

diff_pred = X_test_3Top_2['diffPrediction'].values
Index_list = X_test_3Top_2['Index'].values
X_test_3Top_2.index = X_test_3Top_2['Index']

diff_=[]

for i in range(len(Index_list)):
    temp = anal_group.get_group(Index_list[i])
    temp['analyst_diff'] = abs(temp['VALUE']-temp['ACTUAL'])
    diff_.append((temp['analyst_diff']>=X_test_3Top_2.loc[Index_list[i]]['diffPrediction']).mean())

print('THE AVERAGE PERCENTILE BEAT IS: ',np.mean(np.array(diff_)))
print("End!!")


# In[ ]:




