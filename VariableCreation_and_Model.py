#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Fundamentals_Ticker.csv')


# In[12]:


pd.set_option('display.max_columns',5000)
df1.columns


# In[9]:


df1=df.drop(['Duplicates','tic','Unnamed: 0', 'GVKEY','gvkey','fyearq','fqtr','indfmt','consol','popsrc','curcdq','datafmt','datacqtr','datafqtr','costat','dvpq','cdvcy','dvpy','dvy','ffoq','udmbq','udoltq','dd1q','dlcq','esubq','uceqq','teqq','uinvq','ugiq','cibegniq','ciq','ibadj12','niitq','ugiq','capr3q','glaq','npq','optrfrq','mkvaltq','ivltq','ivstq','chq','altoq','ancq','lltq','wcapq','sic','spcindcd','spcseccd','txty','csh12q'],axis=1)


# In[14]:


df1 = df1.dropna(subset = ['Index'])


# In[15]:


pd.set_option('display.max_rows',5000)
#df1.isnull().sum().sort_values()


# In[17]:


df1=df1.groupby('IBES Ticker').apply(lambda group: group.interpolate(method='linear'))


# In[23]:


#df1.isnull().sum().sort_values()


# In[9]:


#df1.corr().to_csv('corr.csv')


# In[19]:


df1['Volatility']=(df1['prchq']-df1['prclq'])/df1['prclq']*100


# In[20]:


l=['cshiq','cshopq','npatq','rdipq']
for i in l:
    df1['dummy'+i]=np.where(df1[i]>0,1,0)


# In[21]:


df1.drop(['prchq','prclq','cshiq','cshopq','npatq','rdipq','xrdq','recchy','invchy','lctq','actq','oancfy','capxy'],axis=1,inplace=True)


# In[24]:


#df1.isnull().sum().sort_values()


# In[25]:


df1['txt']=np.where(((np.isnan(df1['txpq']))*1==0) & (np.isnan(df1['txtq'])*1==1),df1['txpq'],df1['txtq'])
df1['csh']=np.where(((np.isnan(df1['chechy']))*1==0) & (np.isnan(df1['cheq'])*1==1),df1['chechy'],df1['cheq'])
df1['rev']=np.where(((np.isnan(df1['revty']))*1==0) & (np.isnan(df1['revtq'])*1==1),df1['revty'],df1['revtq'])
df1['ppe']=np.where(((np.isnan(df1['ppegtq']))*1==0) & (np.isnan(df1['ppentq'])*1==1),df1['ppegtq'],df1['ppentq'])
df1['dp']=np.where(((np.isnan(df1['dpy']))*1==0) & (np.isnan(df1['dpq'])*1==1),df1['dpy'],df1['dpq'])
df1['cogs']=np.where(((np.isnan(df1['cogsy']))*1==0) & (np.isnan(df1['cogsq'])*1==1),df1['cogsy'],df1['cogsq'])
df1['ib']=np.where(((np.isnan(df1['ibq']))*1==0) & (np.isnan(df1['ibadjq'])*1==1),df1['ibq'],df1['ibadjq'])
df1['sales']=np.where(((np.isnan(df1['rev']))*1==0) & (np.isnan(df1['saleq'])*1==1),df1['rev'],df1['saleq'])
df1['at']=np.where(((np.isnan(df1['aoq']))*1==0) & (np.isnan(df1['atq'])*1==1),df1['aoq'],df1['atq'])
df1['ni']=np.where(((np.isnan(df1['niy']))*1==0) & (np.isnan(df1['niq'])*1==1),df1['niy'],df1['niq'])


# In[26]:


s=['ppegtq','txpq','chechy','ppentq','txtq','cheq','revtq','revty','dpq','dpy','cogsy','cogsq','ibq','ibadjq','saleq','rev','atq','aoq','niq','niy','ltq','dlttq','rectq','capsq','cstkq','invtq','apq','cshoq']
df1.drop(s,axis=1,inplace=True)


# In[28]:


#df1.isnull().sum().sort_values()


# In[33]:


#df1.dropna()
df1.columns


# In[32]:


df1['datadate']=pd.to_datetime(df1['datadate'])


# In[34]:


df1=df1.sort_values(by=['IBES Ticker','datadate'],ascending=[True,True])


# In[35]:


df1['Day'] = df1['datadate'].dt.day
df1['Month'] = df1['datadate'].dt.month
df1['Year'] = df1['datadate'].dt.year


# In[36]:


def align(x):
    a = int(x/3)
    if a == 0 :
        return 12
    else:
        return a*3


# In[37]:


sub_year=((df1['Month'].values/3).astype(int)==0)*1
df1['Year'] = (df1['Year'].values - sub_year)
df1['Month'] = df1['Month'].apply(align)
df1['Day'] = 28


# In[38]:


def index(x):
    if int(x/1000)==0:
        return ('0'+str(x))
    else:
        return str(x)


# In[40]:


df1['Quarter'] = df1['Month']*100 + df1['Year']%100
df1['Index'] = df1['Quarter'].apply(index)
df1['Index'] = df1['IBES Ticker'] + df1['Index']
df1=df1.drop(['Day','Month','Year','Quarter'],axis=1)


# In[459]:


df1.to_csv('CleanedRawData.csv')


# In[26]:


df2=pd.read_csv("IBES_EPS.csv")


# In[27]:


df2.drop(['Unnamed: 0','STDEV'],axis=1,inplace=True)


# In[28]:


#df2


# In[45]:


df1.drop(['gind','gsubind','naics','q_Index','y_Index'],axis=1,inplace=True)


# In[30]:


df3=pd.merge(df1,df2,how='inner',on='Index').drop('MEASURE',axis=1)


# In[59]:


df1.drop(['y_Index','q_Index-1'],axis=1,inplace=True)
df1.isna().sum()


# In[61]:


df1['q_Index'] = df1['Index'].str[-4:-2]
df1['y_Index'] = df1['Index'].str[-2:]
df1


# In[62]:


temp = df1['q_Index'].astype(int) - 3

temp2 = ((temp == 0)*1)


temp3 = temp2*(12)

temp = temp + temp3

df1['q_Month'] = temp
df1['q_Year'] = df1['y_Index'].astype(int) - temp2

df1['y_Month'] = df1['q_Index'].astype(int)
df1['y_Year'] = df1['y_Index'].astype(int) - 1

#df3


# In[63]:


df1['q_Month'] = 100 + df1['q_Month']
df1['q_Year'] = 100 + df1['q_Year']
df1['y_Month'] = 100 + df1['y_Month']
df1['y_Year'] = 100 + df1['y_Year']


# In[64]:


df1['q_Quarter'] = df1['q_Month'].astype(str).str[-2:] + df1['q_Year'].astype(str).str[-2:]

df1['y_Quarter'] = df1['y_Month'].astype(str).str[-2:] + df1['y_Year'].astype(str).str[-2:]

#df3


# In[65]:


df1['q_Index']=df1['IBES Ticker']+df1['q_Quarter']
df1['y_Index']=df1['IBES Ticker']+df1['y_Quarter']
#df3


# In[67]:


df1 = df1.drop(['q_Month', 'q_Year', 'y_Month', 'y_Year', 'q_Quarter', 'y_Quarter'], axis = 1)
df1


# In[68]:


df1.columns


# In[74]:


var = [ 'ceqq','prccq', 'req', 'xoprq','dp','ppe','txt', 'csh', 'cogs', 'sales', 'at', 'ni','ib']


# In[70]:


def qoq(x, a):
    x['new'] = x[a].shift(1)
    x['indexShift1'] = x['Index'].shift(1)
    x['new'] = np.where(x['q_Index'] == x['indexShift1'], x['new'], np.nan)
    return x['new']


# In[71]:


def yoy(x, a):
    x['new'] = x[a].shift(4)
    x['indexShift4'] = x['Index'].shift(4)
    x['new'] = np.where(x['y_Index'] == x['indexShift4'], x['new'], np.nan)
    return x['new']


# In[ ]:


'ceqq','prccq', 'req', 'xoprq','dp',


# In[75]:


loc = df1.shape[1]
df2 = df1.groupby('IBES Ticker', group_keys = False)
for a in var:
    print(a)
    strg1 = "qoq_" + a
    column_qoq = df2.apply(lambda x: qoq(x, a))
    strg2 = "yoy_" + a
    column_yoy = df2.apply(lambda x: yoy(x, a)) 
    df1.insert(loc, strg1, column_qoq)
    df1.insert(loc + 1, strg2, column_yoy)
    loc = loc + 2


# In[77]:


#df1.isnull().sum()


# In[78]:


pd.set_option('display.max_columns',100)
#df3.dropna()


# In[79]:


var_del = [ 'ceqq','prccq', 'req', 'xoprq','dp','ppe','txt', 'csh', 'cogs', 'sales', 'at', 'ni','ib']
#df3=df3.dropna()


# In[81]:


for x in var_del:
    strg1 = "del_qoq_" + x
    strg2 = "del_yoy_" + x
    strg3 = "diff_qoq_" + x
    strg4 = "diff_yoy_" + x
    qoq = "qoq_" + x
    yoy = "yoy_" + x
    df1[strg1] = (df1[x] - df1[qoq])/df1[qoq]*100
    df1[strg2] = (df1[x] - df1[yoy])/df1[yoy]*100
    df1[strg3] = (df1[x] - df1[qoq])
    df1[strg4] = (df1[x] - df1[yoy])
#    df3[strg1] =  df3[qoq]/df3['at']
#    df3[strg2] =  df3[yoy]/df3['at']
#df3['ACTUAL']=df3['ACTUAL']*df3['at']


# In[50]:


#df3.isnull().sum()


# In[106]:


df4=df1.drop(['ceqq','req', 'xoprq', 'prccq','txt', 'csh','cogs', 'sales', 'at', 'ni','datadate','q_Index','y_Index'], axis = 1)


# In[107]:


df4=df4.dropna()


# In[108]:


df4


# In[109]:


df4 = df4.replace([np.inf, -np.inf], np.nan)
for x in var_del:
    strg1 = "del_qoq_" + x
    strg2 = "del_yoy_" + x
    strg3 = "diff_qoq_" + x
    strg4 = "diff_yoy_" + x
    df4[strg1]=np.where((df4[strg3]>0) & (np.isnan(df4[strg1])*1==1),1,df4[strg1])
    df4[strg1]=np.where((df4[strg3]==0) & (np.isnan(df4[strg1])*1==1),0,df4[strg1])
    df4[strg1]=np.where((df4[strg3]<0) & (np.isnan(df4[strg1])*1==1),-1,df4[strg1])
    df4[strg2]=np.where((df4[strg4]>0) & (np.isnan(df4[strg2])*1==1),1,df4[strg2])
    df4[strg2]=np.where((df4[strg4]==0) & (np.isnan(df4[strg2])*1==1),0,df4[strg2])
    df4[strg2]=np.where((df4[strg4]<0) & (np.isnan(df4[strg2])*1==1),-1,df4[strg2])


# In[110]:


#df4.to_csv('temp1.csv')


# In[111]:


#df4.isnull().sum()


# In[112]:


df4=df4.dropna()
df4


# In[131]:


df5.columns


# In[127]:


l=[ 'qoq_ceqq',
       'yoy_ceqq', 'qoq_prccq', 'yoy_prccq', 'qoq_req', 'yoy_req', 'qoq_xoprq',
       'yoy_xoprq', 'qoq_txt',
       'yoy_txt', 'qoq_csh', 'yoy_csh', 'qoq_cogs', 'yoy_cogs', 'qoq_sales',
       'yoy_sales', 'qoq_at', 'yoy_at', 'qoq_ni', 'yoy_ni','diff_qoq_ceqq',
       'diff_yoy_ceqq', 'diff_qoq_req',
       'diff_yoy_req', 'diff_qoq_xoprq',
       'diff_yoy_xoprq', 'diff_qoq_prccq',
       'diff_yoy_prccq','diff_qoq_txt', 'diff_yoy_txt','diff_qoq_csh', 'diff_yoy_csh'
  ,'diff_qoq_cogs', 'diff_yoy_cogs','diff_qoq_sales', 'diff_yoy_sales','diff_qoq_at', 'diff_yoy_at','diff_qoq_ni',
       'diff_yoy_ni','ppe', 'dp', 'ib', 'qoq_dp', 'yoy_dp', 'qoq_ppe', 'yoy_ppe', 'qoq_ib', 'diff_qoq_ib',
       'diff_yoy_ib','diff_qoq_dp', 'diff_yoy_dp','diff_qoq_ppe', 'diff_yoy_ppe',
       'yoy_ib']


# In[128]:


df5=df4.drop(l,axis=1)


# In[130]:


df5.to_csv('fundamentalsdata.csv')


# In[126]:


gk=df5.groupby('IBES Ticker', group_keys = False)


# In[127]:


v=['del_qoq_ACTUAL','del_qoq_prccq','del_yoy_prccq','Volatility']
for i in v:
    col = gk.apply(lambda x : x[i].shift(-1))
    df5.insert(df5.shape[1], i+str('lead'), col)


# In[128]:


df5=df5.set_index('Index')


# In[129]:


df5.dropna(inplace = True)


# In[133]:


df5.drop(['tic','del_qoq_prccq','del_yoy_prccq','Volatility'], axis = 1, inplace = True)


# In[135]:


df5.to_csv('NeuralData.csv')


# In[134]:


df5


# In[501]:


'''from scipy import stats
z=np.abs(stats.zscore(df5))
threshold=3
df_5=df5[(z<threshold).all(axis=1)]
df_5'''


# In[ ]:


from keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


# In[ ]:


X=df5.drop(['del_qoq_ACTUAL_lead'],axis=1)
y=df5['del_qoq_ACTUAL_lead']


# In[600]:


y_train


# In[591]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[603]:


regressor = Sequential()
regressor.add(Dense(units=30, input_dim=56,activation='tanh'))
regressor.add(Dense(units=15,activation='relu'))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae'])
regressor.fit(X_train,np.asarray(y_train),epochs=2000,batch_size=10000)


# In[417]:


y_pred = regressor.predict(X_test.drop(['NUMEST','MEDEST','MEANEST','HIGHEST','LOWEST'],axis=1))


# In[432]:


np.sum(y_pred>0)


# In[419]:


y_test


# In[420]:


from sklearn.metrics import r2_score
coefficient_of_dermination = r2_score(y_test, y_pred)
coefficient_of_dermination


# In[ ]:


from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_test, y_pred)
a**0.5


# In[582]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[583]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[584]:


from sklearn.metrics import r2_score
coefficient_of_dermination = r2_score(y_test, predictions)
coefficient_of_dermination


# In[585]:


import xgboost as xgb
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")


# In[569]:


from sklearn.preprocessing import  MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.30)


# In[604]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=1)
'''xgb_model = xgb.XGBRegressor()
# Tuning to get best fir parameters
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(15, 50), # default 100
    "subsample": uniform(0.6, 0.4)
         }
search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=100, cv=3, verbose=0, n_jobs=1, return_train_score=True)
search.fit(X_train.drop(['NUMEST','MEDEST','MEANEST','HIGHEST','LOWEST'],axis=1), y_train)
print(search.best_params_)
print(search.best_score_)'''


xg_reg = xgb.XGBRegressor(colsample_bytree=0.7079533931624865, gamma=0.29288779063673165, learning_rate=0.31206907242748727, max_depth=5, n_estimators=26, subsample= 0.7552679704826087)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

print( xg_reg.score(X_test, y_test) ) # Standard Rsq -> same as r2_score
print( metrics.explained_variance_score(y_test, preds) )
print( metrics.r2_score(y_test, preds) )


# In[409]:


len(np.unique(preds))


# In[410]:


len(y_test.unique())


# In[411]:


preds


# In[412]:


y_test


# In[ ]:





# In[ ]:




