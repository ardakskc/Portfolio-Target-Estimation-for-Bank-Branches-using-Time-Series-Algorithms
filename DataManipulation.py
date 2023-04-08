import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

plt.style.use('dark_background')

# load the dataset
df = pd.read_csv('hedef2.csv',sep=';',encoding='ISO:8859-1')
print(df.dtypes)

print(df.columns.tolist())
df['REF_DONEMINSONGUNU'] = pd.to_datetime(df['REF_DONEMINSONGUNU'],infer_datetime_format=True).dt.strftime('%m-%Y')#
#df.set_index('REF_DONEMINSONGUNU', inplace=True)

df_new = df[(df['REF_SUBE_ID']==2)].copy()
tarih = df_new['REF_DONEMINSONGUNU'].copy().to_frame()
tarih.columns.values[0]='Tarih'
df_new.drop('REF_SUBE_ID', axis=1, inplace=True)
df_new.drop('FACT_HEDEFDEGER', axis=1, inplace=True)
df_new=pd.concat([df_new, tarih], axis=1)

df.set_index('REF_DONEMINSONGUNU', inplace=True)
df_new.set_index('REF_DONEMINSONGUNU', inplace=True)



sube_sayisi = np.unique(df['REF_SUBE_ID'].values)
#df1 = df[(df['REF_SUBE_ID']==1)].get("FACT_HEDEFDEGER").to_frame()
for i in sube_sayisi:
    column_name = 'Sube_{}'.format(i)
    df1 = df[(df['REF_SUBE_ID']==i)].copy()
    df1.drop('REF_SUBE_ID', axis=1, inplace=True)
    #df1.drop('REF_DONEMINSONGUNU', axis=1, inplace=True)
    df1.columns.values[0]=column_name
    #df_new=df_new.append(df1)
    df_new=pd.concat([df_new, df1], axis=1)

df_new.to_csv('modified.csv', index=False, sep=';',encoding='ISO:8859-1')
print(df.head())

#df.set_index('REF_DONEMINSONGUNU', inplace=True)
df_new['Tarih'] = pd.to_datetime(df_new['Tarih'],infer_datetime_format=True)
df.head()
df.dtypes
#plt.plot(df['FACT_HEDEFDEGER'])

df.shape
######################################################################

