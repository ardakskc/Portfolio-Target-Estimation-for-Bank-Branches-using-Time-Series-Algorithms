import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tsfel as tsfel

plt.style.use('dark_background')

# load the dataset
df = pd.read_csv('modified.csv',sep=';',encoding='ISO:8859-1')
df['Tarih'] = pd.to_datetime(df['Tarih'],infer_datetime_format=True)

print(df.dtypes)

df.set_index('Tarih', inplace=True)

plt.plot(df['Sube_3'])
plt.show()

#Is the data stationary?
#Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df['Sube_3'])
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#Since data is not stationary, we may need SARIMA and not just ARIMA


#Additive time series:
#Value = Base Level + Trend + Seasonality + Error
#Multiplicative Time Series:
#Value = Base Level x Trend x Seasonality x Error
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df['Sube_3'],
                            model ='additive')


#Extract and plot trend, seasonal and residuals.
trend = decomposed.trend
seasonal = decomposed.seasonal #Cyclic behavior may not be seasonal!
residual = decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Sube_3'], label='Original', color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='yellow')
plt.legend(loc='upper left')
plt.show()

#AUTOCORRELATION
#Autocorrelation is simply the correlation of a series with its own lags.
# Plot lag on x axis and correlation on y axis
#Any correlation above confidence lnes are statistically significant.

from statsmodels.tsa.stattools import acf

acf_144 = acf(df.Sube_3, nlags=64)
plt.plot(acf_144)

#Obtain the same but with single line and more info...
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Sube_3)
plt.show()


############################
##Different Feature extractor
cfg = tsfel.get_features_by_domain()
# Extract features
X = tsfel.time_series_features_extractor(cfg, df,fs=10)
############################


##BUTUN SUBELER ICIN ZAMAN SERİSİ METRİKLERİ ÇIKARTILDI#########
"""
try:
    import tsfel

    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    tsfel_df = pd.DataFrame()
    # Extract features
    for i in df.columns:
        temp_df = df[i]
        X = tsfel.time_series_features_extractor(cfg, temp_df)
        X['sube'] = i

        tsfel_df = tsfel_df.append(X, ignore_index=True)
except Exception as e:
    print('ERR#10: Şubelerin zaman serisi öznitelik çıkarımında (TSFEL) hata oluştu')

tsfel_df.to_excel('tsfel_df.xlsx')
#################################################################
"""
