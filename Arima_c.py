import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from Arayuz import arayuz


class Arima(object):
    df = None
    sube = None
    model_a = None
    prediction =None
    X_test = None
    forecast = None
    train_prediction = None

    def __init__(self,sube):
        self.sube=sube
        plt.style.use('dark_background')

        # load the dataset
        self.df = pd.read_csv('modified.csv', sep=';', encoding='ISO:8859-1')
        self.df['Tarih'] = pd.to_datetime(self.df['Tarih'], infer_datetime_format=True)

        print(self.df.dtypes)

        self.df.set_index('Tarih', inplace=True)

    def test_pred(self):
        if self.model_a is not None:
            # Predicted and actual
            arayuz.plt.plot(self.prediction,label='Predicted')
            arayuz.plt.legend(loc='upper left')
            arayuz.plt.plot(self.X_test,label='Original Test Val')
            arayuz.plt.legend(loc='upper left')
            arayuz.plt.show()

    def model_sonuc(self):
        if self.model_a is not None:
            ##################
            arayuz.plt.plot(self.df[self.sube], label='Truth')
            arayuz.plt.plot(self.train_prediction, label='Train Pred', color='green')
            arayuz.plt.plot(self.prediction, label='Test', color='yellow')
            arayuz.plt.plot(self.forecast, label='Forecast', color='red')
            arayuz.plt.legend(loc='upper left')
            arayuz.plt.show()

    def auto_ar(self):
        sonuc_string = []
        sonuc_string.append("{}\nArima\n".format(self.sube))
        ####### Sube eger problemliyse knn sonuclarini kullan
        prob_sube = pd.read_csv('problemli_sube_komsu_oranlari_prtfy_v2.csv', sep=';')
        ##Problemli Subeler icin bu verileri kullanacaksın. BURADAN DEVAM ET.
        prob = pd.read_csv('problem_sube.csv', sep=';', encoding='ISO:8859-1')
        prob.drop([390, 391], inplace=True)

        list_sorunlu_sube = prob['Sube'].tolist()

        if self.sube in list_sorunlu_sube:
            sonuc_string.append("{} Verileri Sorunlu".format(self.sube) + "\nVeriler knn yöntemi ile dolduruldu.\n")
            print("{}. Sube Sorunlu".format(self.sube))
            df2 = pd.read_csv('problemli_sube_komsu_oranlari_prtfy_v2.csv',
                              sep=';')  # Sorunlu subelerin düzenlenmis dönem mevduat hedefleri.)
            df2['Tarih'] = pd.to_datetime(df2['Tarih'], infer_datetime_format=True)
            df2.set_index('Tarih', inplace=True)
            self.df[self.sube] = df2[self.sube].copy()
        ####### Bitis

        # Is the data stationary?

        adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(self.df[self.sube])
        print("pvalue = ", pvalue, " if above 0.05, data is not stationary")

        ####
        # Split data
        size = int(len(self.df[self.sube]) * 0.8)
        X_train, X_test = self.df[self.sube][0:size], self.df[self.sube][size:len(self.df[self.sube])]
        ####
        # Since data is not stationary, we may need SARIMA and not just ARIMA

        # We can go through the exercise of making the data stationary and performing ARIMA
        # Or let auto_arima provide the best model (e.g. SARIMA) and parameters.
        # Auto arima suggests best model and parameters based on
        # AIC metric (relative quality of statistical models)

        # Autoarima gives us bet model suited for the data
        # p - number of autoregressive terms (AR)
        # q - Number of moving avergae terms (MA)
        # d - number of non-seasonal differences
        # p, d, q represent non-seasonal components
        # P, D, Q represent seasonal components
        arima_model = auto_arima(X_train, start_p=1, d=1, start_q=1,
                                 max_p=6, max_q=6, max_d=6, m=12,
                                 start_P=0, D=1, start_Q=0, max_P=6, max_D=6, max_Q=6,
                                 seasonal=True,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True, n_fits=100,
                                 maxiter=100)

        # print the summary
        print(arima_model.summary())
        # Model: SARIMAX(1,1,0)x(1, 1, 0, 12)

        # SARIMAX on training set

        model = SARIMAX(X_train,
                        order=arima_model.order,
                        seasonal_order=arima_model.seasonal_order)

        result = model.fit()
        result.summary()

        self.model_a = model

        # Train prediction
        start_index = 0
        end_index = len(X_train) - 1
        train_prediction = result.predict(start_index, end_index)

        # Prediction
        start_index = len(X_train)
        end_index = len(self.df[self.sube]) - 1
        prediction = result.predict(start_index, end_index).rename('Predicted values')
        # Rename the column


        #Plot icin attribute tanımı
        self.prediction = prediction
        self.X_test=X_test

        # mean squared error
        trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
        sonuc_string.append(('Train Score: %.2f RMSE\n' % (trainScore)))
        testScore = math.sqrt(mean_squared_error(X_test, prediction))
        sonuc_string.append(('Test Score: %.2f RMSE\n' % (testScore)))

        # MAPE
        trainMape = mean_absolute_percentage_error(X_train, train_prediction)
        testMape = mean_absolute_percentage_error(X_test, prediction)

        sonuc_string.append(('Train MAPE: %.2f\n' % (trainMape)))
        sonuc_string.append(('Test MAPE: %.2f\n' % (testMape)))

        # Forecast for the next 4 season
        forecast = result.predict(start=len(self.df),
                                  end=(len(self.df) - 1) + 5,
                                  typ='levels').rename('Forecast')
        self.forecast=forecast
        self.train_prediction=train_prediction
        #FORECAST SONUCUNU GÖSTER
        return sonuc_string



# Grafik(ihtiyac halinde kullan)
"""plt.figure(figsize=(12, 8))
plt.plot(X_train, label='Training', color='green')
plt.plot(X_test, label='Test', color='yellow')
plt.plot(forecast, label='Forecast', color='cyan')
plt.legend(loc='upper left')
plt.show()"""

"""
###########################################
###########################################
#####SUBE_HATALARI#########################
subeler=df.columns.values
try:
    f = open("ArimaSonuc.txt", 'w')  # write in text mode
    f.write("ARIMA SUBE MODEL HATALARI\n")
    for sube in subeler:

        f.write("####### {} #######\n".format(sube))
        from pmdarima.arima import auto_arima

        arima_model = auto_arima(df[sube], start_p=1, d=1, start_q=1,
                                 max_p=6, max_q=6, max_d=6, m=12,
                                 start_P=0, D=1, start_Q=0, max_P=6, max_D=6, max_Q=6,
                                 seasonal=True,
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True, n_fits=100)

        #print(arima_model.summary())

        size = int(len(df[sube]) * 0.8)
        X_train, X_test = df[sube][0:size], df[sube][size:len(df[sube])]

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(X_train,
                        order=arima_model.order,
                        seasonal_order=arima_model.seasonal_order)

        result = model.fit()
        #result.summary()
        # Train prediction
        start_index = 0
        end_index = len(X_train) - 1
        train_prediction = result.predict(start_index, end_index)

        # Prediction
        start_index = len(X_train)
        end_index = len(df[sube]) - 1
        prediction = result.predict(start_index, end_index).rename('Predicted values')

        # mean squared error
        trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
        f.write('Train Score: %.2f RMSE\n' % (trainScore))
        testScore = math.sqrt(mean_squared_error(X_test, prediction))
        f.write('Test Score: %.2f RMSE\n' % (testScore))

        # MAPE
        trainMape = mean_absolute_percentage_error(X_train, train_prediction)
        testMape = mean_absolute_percentage_error(X_test, prediction)

        f.write('Train MAPE: %.2f\n' % (trainMape))
        f.write('Test MAPE: %.2f\n' % (testMape))

        print("\n#################\n")
        print("{} - %.2f - %.2f".format(sube) % (trainMape,testMape))
        print("\n#################\n")


    f.close()
except :
    print("Error")
    exit()
"""

##############Sonuc analizi################
##############Problemli Subelerin Cikarimi########
"""
f = open('ArimaSonuc.txt', 'r')
print(f.readline())
sube_sayisi = len(df.columns)

problem_sube = pd.DataFrame(columns=['Sube','Mape'])

for i in range(sube_sayisi):
    sube = f.readline()
    sube = sube.split(" ")[1]
    f.readline()
    f.readline()
    f.readline()
    mape = f.readline()
    mape = mape.split(" ")[2].split("\n")[0]
    mape = float(mape)
    if mape>0.15:#Mapesi %15ten fazla olan subeleri kaydeder.
        temp = {'Sube': sube, 'Mape': mape}
        problem_sube=problem_sube.append(temp,ignore_index=True)

problem_sube.to_csv('problem_sube.csv', index=False, sep=';',encoding='ISO:8859-1')
############################################################
"""

##################################Problemli Subeler icin Yeni Set######################################################
#######################################################################################################################
# Problemli subeler icin tsfel metriklerine göre en yakın 5 komşusunu bulup, problemli şubelerde
# her dönem için komşularının dönem ortalamasını alır. Ve komşularının mevduat değerlerine göre hesaplama yapılır.
"""prob = pd.read_csv('problem_sube.csv',sep=';',encoding='ISO:8859-1')
prob.drop([390,391], inplace=True)

list_sorunlu_sube = prob['Sube'].tolist()

tsfel_df = pd.read_excel('tsfel_df.xlsx')

sorunsuz_df = tsfel_df[~tsfel_df['sube'].isin(list_sorunlu_sube)]

# kaç yakın şubeye bakılacağı, k sayısı + 1 olarak girilmeli, şubenin kendisi çıkarılacak
k = 6

from sklearn.neighbors import NearestNeighbors
import numpy as np

from sklearn.preprocessing import MinMaxScaler

prob_sube = pd.DataFrame()
komsu_df = pd.DataFrame()
# sorunlu şubeler arasından eğitim kümesine en yakın olan şubeler bulunacak (k adet)
# knn algoritması kullanılacak


for i in list_sorunlu_sube:
    temp = tsfel_df[tsfel_df['sube'] == i]
    temp_df = sorunsuz_df.append(temp, ignore_index=True)
    temp_df.index = temp_df['sube']
    temp_df = temp_df.drop(['sube'], axis=1)

    scaler = MinMaxScaler()
    temp_df = pd.DataFrame(scaler.fit_transform(temp_df), columns=temp_df.columns,
                           index=temp_df.index)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(temp_df)
    distances, indices = nbrs.kneighbors(temp_df.iloc[-1:, :])
    print('-------------' + str(i))
    print(distances)

    indices = indices

    komsular = temp_df.iloc[indices.tolist()[0][1:]].index.tolist()
    print(komsular)

    komsu_df[str(i) + '_subeler'] = pd.DataFrame(komsular)
    komsu_df[str(i) + '_mesafe'] = pd.DataFrame(distances.tolist()[0])

    komsu_oranlar = df[komsular]

    komsu_oranlar = komsu_oranlar.mean(axis=1)

    prob_sube[i] = komsu_oranlar

komsu_df.to_csv('komsu_mesafe_prtfy_v2.csv', sep=';')#Her sorunlu sube icin 5 yakın komsu ve oranları.
prob_sube.to_csv('problemli_sube_komsu_oranlari_prtfy_v2.csv', sep=';')#Sorunlu subelerin düzenlenmis dönem mevduat hedefleri.
"""
