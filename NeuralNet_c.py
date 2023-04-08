
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from Arayuz import arayuz


class NNet(object):
    df = None
    sube = None
    model_a = None
    prediction = None
    X_test = None
    forecast = None
    train_prediction = None
    testPredictPlot = None
    forPredictPlot = None

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
            arayuz.plt.plot(self.df[self.sube].values, label='Truth')
            arayuz.plt.plot(self.train_prediction, label='Train Pred', color='green')
            arayuz.plt.plot(self.testPredictPlot, label='Test', color='yellow')
            arayuz.plt.plot(self.forPredictPlot, label='Forecast', color='red')
            arayuz.plt.legend(loc='upper left')
            arayuz.plt.show()

    #Data sekansı oluşturur ki sinir ağına sıralı veri verip sonraki veriyi tahmin edebilelim.
    #Creates a dataset where X is the number of past values at a given time (t, t-1, t-2...)
    #And Y is the truth prediction value at the next time (t + 1).
    #seq_size is a window size for prediction
    def to_sequences(self, dataset, seq_size=1):
        x = []
        y = []

        for i in range(len(dataset) - seq_size - 1):
            # print(i)
            window = dataset[i:(i + seq_size), 0]
            x.append(window)
            y.append(dataset[i + seq_size, 0])

        return np.array(x), np.array(y)

    def neural_net(self):
        sonuc_string = []
        sonuc_string.append("{}\nNeural Network\n".format(self.sube))

        ####### Sube eger problemliyse knn sonuclarini kullan
        prob_sube = pd.read_csv('NNproblemli_sube_komsu_oranlari_prtfy_v2.csv', sep=';')
        ##Problemli Subeler icin bu verileri kullanacaksın. BURADAN DEVAM ET.
        prob = pd.read_csv('problem_sube_NeuralNet.csv',sep=';',encoding='ISO:8859-1')
        prob = prob.drop(labels= 14 , axis=0)

        list_sorunlu_sube = prob['Sube'].tolist()

        if self.sube in list_sorunlu_sube:
            sonuc_string.append("{} Verileri Sorunlu".format(self.sube) + "\nVeriler knn yöntemi ile dolduruldu.\n")
            print("{}. Sube Sorunlu".format(self.sube))
            df2= pd.read_csv('NNproblemli_sube_komsu_oranlari_prtfy_v2.csv', sep=';')#Sorunlu subelerin düzenlenmis dönem mevduat hedefleri.)
            df2['Tarih'] = pd.to_datetime(df2['Tarih'], infer_datetime_format=True)
            df2.set_index('Tarih', inplace=True)
            self.df[self.sube]= df2[self.sube].copy()
        ####### Bitis

        dataset = self.df[self.sube].values

        dataset=dataset.reshape(-1,1)
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)


        #We cannot use random way of splitting dataset into train and test as
        #the sequence of events is important for time series
        #Train test Split
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        seq_size = 4 # Number of time steps to look back
        #Kaç adım geriye bakıp tahmin yapılacak. Veri seti için sequenceler oluşturulur.
        trainX, trainY = self.to_sequences(train, seq_size)
        testX, testY = self.to_sequences(test, seq_size)

        #Forecast Sequence hazırlama
        forc_size = 4
        forc= dataset[len(dataset)-4:len(dataset),:]
        forc=forc.reshape(1,-1)
        forcY=[]
        forcY= np.asarray(forcY)

        #Compare trainX and dataset. You can see that X= values at t, t+1 and t+2
        #whereas Y is the value that follows, t+3 (since our sequence size is 3)

        print("Shape of training set: {}".format(trainX.shape))
        print("Shape of test set: {}".format(testX.shape))

        #Input dimensions are... (N x seq_size)
        print('Build deep model...')
        # create and fit dense model
        model = Sequential()
        model.add(Dense(64, input_dim=seq_size, activation='relu')) #12
        model.add(Dense(32, activation='relu'))  #8
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics = ['acc'])
        print(model.summary())

        ########################
        model.fit(trainX, trainY, validation_data=(testX, testY),
                  verbose=2, epochs=150)

        self.model_a = model

        #######################

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        #######################
        #Forecast next 6 value
        forcY=np.append(forcY, model.predict(forc), axis=None)
        for j in range(4):
            for i in range(3):
                forc[0][i]=forc[0][i+1]
            forc[0][3]=forcY[-1]
            forcY=np.append(forcY, model.predict(forc), axis=None)
        forcY=forcY.reshape(-1,1)
        #######################

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_inverse = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY_inverse = scaler.inverse_transform([testY])
        forcY = scaler.inverse_transform(forcY)

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
        sonuc_string.append('Train Score: %.2f RMSE\n' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
        sonuc_string.append('Test Score: %.2f RMSE\n' % (testScore))

        #MAPE
        trainMape = mean_absolute_percentage_error(trainY_inverse[0], trainPredict[:,0])
        testMape = mean_absolute_percentage_error(testY_inverse[0], testPredict[:,0])
        sonuc_string.append('Train MAPE: %.2f\n' % (trainMape))
        sonuc_string.append('Test MAPE: %.2f\n' % (testMape))


        # shift train predictions for plotting
        #we must shift the predictions so that they align on the x-axis with the original dataset.
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict

        ##############################

        # shift forecast predictions for plotting
        forPredictPlot = np.empty_like(dataset)
        forPredictPlot[:, :] = np.nan
        forPredictPlot = np.append(forPredictPlot,testPredict[-1],axis=None)
        forPredictPlot = np.append(forPredictPlot,forcY,axis=None)
        forPredictPlot = forPredictPlot.reshape(-1,1)
        ##############################

        self.train_prediction = trainPredictPlot
        self.forPredictPlot = forPredictPlot
        self.testPredictPlot = testPredictPlot

        # Predicted and actual
        testY_inverse=testY_inverse.reshape(-1,1)
        testPredict = pd.DataFrame(testPredict)
        testY_inverse = pd.DataFrame(testY_inverse)

        self.prediction = testPredict
        self.X_test = testY_inverse
        self.forecast = forcY

        return sonuc_string



"""###########################################
###########################################
#####SUBE_HATALARI#########################
subeler = df.columns.values
try:
    f = open("NeuralNetSonuc.txt", 'w')  # write in text mode
    f.write("NEURALNET SUBE MODEL METRIKLERI\n")
    for sube in subeler:
        f.write("####### {} #######\n".format(sube))
        dataset = df[sube].values

        dataset = dataset.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        seq_size = 4
        # Kaç adım geriye bakıp tahmin yapılacak. Veri seti için sequenceler oluşturulur.
        trainX, trainY = to_sequences(train, seq_size)
        testX, testY = to_sequences(test, seq_size)

        model = Sequential()
        model.add(Dense(64, input_dim=seq_size, activation='relu'))  # 12
        model.add(Dense(32, activation='relu'))  # 8
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])
        print(model.summary())

        model.fit(trainX, trainY, validation_data=(testX, testY),
                  verbose=2, epochs=150)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_inverse = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY_inverse = scaler.inverse_transform([testY])



        # mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
        f.write('Train Score: %.2f RMSE\n' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
        f.write('Test Score: %.2f RMSE\n' % (testScore))

        # MAPE
        trainMape = mean_absolute_percentage_error(trainY_inverse[0], trainPredict[:, 0])
        testMape = mean_absolute_percentage_error(testY_inverse[0], testPredict[:, 0])

        f.write('Train MAPE: %.2f\n' % (trainMape))
        f.write('Test MAPE: %.2f\n' % (testMape))

        print("\n#################\n")
        print("{} - %.2f - %.2f".format(sube) % (trainMape, testMape))
        print("\n#################\n")

    f.close()
except:
    print("Error")
    exit()
"""


####HATASI YÜKSEK ŞUBELERI CIKAR.
##############Sonuc analizi################
##############Problemli Subelerin Cikarimi########
"""
f = open('NeuralNetSonuc.txt', 'r')
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

problem_sube.to_csv('problem_sube_NeuralNet.csv', index=False, sep=';',encoding='ISO:8859-1')
"""
############################################################
#############Problemli şubeler için knn uygula########
# Problemli subeler icin tsfel metriklerine göre en yakın 5 komşusunu bulup, problemli şubelerde
# her dönem için komşularının dönem ortalamasını alır. Ve komşularının mevduat değerlerine göre hesaplama yapılır.
"""
prob = pd.read_csv('problem_sube_NeuralNet.csv',sep=';',encoding='ISO:8859-1')
prob = prob.drop(labels= 14 , axis=0)

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

prob_sube.to_csv('NNproblemli_sube_komsu_oranlari_prtfy_v2.csv', sep=';')#Sorunlu subelerin düzenlenmis dönem mevduat hedefleri."""
##########################################################################