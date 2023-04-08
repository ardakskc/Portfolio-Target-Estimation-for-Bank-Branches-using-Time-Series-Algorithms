from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Arima_c, NeuralNet_c, LSTM_c

df = pd.read_csv('modified.csv', sep=';', encoding='ISO:8859-1')
df['Tarih'] = pd.to_datetime(df['Tarih'], infer_datetime_format=True)

print(df.dtypes)

df.set_index('Tarih', inplace=True)
"""
subeler = df.columns.values
f = open("OptimizeSonuc_orj.txt", 'w')  # write in text mode
f.write("#########################################\n")
for selected_sube in subeler:

    a1 = Arima_c.Arima(selected_sube)
    sonuc_string1 = a1.auto_ar()
    mape1 = sonuc_string1[4].split(" ")[2].split("\n")[0]
    mape1 = float(mape1)

    a2 = NeuralNet_c.NNet(selected_sube)
    sonuc_string2 = a2.neural_net()
    mape2 = sonuc_string2[4].split(" ")[2].split("\n")[0]
    mape2 = float(mape2)

    a3 = LSTM_c.Lstm(selected_sube)
    sonuc_string3 = a3.lstm_model()
    mape3 = sonuc_string3[4].split(" ")[2].split("\n")[0]
    mape3 = float(mape3)

    if mape1 < mape2:
        if mape1 < mape3:
            a = a1
            sonuc_string = sonuc_string1
        else:
            a = a3
            sonuc_string = sonuc_string3
    else:
        if mape2 < mape3:
            a = a2
            sonuc_string = sonuc_string2
        else:
            a = a3
            sonuc_string = sonuc_string3
    sonuc_string.append("#########################################\n")
    s = ""
    s = s.join(sonuc_string)
    f.write(s)

f.close()"""


f = open('OptimizeSonuc.txt', 'r')
sube_sayisi = len(df.columns)

optimize = pd.DataFrame(columns=['Sube','Model','Train_Rmse','Test_Rmse','Train_Mape','Test_Mape'])

for i in range(sube_sayisi):
    f.readline()
    sube = f.readline()
    model = f.readline()

    train_rmse= f.readline().split(" ")[2]
    test_rmse = f.readline().split(" ")[2]
    train_mape = f.readline().split(" ")[2]
    test_mape = f.readline().split(" ")[2]

    train_rmse = float(train_rmse)
    test_rmse = float(test_rmse)
    train_mape = float(train_mape)
    test_mape = float(test_mape)

    temp = {'Sube': sube, 'Model': model, 'Train_Rmse': train_rmse, 'Test_Rmse': test_rmse, 'Train_Mape': train_mape, 'Test_Mape': test_mape}
    optimize = optimize.append(temp,ignore_index=True)

optimize.to_excel('optimize.xlsx', index=False,encoding='ISO:8859-1')
############################################################


