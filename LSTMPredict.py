import numpy
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
from keras.models import Sequential, load_model

from ARIMAPredict import myData


class LSTMmodel(object):
    @staticmethod
    def createDataset(dataset, look_back):
        # 这里的look_back与timestep相同
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return numpy.array(dataX), numpy.array(dataY)

    def __init__(self, Data: pd.DataFrame, lookBack=3):
        self.dataX, self.dataY = self.createDataset(Data, lookBack)
        self.modelPath = os.path.join("data", "Test" + ".h5")
        pass
    
    @staticmethod
    def dataX2TrainData(dataX):
        return numpy.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    def LSTMFitting(self):
        trainX = self.dataX2TrainData(self.dataX)

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(None, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, self.dataY, epochs=100, batch_size=1, verbose=2)
        model.save(self.modelPath)
        # make predictions
        return model

    @staticmethod
    def loadModel(path):
        model = load_model()
        return model

    def LSTMpredicting(self, predata, ReadMode=False):
        if ReadMode:
            model = load_model(self.modelPath)
        else:
            model = self.LSTMFitting()
        trainPredict = model.predict(predata)
        return trainPredict

    def show(self, ReadMode=False):
        self.LSTMpredicting(self.dataX2TrainData(self.dataX))
        plt.figure("LSTM")
        plt.subplot(3, 1, 1)
        plt.plot(self.dataY)
        plt.subplot(3, 1, 2)
        plt.plot(self.dataY)
        plt.show()
        pass


if __name__ == '__main__':
    data, vflag = myData.ReadAndProcessData()

    lstmObj = LSTMmodel(data, 5)
    lstmObj.show()
    pass
