import math
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.tsa.stattools import adfuller  # adf检验

from pandas.plotting import autocorrelation_plot
# import mpl_toolkits.mplot3d

import statsmodels.graphics.tsaplots as tsaplots
# import plot_acf, plot_pacf
# acf,pacf图


class myData(object):
    @staticmethod
    def str2Datetime(st: str, year=1900):
        return datetime.strptime(str(year) + "/" + st, "%Y/%m/%d/%H")

    @staticmethod
    def ReadAndProcessData(inputItem=0):
        if inputItem == 0:
            path = "data/bitcoin.csv"
            flag = 'Value'
        else:
            path = "data/gold.csv"
            flag = 'USD (PM)'
        data = pd.read_csv(path)
        data.rename(columns={flag: 'value'}, inplace=True)
        # data['Date'] = data['Date'].apply(str2Datetime)
        temp = datetime(1900, 1, 1)
        baseyear = 2016
        for item, idx in zip(data['Date'], range(len(data['Date']))):
            obj = myData.str2Datetime(item, year=baseyear)
            if (temp.month == 12) & (obj.month == 1):
                baseyear += 1
                obj = myData.str2Datetime(item, year=baseyear)
            data.loc[idx, 'Date'] = obj
            temp = data['Date'][idx]
        return data, 'value'

    @staticmethod
    def GetGrowthPercentage(data: pd.DataFrame):
        df = data['value'].diff()[1:].values
        growthPer = np.array(df / data['value'][0:-1].values)
        result = pd.DataFrame(columns=['Date', 'value'])
        result['Date'] = data['Date'][1:].values
        result['value'] = growthPer
        return result


def getSlicedGrowthPerData(start=0, end=0, Item=None):
    if Item is None:
        Item = 0
    temp = myData.ReadAndProcessData(Item)[0]
    if end <= start or end >= len(temp):
        end = len(temp)-1
    return myData.GetGrowthPercentage(temp)[start:end]


class DataQualifier(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        pass

    def AcfPloter(self):
        tsaplots.plot_acf(self.data.dropna(), auto_ylims=True).show()
        plt.show()
        pass

    def PacfPloter(self):
        tsaplots.plot_pacf(self.data.dropna(), auto_ylims=True).show()
        plt.show()
        pass

    def ADFGetter(self):
        adfResult = adfuller(self.data.dropna())
        output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                     "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                              columns=['value'])
        output['value']['Test Statistic Value'] = adfResult[0]
        output['value']['p-value'] = adfResult[1]
        output['value']['Lags Used'] = adfResult[2]
        output['value']['Number of Observations Used'] = adfResult[3]
        output['value']['Critical Value(1%)'] = adfResult[4]['1%']
        output['value']['Critical Value(5%)'] = adfResult[4]['5%']
        output['value']['Critical Value(10%)'] = adfResult[4]['10%']
        return output


class TurningPoint(object):
    def __init__(self, date, growth_ahead, growth_after):
        pass


class TurningPointDetector(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @staticmethod
    def detecting(df):
        result = []
        for iter in range(len(df)-1):
            if df[iter]*df[iter+1] <= 0:
                result.append(iter+1)
        return result


def getInvestPeriod(data):
    print([int(math.exp(i / 10) - 1) for i in range(20, 76)])
    item = np.array([int(math.exp(i / 10) - 1) for i in range(25, 76)])
    item = -item + item.max()
    result = np.array(data['Date'].values)[item]
    return result


if __name__ == '__main__':
    # gp 二阶差分
    # data = getSlicedGrowthPerData(Item=1)
    # obj = DataQualifier(data['value'].diff().diff())
    # print(obj.ADFGetter())

    data = myData.ReadAndProcessData(0)[0]
    plt.plot(data["Date"].values, data["value"].values)



    plt.vlines(getInvestPeriod(data),0, 10000)
    # pos = plt.ginput()
    # print(pos)
    plt.show()
    # ids = TurningPointDetector.detecting(data['value'].values)
    # print(ids)
    #
    # points = data[ids]
    # print(points)

    pass
