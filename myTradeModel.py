import math

from myData import *
from getSwapData import optimMyTarget, funcGetNextDay
from GetWaveLetInfo import *
from ARIMAPredict import *
from clustering import *


class Investor(object):
    def __init__(self, item=0):
        self.data = myData.ReadAndProcessData(item)[0]
        self.GrowthPer = myData.GetGrowthPercentage(self.data).set_index('Date')
        self.main()
        pass

    def getPeriodGrowthPercentSum(self, period=None):
        if period is None:
            period = self.periodGetter(0, len(self.data['value']) - 1)
        start, end = period
        # 使用环比增
        totalGrowth = self.GrowthPer[start:end]['value'].values
        return totalGrowth.sum()

    def periodGetter(self, start, end):
        return [self.data['Date'][start], self.data['Date'][end]]

    def main(self):
        print(self.getPeriodGrowthPercentSum())
        pass


class tempData(object):
    def __init__(self, item):
        self.data = myData.ReadAndProcessData(item)[0]
        self.gp = getSlicedGrowthPerData(Item=item)
        # self.ARIMA = myARIMA(self.data)
        # self.model = self.ARIMA.ARIMAFit()
        pass

    def getDate(self):
        return self.data['Date']


class BestInvest(object):
    def __init__(self):
        self.ms = [100, 100, 800]
        self.bound = [1000, 1000, 1000]
        self.cm = [0.02, 0.01, 0]

        self.bitData = tempData(0)
        self.goldData = tempData(1)

        self.ttlen = len(self.bitData.data)
        self.date = self.bitData.getDate()

        self.data = self.connect()

        print(self.getPeriodGrowth(0, 20))
        # self.main()
        pass

    def periodGetter(self, start, end):
        return [self.date[start], self.date[end]]

    def getPeriodGrowth(self, start, end):
        g1, g2 = 1, 1
        if end > len(self.data):
            end = len(self.data)
        for i, j in zip(self.data['bit'][start:end], self.data['gold'][start:end]):
            g1 *= (i + 1)
            g2 *= (j + 1)
        return g1 - 1, g2 - 1

    def waveInfoAna(self):

        pass

    def getInvestPeriod(self):
        result = []
        return result

    def connect(self):
        _obj = self.bitData.gp.rename(columns={'value': 'bit'}).set_index('Date')
        _obj = _obj.join(self.goldData.gp.rename(columns={'value': 'gold'}).set_index('Date'))
        # 认为周末不增长
        return _obj.fillna(value=0).reset_index()

    def GreedyMethod(self, passWindow=5, showflag=False):
        # print(self.goldData.model.fittedvalues)
        # input("next:")
        # goldFit = self.goldData.model.predict(self.date[0:-1].values)
        bound, cm, ms = self.bound, self.cm, self.ms
        ttData = self.data
        result, bitRe, goldRe, cashRe = [], [], [], []
        transReBit, transReGold, dateRe = [], [], []
        investTime = []

        for bit, gold, date, idx in zip(ttData['bit'], ttData['gold'], ttData['Date'], range(len(ttData))):
            # print(bit,gold)
            if idx % passWindow == 0:
                if datetime.isoweekday(date) < 6:
                    bound = self.bound
                    pass
                else:
                    # 通过上界约束
                    bound[1] = 0
                    pass
                _bit, _gold = self.getPeriodGrowth(idx, idx+passWindow)
                trans = optimMyTarget(ms, [_bit, _gold, 0], bound, cm).tolist()
                investTime.append(date)
                transReBit.append(-trans[0] * (1 + cm[0]) + trans[1] * (1 - cm[0]))
                transReGold.append(-trans[2] * (1 + cm[2]) + trans[3] * (1 - cm[2]))
            else:
                trans = [0, 0, 0, 0, 0, 0]
            ms = funcGetNextDay(trans, ms, [bit, gold], cm)
            # print(trans)
            dateRe.append(date)
            result.append(sum(ms))

            bitRe.append(ms[0])
            goldRe.append(ms[1])
            cashRe.append(ms[2])
        print("""
        最终结果：%f
        比特币：%f
        黄金：%f
        现金：%f
        """ % (sum(ms),ms[0],ms[1],ms[2]))
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(investTime, transReBit), plt.title('transReBit')
        plt.subplot(1, 2, 2)
        plt.plot(investTime, transReGold), plt.title('transReGold')
        plt.savefig('transReport-window' + str(passWindow) + '.png')

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1), plt.plot(dateRe, bitRe), \
            # plt.vlines(investTime, 0, max(bitRe))
        plt.title("bitcoin")
        plt.subplot(2, 2, 2), plt.plot(dateRe, goldRe), plt.title('Gold')
        plt.subplot(2, 2, 3), plt.plot(dateRe, cashRe), plt.title('Cash')
        plt.subplot(2, 2, 4), plt.plot(dateRe, result), plt.title('sum')
        plt.savefig('cap-window' + str(passWindow) + '.png')

        if showflag:
            plt.show()
        pass

    def windowsPredict(self, passWindow=5, showflag=False):
        # print(self.goldData.model.fittedvalues)
        # input("next:")
        # goldFit = self.goldData.model.predict(self.date[0:-1].values)

        obj = KMeansFitter()
        InvestDate = obj.Clustering()
        count = 0

        bound, cm, ms = self.bound, self.cm, self.ms
        ttData = self.data
        result, bitRe, goldRe, cashRe = [], [], [], []
        transReBit, transReGold, dateRe = [], [], []
        investTime = []

        for bit, gold, date, idx in zip(ttData['bit'], ttData['gold'], ttData['Date'], range(len(ttData))):
            # print(bit,gold)
            if idx in InvestDate:
                count += 1
                if datetime.isoweekday(date) < 6:
                    bound = self.bound
                    pass
                else:
                    # 通过上界约束
                    bound[1] = 0
                    pass

                try:
                    tempId = InvestDate[count]
                except :
                    tempId = 100000
                _bit, _gold = self.getPeriodGrowth(idx, tempId)
                trans = optimMyTarget(ms, [_bit, _gold, 0], bound, cm).tolist()
                investTime.append(date)
                transReBit.append(-trans[0] * (1 + cm[0]) + trans[1] * (1 - cm[0]))
                transReGold.append(-trans[2] * (1 + cm[2]) + trans[3] * (1 - cm[2]))
            else:
                trans = [0, 0, 0, 0, 0, 0]
            ms = funcGetNextDay(trans, ms, [bit, gold], cm)
            # print(trans)
            dateRe.append(date)
            result.append(sum(ms))

            bitRe.append(ms[0])
            goldRe.append(ms[1])
            cashRe.append(ms[2])
        print("""
        最终结果：%f
        比特币：%f
        黄金：%f
        现金：%f
        """ % (sum(ms),ms[0],ms[1],ms[2]))
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(investTime, transReBit), plt.title('transReBit')
        plt.subplot(1, 2, 2)
        plt.plot(investTime, transReGold), plt.title('transReGold')
        plt.savefig('transReport-window' + str(passWindow) + '.png')

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1), plt.plot(dateRe, bitRe), \
            plt.vlines(investTime, 0, max(bitRe))
        plt.title("bitcoin")
        plt.subplot(2, 2, 2), plt.plot(dateRe, goldRe), plt.title('Gold')
        plt.subplot(2, 2, 3), plt.plot(dateRe, cashRe), plt.title('Cash')
        plt.subplot(2, 2, 4), plt.plot(dateRe, result), plt.title('sum')
        # plt.savefig('cap-window' + str(passWindow) + '.png')

        if showflag:
            plt.show()
        pass


if __name__ == '__main__':
    # Investor(0)
    # for i in [2, 3, 4, 5, 6, 7, 8]:
    #     BestInvest().GreedyMethod(i,True)
    #     input("next:")
    BestInvest().windowsPredict(2,True)
    # BestInvest().GreedyMethod(1, True)
    pass
