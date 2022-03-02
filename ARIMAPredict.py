import itertools
import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

from myData import *

from statsmodels.tsa.arima.model import ARIMA


class myARIMA(object):
    def __init__(self, data, vFlag='value'):
        self.data = data
        self.v = data[vFlag]
        # self.target =self.ARIMAFit()
        pass

    def ARIMAParaGet(self, pmax=3, qmax=3):
        bic_matrix = []
        for p in range(pmax + 1):
            temp = []
            for q in range(qmax + 1):
                try:
                    temp.append(ARIMA(self.v, np.zeros(self.v.size), (p, 1, q)).fit().bic)
                except:
                    print("skip")
                    temp.append(None)
                bic_matrix.append(temp)

        bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe 数据结构
        p, q = bic_matrix.stack().idxmin()  # 先使用stack 展平， 然后使用 idxmin 找出最小值的位置
        print(u'BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1
        return p, q

    def pdqHeatMap(self, p_max, d_max, q_max, p_min=1, d_min=1, q_min=1):
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                                   columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
        for p, d, q in itertools.product(range(p_min, p_max + 1),
                                         range(d_min, d_max + 1),
                                         range(q_min, q_max + 1)):
            if p == 0 and d == 0 and q == 0:
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue

            try:
                model = sm.tsa.ARIMA(self.v, order=(p, d, q),
                                     # enforce_stationarity=False,
                                     # enforce_invertibility=False,
                                     )
                results = model.fit()
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
            except:
                continue
        results_bic = results_bic[results_bic.columns].astype(float)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(results_bic,
                         mask=results_bic.isnull(),
                         ax=ax,
                         annot=True,
                         fmt='.2f',
                         )
        ax.set_title('BIC')
        plt.show()
        return results_bic

    def ARIMAParaGet_2(self):
        train_results = sm.tsa.arma_order_select_ic(self.v, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

        print('AIC', train_results.aic_min_order)
        print('BIC', train_results.bic_min_order)

    def ARIMAFit(self, order=(5, 2, 5)):
        data = self.v
        model = ARIMA(data, np.zeros(data.size), order=order).fit()
        print(model.summary())  # 生成一份模型报告
        return model

    def ARIMAPredic(self, p=5, q=5):
        # p, q = self.ARIMAParaGet(5, 5)
        # 30, 5
        model = self.ARIMAFit((p, 2, q))
        data = self.v[0:-1].values
        timedata = self.data['Date'][0:-1].values
        predictions_ARIMA = pd.Series(model.fittedvalues, copy=True)
        predictions_ARIMA = predictions_ARIMA[1:].values
        print("predictions_ARIMA:\n", predictions_ARIMA)
        print("raw:\n", data)
        return model, data, timedata, predictions_ARIMA

    def test(self):
        model, data, timedata, predictions_ARIMA = self.ARIMAPredic()
        resid = model.resid  # 赋值
        fig = plt.figure(figsize=(12, 8))
        fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
        plt.show()
        pass

    def show(self):
        model, data, timedata, predictions_ARIMA = self.ARIMAPredic()

        try:
            plt.figure(figsize=(10, 6))
            plt.subplot(4, 1, 1)
            plt.plot(timedata, predictions_ARIMA, label="forecast")
            plt.plot(timedata, data, label="raw")
            plt.xlabel('date', fontsize=12, verticalalignment='top')
            plt.ylabel('price', fontsize=14, horizontalalignment='center')
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.plot(timedata, predictions_ARIMA - data, label="difference")
            plt.xlabel('date', fontsize=12, verticalalignment='top')
            plt.ylabel('difference level', fontsize=14, horizontalalignment='center')
            plt.subplot(4, 1, 3)
            plt.plot(timedata[0:-1], np.diff(predictions_ARIMA), label="forecast")
            plt.plot(timedata[0:-1], np.diff(data), label="raw")
            plt.xlabel('date', fontsize=12, verticalalignment='top')
            plt.ylabel('price df', fontsize=14, horizontalalignment='center')
            plt.legend()
            plt.subplot(4, 1, 4)
            plt.plot(timedata[0:-1], np.diff(predictions_ARIMA) - np.diff(data), label="difference")
            plt.xlabel('date', fontsize=12, verticalalignment='top')
            plt.ylabel('difference level', fontsize=14, horizontalalignment='center')
        except Exception as e:
            print("fail\n", repr(e))
        plt.show()


def ARIMAperiod(data):
    obj_1 = DataQualifier()
    print(obj_1.ADFGetter())
    input("stop:")
    obj_1.AcfPloter()
    obj_1.PacfPloter()
    input("stop:")

    obj = myARIMA(data=data, vFlag='value')
    # obj.ARIMAParaGet_2()
    # obj.pdqHeatMap(5, 2, 5)
    obj.test()
    obj.show()
    pass


def main():
    data = getSlicedGrowthPerData(Item=1)
    data = myData.ReadAndProcessData(1)[0]
    print(data)
    obj_1 = DataQualifier(data['value'])
    print(obj_1.ADFGetter())
    input("stop:")
    obj_1.AcfPloter()
    obj_1.PacfPloter()
    obj_2 = myARIMA(data)
    # obj_2.pdqHeatMap(30, 3, 6, 28, 2, 5)
    # obj_2.ARIMAParaGet_2()
    obj_2.test()
    obj_2.show()
    pass


if __name__ == '__main__':
    main()
    # input()
    # for i in range(2):
    #     data = myData.ReadAndProcessData(i)[0]
    #     data = myData.GetGrowthPercentage(data=data)
    #     ARIMAperiod(data)
    #     input('next: ')
