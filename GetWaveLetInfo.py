import pywt
from myData import *
from scipy import signal


def CWT(data, fs=25600, wavename="morl"):
    t = np.arange(0, len(data)) / fs
    # wavename = "cgau8"   # cgau8 小波
    # wavename = "morl"  # morlet 小波
    # wavename = "cmor3-3"  # cmor 小波

    totalscale = 256
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换

    # print("min(frequencies):", min(frequencies))
    # print("max(frequencies):", max(frequencies))

    return t, frequencies, cwtmatr


def myFilter(data):
    b, a = signal.butter(1, 0.8, btype='lowpass', output='ba')
    # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  # data为要过滤的信号
    return filtedData


class WaveFitter(object):
    def __init__(self, data: pd.DataFrame):
        # self.obj = pywt.WaveletPacket(data=data['value'].values, wavelet='db1', mode='symmetric')
        self.Date = data['Date']
        # self.v = data['value'].values
        self.Approximation, self.coefficients = pywt.dwt(data['value'], wavelet='bior6.8')
        self.t, self.frequencies, self.cwtmatr = CWT(data['value'].values, 1)
        pass

    def CWTshow(self):
        # data = self.v
        t, frequencies, cwtmatr = self.t, self.frequencies, self.cwtmatr
        plt.figure(figsize=(12, 6))
        # plt.plot(t, data)
        # plt.xlabel("Time(s)", fontsize=14)
        # plt.ylabel("Amplitude(g)", fontsize=14)

        t, frequencies_mesh = np.meshgrid(t, frequencies)
        fig = plt.figure()

        # ax1 = fig.add_subplot(111, projection='3d')
        # ax1.contourf(t, frequencies_mesh, cwtmatr, cmap=plt.cm.hot)  # 画等高线图
        # plt.xlabel("Time(s)", fontsize=14)
        # plt.ylabel("Frequency(Hz)", fontsize=14)

        # ax2 = fig.add_subplot(222, projection='3d')
        # ax2.plot_wireframe(t, frequencies_mesh, cwtmatr, cmap=plt.cm.hot)
        # plt.xlabel("Time(s)", fontsize=14)
        # plt.ylabel("Frequency(Hz)", fontsize=14)

        ax3 = fig.add_subplot(111, projection='3d')
        ax3.plot_surface(t, frequencies_mesh, cwtmatr, cmap=plt.cm.hot)
        plt.xlabel("Time(s)", fontsize=14)
        plt.ylabel("Frequency(Hz)", fontsize=14)
        plt.show()

    def disp(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.Approximation)
        plt.subplot(2, 1, 2)
        plt.plot(self.coefficients)
        plt.show()
        pass

    def getCoef(self):
        def findPeaks(item):
            f = self.frequencies[::-1]
            df = np.diff(item, append=0)
            # print(item)
            thresh = 6e-3
            bi = np.bitwise_and(df < thresh, df > -thresh)
            # print(f.shape, bi.shape)
            return f[bi], item[bi]

        mat = self.cwtmatr
        result = np.zeros((mat.shape[1],4))
        for i, idx in zip(mat.T, range(mat.shape[1])):
            # print(idx)
            # plt.subplot(1, 2, 1)
            f, v = findPeaks(i)
            result[idx] = np.polyfit(f, v, 3)

        return result

    def CoefShow(self):
        Coef = self.getCoef()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = np.arange(0, Coef.shape[1], 1)
        y = np.arange(0, Coef.shape[0], 1)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, Coef, rstride=1,  # row 行步长
                        cstride=2,  # colum 列步长
                        cmap=plt.cm.hot)  # 渐变颜色
        ax.set_xlabel("coef")
        ax.set_ylabel("time")
        # ax.label
        # ax.label("time")
        plt.show()

    def matAna(self):
        mat = self.cwtmatr
        f = self.frequencies[::-1]
        print(f)
        print(mat.shape)

        for i, idx in zip(mat.T, range(9)):
            plt.subplot(3, 3, idx + 1)
            plt.plot(f, i, ), plt.xlabel('frequencies')
        plt.show()

        def findPeaks(item):
            f = self.frequencies[::-1]
            df = np.diff(item, append=0)
            # print(item)
            thresh = 6e-3
            bi = np.bitwise_and(df < thresh, df > -thresh)
            print(f.shape, bi.shape)
            return f[bi], item[bi]

        for i, idx in zip(mat.T, range(9)):
            # print(idx)
            # plt.subplot(1, 2, 1)
            plt.subplot(3, 3, idx + 1)
            f, v = findPeaks(i)
            plt.plot(f, myFilter(v), ), plt.xlabel('frequencies')
            # plt.plot(myFilter(i-min(i)))
            # Approximation, coefficients = pywt.dwt(i, wavelet='bior6.8')
            # plt.subplot(2, 2, 2)
            # plt.plot(Approximation)
            # plt.subplot(2, 2, 4)
            # plt.plot(coefficients)
        plt.show()

    def show(self):
        fig = plt.figure('WaveLet')
        ax = Axes3D(fig)
        x, y = np.meshgrid(self.Date.values, np.linspace(-self.Approximation.max(), self.Approximation.max(), 100))
        ax.plot(x, y, np.linspace(0, 1, len(x)), '.')
        plt.show()
        pass


def normShow():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.arange(-4, 4, 0.25)
    y = np.arange(-4, 4, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.sin(r)

    ax.plot_surface(x, y, z, rstride=1,  # row 行步长
                    cstride=2,  # colum 列步长
                    cmap=plt.cm.hot)  # 渐变颜色
    ax.contourf(x, y, z,
                zdir='z',  # 使用数据方向
                offset=-2,  # 填充投影轮廓位置
                cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)

    plt.show()
    pass


if __name__ == '__main__':
    # normShow()
    data = myData.GetGrowthPercentage(myData.ReadAndProcessData()[0])[50:80]
    # plt.plot(data['Date'], data['value'])
    # plt.show()
    obj = WaveFitter(data)
    obj.CoefShow()
    print(obj.getCoef())
    # obj.disp()
    # obj.matAna()
    # obj.CWTshow()
    # obj.CWTshow()
    # obj.disp()

    pass
