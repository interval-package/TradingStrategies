from GetWaveLetInfo import *


def getSample(step=10):
    data = getSlicedGrowthPerData()['value'].values
    result = []
    for iter in range(len(data)-step):
        result.append(data[iter:iter+step])
    return result


def getWaveSample(obj):
    result = []
    for item in obj:
        result.append(WaveFitter.CWT(item)[2])
    return result


if __name__ == '__main__':
    sample = getSample()
    print(getWaveSample(sample))
