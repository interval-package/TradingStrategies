from GetWaveLetInfo import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import time


class tempData(object):
    def __init__(self, item):
        self.data = myData.ReadAndProcessData(item)[0]
        self.gp = getSlicedGrowthPerData(Item=item)
        # self.ARIMA = myARIMA(self.data)
        # self.model = self.ARIMA.ARIMAFit()
        pass

def clusteringPeriod(inputs, batch_size=45, n_clusters=5):
    # #############################################################################
    # k means
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
    t0 = time.time()
    k_means.fit(inputs)
    t_batch = time.time() - t0

    # print('cluster_centers of k_means:\n', k_means.cluster_centers_)

    # #############################################################################
    # Compute clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )
    t0 = time.time()
    mbk.fit(inputs)
    t_mini_batch = time.time() - t0

    # print('cluster_centers of MiniBatchKMeans:\n', mbk.cluster_centers_)

    return k_means, t_batch, mbk, t_mini_batch


def getKMeanRes(k_means, mbk, inputs):
    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = k_means.cluster_centers_
    # order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
    # mbk_means_cluster_centers = mbk.cluster_centers_[order]

    # 这里就是使用pairwise_distances_argmin，来进行簇分类了，很妙
    k_means_labels = pairwise_distances_argmin(inputs, k_means_cluster_centers)
    # mbk_means_labels = pairwise_distances_argmin(inputs, mbk_means_cluster_centers)
    return k_means_labels


def getResult(inputs, k_means_cluster_centers):
    return pairwise_distances_argmin(inputs, k_means_cluster_centers)


def clusterPloting(inputs, batch_size=45, n_clusters=5):
    k_means, t_batch, mbk, t_mini_batch = clusteringPeriod(inputs, batch_size, n_clusters)
    n_clusters = k_means.n_clusters
    # #############################################################################
    # Plot result

    # fig = plt.figure(figsize=(8, 3))
    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    # colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(inputs, k_means_cluster_centers)
    # ax = fig.add_subplot(1, 1, 1)
    # for k, col in zip(range(n_clusters), colors):
    #     my_members = k_means_labels == k
    #     cluster_center = k_means_cluster_centers[k]
    #     ax.plot(inputs[my_members, 0], inputs[my_members, 1], "w", markerfacecolor=col, marker=".")
    #     ax.plot(
    #         cluster_center[0],
    #         cluster_center[1],
    #         "o",
    #         markerfacecolor=col,
    #         markeredgecolor="k",
    #         markersize=6,
    #     )
    # ax.set_title("KMeans")
    # ax.set_xticks(())
    # ax.set_yticks(())
    # plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))
    #
    # plt.show()

    return inputs, k_means_cluster_centers, n_clusters


class KMeansFitter(object):
    def __init__(self):
        self.gp = tempData(0).gp
        pass

    # def getPeriod(self, start, end):
    #     return self.gp[start:en]

    def getPeriodGrowth(self, start, end):
        g1 = 1
        if end > len(self.gp):
            end = len(self.gp)
        for i in self.gp['value'][start:end]:
            g1 += abs(i)
        return g1

    def getData(self, start, step=20):
        obj = WaveFitter(self.gp[start:start + step])
        vec = obj.getCoef().ravel()
        return vec

    def connectData(self, gap, step=10):
        l = (len(self.gp) - 1)
        result = np.zeros((int(l / step), gap * 4))
        ids = []
        for i, idx in zip(range(0, l, step), range(0, int(l / step))):
            result[idx] = self.getData(i, gap)
            ids.append(i)
        return result, ids

    def Clustering(self, gap=20, step=20, n_clusters=10):
        # file = open('data/Centers.txt', mode='w')
        # print(time.time(), file=file)
        sample, ids = self.connectData(gap, step)
        # centers = clusterPloting(sample, n_clusters=9)[1]
        # print(centers, file=file)
        # 得到分类结果
        ce = getResult(sample, clusterPloting(sample, n_clusters=n_clusters)[1])
        # print(ce)
        # fr = pd.DataFrame(columns=['type', 'id'])
        # fr['type'] = ce
        # fr['id'] = ids
        ce = np.array(ce)
        ids = np.array(ids)
        # plt.subplot(1, 2, 1)
        # plt.bar(ids, ce+1)
        # # plt.plot(ids, ce+1)
        # # plt.vlines(ids, 0, ce * 0.1, 'r')
        # plt.subplot(1, 2, 2)
        # plt.bar(ids, ce+1)
        # plt.show()
        result = pd.DataFrame(columns=['type', 'mean', 'std'])
        for i in range(9):
            # print(i)
            temp = ids[ce == i]
            tempHolder = []
            for j in temp:
                tempHolder.append(self.getPeriodGrowth(j, j + gap))
            tempHolder = np.array(tempHolder, float)
            result.loc[i] = [int(i), tempHolder.mean(), tempHolder.std()]
            # print("item type : %d \nitem mean : %f \nitem std  : %f" % (i, tempHolder.mean(), tempHolder.std()))
        result.sort_values(by='mean', inplace=True)
        result.reset_index(inplace=True)

        def getgap(type):
            ob = result['type']
            agap = (ob[ob == type].index[0]-n_clusters+1).astype(int)
            agap = (gap/agap).astype(int)
            # agap = result[result.type == type].index.tolist()+1
            return agap

        outputs = []
        for i_type, i in zip(ce, ids):
            agap = getgap(i_type)
            for j in range((gap/agap).astype(int)):
                # print(i,j,i+j*agap)
                outputs.append(i+j*agap)

        return outputs
        # print(result.reset_index())
        # print(fr)
        # plt.subplot(1, 2, 1)
        # plt.plot(self.gp['value'])
        # # plt.vlines(ids, 0, ce * 0.1, 'r')
        # plt.subplot(1, 2, 2)
        # plt.bar(ids, ce)
        # plt.show()


if __name__ == '__main__':
    obj = KMeansFitter()
    q = obj.Clustering(50,50)
    print(q[len(q)-1])
    # print(obj.connectData(20,20).shape)
    # 设置预设区间 2,3,4,5,6,8,10,15,30,50
    pass
