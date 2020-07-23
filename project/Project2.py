import csv

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

out_folder = "../out"
figname_prefix = ""


def loadData():
    dataPath = "../data/tourist.csv"
    df = pd.read_csv(dataPath, header=None, index_col=None, encoding="gbk")
    return pd.DataFrame(df.values[:, 1:].T, columns=df[0])


def decompose(df):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(df.values.T)
    return pca, result


def kmeansAndCost(df, k):
    costs = []
    for n in range(0, 10):
        kmeans = KMeans(n_clusters=(n + 1))
        # kmeans.labels_ = labels
        kmeans.fit(df.values.T)
        costs.append(kmeans.inertia_)
    plt.figure()
    x = range(1, 11)
    plt.plot(x, costs)
    plt.xlabel('Number of clusters')
    plt.ylabel('Total sum of square')
    plt.axvline(x=k, c="black", linestyle="dashed")
    plt.title('Optimal number of clusters')
    plt.savefig(f"{out_folder}/{figname_prefix}k-Means聚类折线图.png")


def plotKMeans(df, k):
    pca, embedded = decompose(df)

    kmeans = KMeans(k)
    kmeans.fit(df.values.T)
    clusters = kmeans.labels_

    centers = pca.transform(kmeans.cluster_centers_)

    plt.figure()
    cmap = plt.cm.get_cmap("Accent")

    plt.scatter(embedded[:, 0], embedded[:, 1], c=[cmap(t) for t in clusters])

    for i in range(len(embedded)):
        (x, y) = embedded[i]
        c = clusters[i]
        (x0, y0) = centers[c]
        plt.plot((x, x0), (y, y0), color=cmap(c))

    plt.scatter(centers[:, 0], centers[:, 1], c=[cmap(t) for t in range(k)], marker='D')
    types = [plt.scatter([], [], color=cmap(i)) for i in range(k)]
    names = [f"第{i + 1}类城市" for i in range(k)]
    plt.legend(types, names, loc='best')
    for i in range(0, len(clusters)):
        plt.text(embedded[i, 0], embedded[i, 1], df.columns[i])
    plt.savefig(f"{out_folder}/{figname_prefix}k-Means聚类点图.png")
    return clusters


def dendrogram(df, k, labels):
    from matplotlib import pyplot as plt
    from scipy.cluster import hierarchy
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.cluster import AgglomerativeClustering
    hierarchy.set_link_color_palette(['r', 'g', 'c', 'm', 'y'])

    def plot_dendrogram(model):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        threshold = linkage_matrix[len(linkage_matrix) - k + 1][2]
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, labels=labels,
                   color_threshold=threshold,
                   leaf_font_size=10, leaf_rotation=90, show_leaf_counts=False)

    X = df.values.T

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    plt.figure()
    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(f"{out_folder}/{figname_prefix}聚集过程图.png")


def barplotTypes(df, clusters, k):
    cities = [[] for _ in range(k)]
    for i in range(len(clusters)):
        cities[clusters[i]].append(df.values[:, i])
    avgs = [np.average(c, axis=0) for c in cities]
    fig, axes = plt.subplots(k, 1, sharex='col', figsize=(10, 10))
    times = [' '] * len(avgs[0])
    times[0] = '2018年5月'
    times[8] = '2019年1月'
    times[20] = '2020年1月'
    ymax = np.max(avgs)
    ymin = np.min(avgs)
    for i in range(k):
        data = avgs[i]
        axes[i].set_title(f'第{i + 1}类')
        axes[i].set_ylim(ymin, ymax)
        x = range(len(data))
        axes[i].bar(x, data, color=plt.cm.get_cmap("Set3").colors)
        plt.xticks(x, times)
    plt.savefig(f"{out_folder}/{figname_prefix}各类城市均值图.png")

    pass


def barplotSingleType(df, clusters, m):
    plt.figure()
    cities = []
    names = []
    for i in range(len(clusters)):
        if clusters[i] != m:
            continue
        cities.append(df.values[:, i])
        names.append(df.columns[i])

    times = [' '] * df.shape[0]
    times[0] = '2018年5月'
    times[8] = '2019年1月'
    times[20] = '2020年1月'

    ymax = np.max(cities)
    ymin = np.min(cities)
    fig, axes = plt.subplots(len(cities), 1, sharex='col', figsize=(10, 10), squeeze=False)
    for i in range(len(cities)):
        data = cities[i]
        axes[i, 0].set_title(names[i])
        axes[i, 0].set_ylim(ymin, ymax)
        x = range(len(data))
        axes[i, 0].bar(x, data, color=plt.cm.get_cmap("Set3").colors)
        plt.xticks(x, times)
    plt.savefig(f"{out_folder}/{figname_prefix}第{m + 1}类城市.png")
    pass


def analyze(df, k):
    """

    :param df:
    :param k: the hyper-parameter for clustering
    :return:
    """
    kmeansAndCost(df, k)
    clusters = plotKMeans(df, k)
    dendrogram(df, k, df.columns)
    barplotTypes(df, clusters, k)
    barplotSingleType(df, clusters, 1)


def main():
    global figname_prefix

    df = loadData()
    figname_prefix = "原始_"
    analyze(df, 4)

    df2 = (df - df.mean()) / df.std()
    figname_prefix = "归一化后_"
    analyze(df2, 6)


if __name__ == '__main__':
    main()
