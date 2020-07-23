import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing

from project4 import Project4

df = pd.read_excel(Project4.DATA_PATH, header=1)

label = df.iloc[:, 0:3]
feature = df.iloc[:, 3:]

xdata = feature
ydata = label


def extractFeatureNames(model):
    from sklearn.feature_selection import SelectFromModel
    s = SelectFromModel(model, prefit=True, max_features=18)
    feature_choice = s.get_support()
    return xdata.columns[feature_choice]


def lassoModel():
    # 1. Lasso
    from sklearn.linear_model import Lasso
    lasso = Lasso()
    lasso.fit(xdata, ydata)
    return extractFeatureNames(lasso)


def randomForestModel():
    # 2. Random Forests
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=50)
    rf.fit(xdata, ydata)
    return extractFeatureNames(rf)


#
# # 3. PCA
#
def pcaModel():
    from sklearn.decomposition import PCA
    pca = PCA(15)
    pca.fit(xdata)
    return pca


#
# # 4. 基于Removing features with low variance  移除所有方差不满足阈值的特征
#
def varianceModel():
    from sklearn.feature_selection import VarianceThreshold
    v = VarianceThreshold(200)
    v.fit(xdata)
    return xdata.columns[v.get_support()]


def main():
    featureSelected = [
        lassoModel(),
        randomForestModel(),
        # pcaModel(),
        varianceModel()
    ]
    for fs in featureSelected:
        print([f for f in fs])
        print("size =", len(fs))


"""
Results:
['在炉时间', '抽出温度', '卷取温度平均值', '粗轧2温度', '轧辊1温度', '轧辊7温度', '铝', '钙', '碳', '氢', '锰', '钼', '铌', '化11', '化12', '化13', '化14', '化15']
size = 18
['在炉时间', '碳', '锰', '铌', '化12']
size = 5
['在炉时间', '卷取速度', '粗轧1温度', '粗轧2温度', '轧辊1温度', '冷却时间', '碳', '锰', '化12', '化15', '化16']
size = 11

"""

main()
