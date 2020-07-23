import itertools

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
import prettytable as pt
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = '../data/A2006Data.xlsx'

LABEL_COLUMNS = [u"断裂延伸率", u"屈服强度", u"抗拉强度"]


def splitFeatureLabel(df, selectedFeatures):
    feature = df.drop(LABEL_COLUMNS, axis=1)
    if selectedFeatures is not None:
        feature = df[selectedFeatures]
    label = df[LABEL_COLUMNS]
    return feature, label


def loadData():
    df = pd.read_excel(DATA_PATH, header=1)
    df: pd.DataFrame = (df - df.mean()) / df.std()
    return df


def evaluate(modelAndNames, df, selectedFeatures=None):
    feature, label = splitFeatureLabel(df, selectedFeatures)
    X_train, X_test, ys_train, ys_test = train_test_split(feature, label, test_size=0.3)

    table = pt.PrettyTable()
    table.field_names = ['Model', 'Prediction', 'Score']
    table.float_format = ".2"
    for (model, name) in modelAndNames:
        for col_name in ys_train.columns:
            y_train = ys_train[col_name]
            y_test = ys_test[col_name]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            table.add_row((name, col_name, score))
    print(table)


def buildModels():
    mlp = MLPRegressor((64, 32), max_iter=500)
    decision_tree = DecisionTreeRegressor()
    random_forest = RandomForestRegressor()
    ada_boost = AdaBoostRegressor()
    return [
        (mlp, "MLP"),
        (decision_tree, "Decision tree"),
        (random_forest, "Random forest"),
        (ada_boost, "AdaBoost"),
    ]


def trainModels(models, selectedFeatures=None):
    df = loadData()
    X, labels = splitFeatureLabel(df, selectedFeatures)
    for (model, col_name) in zip(models, labels.columns):
        y = labels[col_name]
        model.fit(X, y)


def score(ys):
    s = 0
    for y in ys:
        if y > 0:  # normalized
            s += 1
    return s


def doEvaluation():
    df = loadData()
    modelAndNames = buildModels()
    for fs in FEATURES_SELECTED_GROUPS:
        print("Selected features: ", fs)
        if fs is not None:
            print("Total: ", len(fs))
        evaluate(modelAndNames, df, fs)


def optimize(models, selected_features, features_to_optimize, feature_delta=0.1, levels=1):
    df = loadData()
    print("Optimizing ", features_to_optimize, " with level = ", levels)
    features, labels = splitFeatureLabel(df, selected_features)
    adjustments = [i * feature_delta for i in range(-levels, levels + 1)]
    selected_feature_count = len(features_to_optimize)
    max_score = labels.shape[1]

    original_score_counting = [0] * (max_score + 1)
    adjusted_score_counting = [0] * (max_score + 1)

    count = 0
    print('-' * (features.shape[0] // 50))
    for ((_, x), y) in zip(features.iterrows(), labels.values):
        original_score = score(y)
        best_score = original_score
        xs = []
        for adjustment in itertools.product(adjustments, repeat=selected_feature_count):
            x = x.copy()
            x[features_to_optimize] = x[features_to_optimize] + adjustment
            xs.append(x)
        nys = [model.predict(xs) for model in models]
        for i in range(len(xs)):
            ny = [nys[j][i] for j in range(len(y))]
            new_score = score(ny)
            if new_score > best_score:
                best_score = new_score
            if new_score >= best_score:
                break
        original_score_counting[original_score] += 1
        adjusted_score_counting[best_score] += 1
        count += 1
        if count % 50 == 0:
            print(".", end='')
    print()
    print("Original: ", original_score_counting)
    print("Adjusted: ", adjusted_score_counting)


FEATURES_SELECTED_GROUPS = [
    None,
    ['在炉时间', '抽出温度', '卷取温度平均值', '粗轧2温度', '轧辊1温度', '轧辊7温度', '铝', '钙', '碳', '氢', '锰', '钼', '铌', '化11', '化12', '化13',
     '化14', '化15'],
    ['在炉时间', '碳', '锰', '铌', '化12'],
    ['在炉时间', '卷取速度', '粗轧1温度', '粗轧2温度', '轧辊1温度', '冷却时间', '碳', '锰', '化12', '化15', '化16']
]


def doOptimization():
    # models and features are selected according to the results of evaluation
    models = [
        MLPRegressor((64, 32), max_iter=500),  # 0.80
        RandomForestRegressor(),  # 0.85
        RandomForestRegressor()  # 0.94
    ]
    selected_features = ['在炉时间', '抽出温度', '卷取温度平均值', '粗轧2温度', '轧辊1温度', '轧辊7温度', '铝', '钙', '碳', '氢', '锰', '钼', '铌',
                         '化11', '化12', '化13', '化14', '化15']

    trainModels(models, selected_features)
    print("Training completed.")
    features_to_optimize = ['在炉时间', '抽出温度', "卷取温度平均值"]
    optimize(models, selected_features, features_to_optimize)

"""
Results:
Selected features:  None
+---------------+------------+-------+
|     Model     | Prediction | Score |
+---------------+------------+-------+
|      MLP      | 断裂延伸率 |  0.75 |
|      MLP      |  屈服强度  |  0.81 |
|      MLP      |  抗拉强度  |  0.92 |
| Decision tree | 断裂延伸率 |  0.62 |
| Decision tree |  屈服强度  |  0.79 |
| Decision tree |  抗拉强度  |  0.90 |
| Random forest | 断裂延伸率 |  0.80 |
| Random forest |  屈服强度  |  0.88 |
| Random forest |  抗拉强度  |  0.94 |
|    AdaBoost   | 断裂延伸率 |  0.76 |
|    AdaBoost   |  屈服强度  |  0.75 |
|    AdaBoost   |  抗拉强度  |  0.86 |
+---------------+------------+-------+

Selected features:  ['在炉时间', '抽出温度', '卷取温度平均值', '粗轧2温度', '轧辊1温度', '轧辊7温度', '铝', '钙', '碳', '氢', '锰', '钼', '铌', '化11', '化12', '化13', '化14', '化15']
Total:  18
+---------------+------------+-------+
|     Model     | Prediction | Score |
+---------------+------------+-------+
|      MLP      | 断裂延伸率 |  0.80 |
|      MLP      |  屈服强度  |  0.83 |
|      MLP      |  抗拉强度  |  0.91 |
| Decision tree | 断裂延伸率 |  0.70 |
| Decision tree |  屈服强度  |  0.77 |
| Decision tree |  抗拉强度  |  0.90 |
| Random forest | 断裂延伸率 |  0.82 |
| Random forest |  屈服强度  |  0.86 |
| Random forest |  抗拉强度  |  0.94 |
|    AdaBoost   | 断裂延伸率 |  0.77 |
|    AdaBoost   |  屈服强度  |  0.74 |
|    AdaBoost   |  抗拉强度  |  0.87 |
+---------------+------------+-------+

Selected features:  ['在炉时间', '碳', '锰', '铌', '化12']
Total:  5
+---------------+------------+-------+
|     Model     | Prediction | Score |
+---------------+------------+-------+
|      MLP      | 断裂延伸率 |  0.81 |
|      MLP      |  屈服强度  |  0.79 |
|      MLP      |  抗拉强度  |  0.90 |
| Decision tree | 断裂延伸率 |  0.71 |
| Decision tree |  屈服强度  |  0.81 |
| Decision tree |  抗拉强度  |  0.92 |
| Random forest | 断裂延伸率 |  0.80 |
| Random forest |  屈服强度  |  0.87 |
| Random forest |  抗拉强度  |  0.94 |
|    AdaBoost   | 断裂延伸率 |  0.78 |
|    AdaBoost   |  屈服强度  |  0.72 |
|    AdaBoost   |  抗拉强度  |  0.86 |
+---------------+------------+-------+

Selected features:  ['在炉时间', '卷取速度', '粗轧1温度', '粗轧2温度', '轧辊1温度', '冷却时间', '碳', '锰', '化12', '化15', '化16']
Total:  11
+---------------+------------+-------+
|     Model     | Prediction | Score |
+---------------+------------+-------+
|      MLP      | 断裂延伸率 |  0.72 |
|      MLP      |  屈服强度  |  0.82 |
|      MLP      |  抗拉强度  |  0.90 |
| Decision tree | 断裂延伸率 |  0.58 |
| Decision tree |  屈服强度  |  0.73 |
| Decision tree |  抗拉强度  |  0.86 |
| Random forest | 断裂延伸率 |  0.77 |
| Random forest |  屈服强度  |  0.83 |
| Random forest |  抗拉强度  |  0.93 |
|    AdaBoost   | 断裂延伸率 |  0.73 |
|    AdaBoost   |  屈服强度  |  0.69 |
|    AdaBoost   |  抗拉强度  |  0.84 |
+---------------+------------+-------+


"""




"""
Best models and features for the three index:

best_models = [
    MLPRegressor((64, 32), max_iter=500),  # 0.80
    RandomForestRegressor(),  # 0.85
    RandomForestRegressor()  # 0.94
]
best_features = [['在炉时间', '碳', '锰', '铌', '化12'],
                 ['在炉时间', '抽出温度', '卷取温度平均值', '粗轧2温度', '轧辊1温度', '轧辊7温度', '铝', '钙', '碳', '氢', '锰', '钼', '铌',
                  '化11', '化12', '化13', '化14', '化15'],
                 ['在炉时间', '碳', '锰', '铌', '化12']
                 ]
"""

"""
Optimizing  ['在炉时间', '抽出温度', '卷取温度平均值']  with level =  1
----------------------------------------------
..............................................
Original:  [130, 841, 1156, 186]
Adjusted:  [84, 812, 1216, 201]
"""


if __name__ == '__main__':
    doEvaluation()
    doOptimization()



