import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("../data/movies.csv", encoding='GBK')
print("Dataset")
print(df)
df_train = df.drop(axis=1, labels=['showdate', 'movieid', 'name']).dropna()
X = df_train.drop(axis=1, labels=['daily_boxoffice'])
print(X)
Y = df_train['daily_boxoffice']
print(Y)

X_train, X_valid, y_train, y_valid \
    = train_test_split(X.values, Y.ravel(), test_size=0.2)
df_predict = df[-18:]
X_predict = df_predict.drop(axis=1, labels=['showdate', 'movieid', 'name','daily_boxoffice']).values
print(X)


def saveResult(path,prediction):
    result = df_predict.copy()
    result['daily_boxoffice'] = prediction
    result.to_csv(path,encoding='UTF-8',float_format= '%.2f')


def randomForest():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    print()
    print("Random forest:")
    print(model.score(X_valid, y_valid))
    model = RandomForestRegressor()
    model.fit(X,Y)
    prediction =model.predict(X_predict)
    saveResult('../out/movie_rf.csv',prediction)
    print()

def decisionTree():
    model = DecisionTreeRegressor()
    model.fit(X_train,y_train)
    print()
    print("Decision Tree:")
    print(model.score(X_valid,y_valid))
    model = DecisionTreeRegressor()
    model.fit(X, Y)
    prediction =model.predict(X_predict)
    saveResult('../out/movie_tree.csv', prediction)

randomForest()
decisionTree()
