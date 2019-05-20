import numpy as np
import pandas as pd
import sns as sns
from scipy.stats import skew
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats import boxcox
import classifiers as cl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load():
    #Reading dataset files and saving them as CSV
    # data = pd.read_csv('dataset/glass.data',delimiter=',', names = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type'])
    # data.round({"RI": 5})
    # print(data['RI'])
    # data.to_csv('dataset/glass1.csv',mode='a',index=False)

    data = pd.read_csv('dataset/glass.csv',delimiter=',')
    #6 classes:
    labels =['building_windows_float_processed',
             'building_windows_non_float_processed',
             'vehicle_windows_float_processed',''
             'containers',
             'tableware',
             'headlamps']

    #outer bracket (select column) , inner bracket (list)
    y = data[['Type']]
    #axis = 1 (drop column)
    x = data.drop(['Type'],axis=1)
    x = boxcox(x)

    # xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
    return x,y

def histogram(x):
    x[x.dtypes[(x.dtypes == "float64") | (x.dtypes == "int64")].index.values].hist(figsize=[11, 11])
    plt.show()

def countType(y):
    print(y['Type'].value_counts())
    plt.hist()
    plt.show()
    types = np.unique(y['Type'])
    for i in range(len(types)):
        fig = plt.figure()
        average = y[[y.columns[i], "Type"]].groupby(['Type'], as_index=False).mean()
        sns.barplot(x='Type', y=y.columns[i], data=average)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def boxcox(X):
    classes = X.columns.values
    # print(classes)
    # print(X[classes[0]])
    X_u = pd.DataFrame()
    for c in classes:
        scaled = preprocessing.scale(X[c])
        boxcox_scaled = preprocessing.scale(boxcox(X[c] + np.max(np.abs(X[c]) + 1))[0])
        X_u[c] = boxcox_scaled
        skness = skew(scaled)
        boxcox_skness = skew(boxcox_scaled)
        figure = plt.figure()
        figure.add_subplot(121)
        plt.hist(scaled, facecolor='blue', alpha=0.5)
        plt.xlabel(c + " - Transformed")
        plt.title("Skewness: {0:.2f}".format(skness))
        figure.add_subplot(122)
        plt.hist(boxcox_scaled, facecolor='red', alpha=0.5)
        plt.title("Skewness: {0:.2f}".format(boxcox_skness))
        plt.show()

def main():

    x, y = load()
    c = x.columns.values
    x = boxcox(x)

    # histogram(x)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

    # PreProcessing
    min_max = MinMaxScaler()
    X_train_minmax = min_max.fit_transform(xTrain)
    X_test_minmax = min_max.fit_transform(xTest)

    # concrete_strategy = cl.SvmClassifier()
    # concrete_strategy = cl.DecisionTree()
    # concrete_strategy = cl.NaiveBayes()
    # concrete_strategy = cl.KnnClassifier()

    # concrete_strategy = cl.RandomForest()
    # context = cl.Context(concrete_strategy)
    # context.context_interface(xTrain, yTrain, xTest, yTest)

    # context.context_interface(X_train_minmax, yTrain, X_test_minmax, yTest)


if __name__ == "__main__":
    main()
