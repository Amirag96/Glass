import numpy as np
import pandas as pd
import sns as sns
from scipy.stats import skew, stats
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import classifiers as cl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer
from sklearn.preprocessing import Normalizer
import seaborn as sns

def load():

    #Reading dataset files and saving them as CSV
    # data = pd.read_csv('dataset/glass.data',delimiter=',', names = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type'])
    # data.to_csv('dataset/glass.csv',mode='a',index=False)
    data = pd.read_csv('dataset/glass.csv',delimiter=',')

    #6 classes:
    labels =['building_windows_float_processed',
             'building_windows_non_float_processed',
             'vehicle_windows_float_processed','',
             'containers',
             'tableware',
             'headlamps']

    #print(data.dtypes)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', 1000)
    #print(data.describe())

    #print(data['Type'].value_counts())
    #sns.countplot(data['Type'])
    #plt.show()

    ##  Data shape before removing outliners    ##
    #print(data.shape)
    #####   Removing outliners  ######
    data = zScore(data)
    #Compare(data)
    #data = IQR(data)
    ##  Data shape After removing outliners     ##
    #print(data.shape)

    #print(data['Type'].value_counts())
    #sns.countplot(data['Type'])
    #plt.show()

    ##  outer bracket (select column) , inner bracket (list)    ##
    y = data[['Type']]
    ##  axis = 1 (drop column)  ##
    x = data.drop(['Type'],axis=1)
    #Graph(data, x)
    #print(x.shape)

    #####   Feature Distribution    #####
    #Graph(data,x)

    #####   Feature Selection   #####
    #x = FeatureSelect(x,y)
    #print(x.shape)

    return x,y

# Z-score is to describe any data point by finding their relationship with the Standard Deviation
# and Mean of the group of data points.
# Z-score is finding the distribution of data where mean is 0 and standard deviation is 1
# i.e. normal distribution

def zScore(data):
    z = np.abs(stats.zscore(data))
    #   we used threshold = 3 ( In most of the cases a threshold of 3 or -3 )
    #print(np.where(z > 3))
    #  1st array is row numbers and 2nd array is column numbers
    #print(z[105][6])
    data = data[(z < 3).all(axis=1)]
    return data

def IQR(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    #print(IQR)
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

def FeatureSelect(x,y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(x)
    return X_new

def data_Normalizer(xTrain,xTest):
    norm = Normalizer()
    xTrain = norm.fit_transform(xTrain)
    xTest = norm.transform(xTest)
    return xTrain,xTest

def data_MinMax(xTrain,xTest):
    min_max = MinMaxScaler()
    xTrain = min_max.fit_transform(xTrain)
    xTest = min_max.fit_transform(xTest)
    return xTrain, xTest

def data_Standard(xTrain, xTest):
    sc_X = StandardScaler()
    xTrain = sc_X.fit_transform(xTrain)
    xTest = sc_X.transform(xTest)
    return xTrain, xTest

def data_Binarizer(xTrain,xTest):
    Bin = Binarizer(threshold=0.0)
    xTrain = Bin.fit_transform(xTrain)
    xTest = Bin.transform(xTest)
    return xTrain, xTest

def Graph(df,x):
    for feat in x:
        skew = df[feat].skew()
        sns.distplot(df[feat], kde=False, label='Skew = %.3f' % (skew), bins=30)
        plt.legend(loc='best')
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

def Compare(data):
    # Average Bar Plot for each variables
    types = np.unique(data['Type'])

    for i in range(len(types)):
        fig = plt.figure()
        average = data[[data.columns[i], "Type"]].groupby(['Type'], as_index=False).mean()
        sns.barplot(x='Type', y=data.columns[i], data=average)
        plt.show()

def main():

    x, y = load()

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

    #####   Normalizing Data    #####
    xTrain,xTest = data_Standard(xTrain, xTest)
    #xTrain, xTest = data_Normalizer(xTrain, xTest)
    # xTrain, xTest = data_MinMax(xTrain, xTest)
    # xTrain,xTest = data_Binarizer(xTrain, xTest)


    #####   Classifiers    #####
    #concrete_strategy = cl.NaiveBayes()
    #concrete_strategy = cl.SvmClassifier()
    #concrete_strategy = cl.KnnClassifier()
    #concrete_strategy=cl.LogesticRegression()
    concrete_strategy = cl.RandomForest()
    #concrete_strategy = cl.DecisionTree()

    context = cl.Context(concrete_strategy)
    context.context_interface(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
