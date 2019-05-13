import numpy as np
import pandas as pd
import classifiers as cl
from sklearn.model_selection import train_test_split


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
    # xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
    return x,y

def main():

    x, y = load()
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)
    concrete_strategy_a = cl.SvmClassifier()
    context = cl.Context(concrete_strategy_a)
    context.context_interface(xTrain, yTrain, xTest, yTest)

if __name__ == "__main__":
    main()
