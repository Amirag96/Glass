import abc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import confusion_matrix


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def context_interface(self, x, y, xt, yt):
        return self._strategy.algorithm_interface(x, y, xt, yt)

def Result(yTest,yPred):

    # Compute confusion matrix to evaluate the accuracy of a classification
    cm = confusion_matrix(yTest, yPred)
    # normailze = true  If False, return the number of correctly classified samples.
    # Otherwise, return the fraction of correctly classified samples.
    acc = accuracy_score(yTest, yPred, normalize=True, sample_weight=None)
    # Build a text report showing the main classification metrics
    # (Ground truth (correct) target values, Estimated targets as returned by a classifier)
    cr = classification_report(yTest, yPred)
    print("Confusion Matrix: ", cm)
    print("Accuracy: ", acc)
    print("Classification Report: ", cr)

class Strategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def algorithm_interface(self, x, y, xt, yt):
        pass

class SvmClassifier(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        #Kernels [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(xTrain, yTrain.values.ravel())
        yPred = svclassifier.predict(xTest)
        #joblib.dump(svclassifier, 'models/svmfs.joblib')

        Result(yTest,yPred)


class DecisionTree(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)

        Result(yTest,yPred)


class NaiveBayes(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        clf = GaussianNB()
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)

        Result(yTest,yPred)


class KnnClassifier(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        knnclassifier = KNeighborsClassifier(n_neighbors=1)
        knnclassifier.fit(xTrain, yTrain)
        yPred = knnclassifier.predict(xTest)

        Result(yTest, yPred)


class RandomForest(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):

        rfclassifier = RandomForestClassifier(n_estimators=100 ,random_state=0)
        #rfclassifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
        rfclassifier.fit(xTrain, yTrain)
        yPred = rfclassifier.predict(xTest)

        Result(yTest,yPred)

