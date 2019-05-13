import abc
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from tensorflow import confusion_matrix


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def context_interface(self, x, y, xt, yt):
        return self._strategy.algorithm_interface(x, y, xt, yt)


class Strategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def algorithm_interface(self, x, y, xt, yt):
        pass

class SvmClassifier():
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        #Kernels [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(xTrain, yTrain.values.ravel())
        yPred = svclassifier.predict(xTest)

        # Compute confusion matrix to evaluate the accuracy of a classification
        cm = confusion_matrix(yTest, yPred)
        # normailze = true  If False, return the number of correctly classified samples.
        # Otherwise, return the fraction of correctly classified samples.
        acc = accuracy_score(yTest, yPred, normalize=True, sample_weight=None)
        # Build a text report showing the main classification metrics
        # (Ground truth (correct) target values, Estimated targets as returned by a classifier)
        cr = classification_report(yTest, yPred)

        #joblib.dump(svclassifier, 'models/svmfs.joblib')

        print("Confusion Matrix: " , cm)
        print("Accuracy: " , acc)
        print("Classification Report: " , cr)
        #return cm, acc, cr
        pass