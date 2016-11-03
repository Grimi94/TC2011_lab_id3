import numpy as np
import pandas as pd


class DecisionTree(object):

    def __init__(self):
        self.root = None
        self.feature_names = None
        self.categorical = None


    def _information_gain(self, y, labels):
        #TODO: sum labels after
        return self._entropy(y) - (np.abs(self._entropy(labels)) * float(len(labels))/len(labels))


    def _build_tree(self, X, y):
        node = TreeNode()


    def fit(self, X, y, feature_names):
        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        is_categorical = lambda x: isinstance(x, str) or \
                         isinstance(x, bool) or \
                         isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)


    def predict(self, X):
        pass


    def _entropy(self, y):
        pass

    def _choose_split_index(X, y):
        pass

    def _make_split(self, X, y):
        pass

    def _get_p(self, y):
        return np.unique(y, return_counts=True)[1]/float(len(y))

class TreeNode(object):
    def __init__(self):
        self.feature_name = None
        self.values = []
        self.nodes = []
        self.value = None
        self.column = None
        self.leaf = False
