import numpy as np
import pandas as pd
from collections import Counter

class DecisionTree(object):

    def __init__(self):
        self.root           = None
        self.feature_names  = None
        self.categorical    = None
        self.feature_values = None


    def _information_gain(self, y, subsets):
        """
        Calculate information of subsets
        """
        n             = y.shape[0]
        child_entropy = 0

        for y_i in subsets:
            child_entropy += self._entropy(y_i) * y_i.shape[0] / float(n)

        return self._entropy(y) - child_entropy


    def _build_tree(self, X, y, feature_names):
        node = TreeNode()
        split_column, split_values, splits = self._choose_split_column(X, y)

        # Check if we finished one node
        if split_column == None or len(np.unique(y)) == 1:
            node.leaf    = True
            node.classes = Counter(y)
            node.name    = node.classes.most_common(1)[0][0]
        else:
            node.name   = feature_names[split_column]
            node.column = split_column
            node.values = self.feature_values[node.name]

            # For every split value we subset X and y and recurse down
            for idx, split_value in enumerate(split_values):

                new_X = np.delete(X, split_column, axis=1)
                new_feature_names = np.delete(feature_names, split_column)
                new_X = new_X[splits[idx]]
                new_y = y[splits[idx]]
                node.children[split_value] = self._build_tree(new_X, new_y, new_feature_names)

        return node


    def fit(self, X, y, feature_names=None, feature_values=None):
        self.feature_values = feature_values

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        is_categorical   = lambda x: isinstance(x, str) or \
                           isinstance(x, bool) or \
                           isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y, self.feature_names)


    def predict(self, X):
        """
        Currently not implemented
        """
        pass


    def _entropy(self, y):
        """
        Calculates the entropy of a given array
        """
        # Get size
        n         = y.shape[0]
        summation = 0

        # Summatory
        for c_i in np.unique(y):
            prob = sum(y == c_i) / float(n)
            summation += prob * np.log2(prob)

        return -summation


    def _choose_split_column(self, X, y):
        split_column, split_values, splits = None, None, None
        gain = 0

        # Itearte over columns
        for i in xrange(X.shape[1]):
            values  = np.unique(X[:, i])
            subsets = []
            conditions = []

            if len(values) < 1:
                continue
            # Create y subset for each of the column values
            for value in values:
                conditions.append(X[:,i] == value)
                subsets.append(y[X[:,i] == value])

            # Calculate information gain of the column
            new_gain = self._information_gain(y, subsets)

            if new_gain > gain:
                split_column = i
                split_values = values
                gain = new_gain
                splits = conditions

        return split_column, split_values, splits

    def __str__(self):
        return str(self.root)


class TreeNode(object):
    def __init__(self):
        self.name     = None
        self.children = {}
        self.values   = None
        self.column   = None
        self.leaf     = False
        self.classes  = None

    def describe(self, level=0, suffix=""):
        result = suffix

        if not self.leaf:
            #TODO: finish loop
            for node_name in self.values:
                child = self.children[node_name]
                result += "  " * level
                result += self.name + ": " + node_name
                result += child.describe(level + 1, "\n")

            # for key, val in self.children.iteritems():
            #     result += "  " * level
            #     result += self.name + ": " + key
            #     result += val.describe(level + 1, "\n")
        else:
            result += "  " * level
            result += "ANSWER: " + self.classes.keys()[0] + "\n"

        return result


    def __repr__(self):
        return self.describe()
