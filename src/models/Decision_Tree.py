import numpy as np


class Split_Option:
    def __init__(self, feature_index, threshold, info_gain):
        self.feature = feature_index
        self.threshold = threshold
        self.info_gain = info_gain

    def load_split(self, X, y):
        self.x_left = X[X[:, self.feature] <= self.threshold]
        self.x_right = X[X[:, self.feature] > self.threshold]

        feature_values = X[:, self.feature]
        self.y_left = y[feature_values <= self.threshold]
        self.y_right = y[feature_values > self.threshold]

class DT_Node:
    def __init__(self, feature: int, threshold: float, info_gain: float,
                 node_class: int, left, right):
        # if feature is -1 then it's leaf node
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.node_class = node_class
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.feature == -1

    def get_class(self):
        return self.node_class


class DecisionTree:
    def __init__(self, max_depth: int = 1000, min_samples_split: int = 3):
        # min_samples_split is the number of data points that need to be on a
        # node to warrent a split. If count of data points < min_samples_split
        # then the node is not split any further.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _build(self, X, y, depth=0):

        # If there are fewer samples than min_samples_split
        # or depth limit reached, stop splitting and return a leaf node.
        if X.shape[0] < self.min_samples_split or depth >= self.max_depth:
            return DT_Node(
                feature=-1,
                threshold=-1,
                info_gain=-1,
                # count occurrences of each unique value in y, then obtains the
                # index of the one that is most common as the predicted class
                node_class=np.argmax(np.bincount(y)),
                left=None,
                right=None
            )

        # Find the best split
        bs = self.pick_best_split(X, y)

        # If no information gain, return a leaf node
        if bs.info_gain == -1 or len(bs.y_left) == 0 or len(bs.y_right) == 0:
            return DT_Node(
                feature=-1,
                threshold=-1,
                info_gain=-1,
                node_class=np.argmax(np.bincount(y)),
                left=None,
                right=None
            )

        # split class
        # - feature
        # - threshold
        # - info_gain
        # - x_left
        # - x_right
        # - y_left
        # - y_right

        left = self._build(bs.x_left, bs.y_left, depth + 1)
        right = self._build(bs.x_right, bs.y_right, depth + 1)

        return DT_Node (
            feature=bs.feature,
            threshold=bs.threshold,
            info_gain=bs.info_gain,
            node_class=np.argmax(np.bincount(y)),
            left=left,
            right=right
        )

    def entropy(y):
        '''
        y: a numpy array of class values

        return:
            The entropy of the class values in y.
        '''

        # YOUR CODE HERE (~ 4-5 lines)
        unique_values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy_value = -np.sum(probabilities * np.log2(probabilities))

        return entropy_value

    def entropy_of_split(y, y_left, y_right):
        '''
        y: a numpy array of class values
        y_left: a numpy array of class values in the left half of the split
        y_right: a numpy array of class values in the right half of the split

        return:
            The entropy of the class values in y given the split.
        '''
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        # YOUR CODE HERE (~ 4-5 lines)
        entropy_value = (len(y_left) / len(y)) * y.entropy(y_left) + (len(y_right) / len(y)) * y.entropy(y_right)

        return entropy_value

    def information_gain(y, y_left, y_right):
        '''
        y: a numpy array of class values
        y_left: a numpy array of class values in the left half of the split
        y_right: a numpy array of class values in the right half of the split

        return:
            The information gain provided by the split.
        '''

        # YOUR CODE HERE (~ 1 line)
        info_gain = y.entropy(y) - y.entropy_of_split(y, y_left, y_right)

        return info_gain

    def pick_best_split(X, y):
        '''
        X: a numpy array (dataset) of shape (n, m)
        y: an array containing the class for each row of X, of shape (n, 1)

        return:
            A Best_Split() object that contains information about the best split
            given the data. This is a greedy selection based on information gain.
        '''

        best_split = Split_Option(feature_index=-1, threshold=-1, info_gain=-1)

        # YOUR CODE HERE (~ 10-15 lines)
        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                y_left = y[feature_values <= threshold]
                y_right = y[feature_values > threshold]
                info_gain = X.information_gain(y, y_left, y_right)

                # Skip invalid splits where one side is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                if info_gain > best_split.info_gain:
                    best_split = Split_Option(feature_index, threshold, info_gain)
        # END OF YOUR CODE

        best_split.load_split(X, y)

        return best_split

    def fit(self, X, y):
        self.root = self._build(X, y)

    # node class
    # node
    # - feature
    # - threshold
    # - info_gain
    # - node_class
    # functions
    # is_leaf() # returns boolean t/f if it is a leaf node
    # get_class() # returns majority value of classified points at a given leaf

    def _predict(self, x, node):

        if node.is_leaf():
            return node.get_class()

        # continues deeper into the tree
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])
