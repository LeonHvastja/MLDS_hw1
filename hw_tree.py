import numpy as np


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c = rand.sample(range(0, X.shape[1]), round(X.shape[1]**0.5))
    return c


class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples 
    
    def build(self, X: np.array, y: np.array):
        """
        Recusrively build a tree, stop recursion when a split has a child node with gini impurity 0 or
        when we have less than min_samples samples.
        """
        if (len(y) < self.min_samples): # we are in a leaf node
            return TreeNode(None, None, round(np.mean(y))) # make the majority class the prediction for this node
        if (np.all(y == 1)): # check if we have a node with all ones
            return TreeNode(None, None, 1)
        if (np.all(y == 0)):
            return TreeNode(None, None, 0)
        
        decision_rule = self.find_decision_rule(X, y)
        feature, split_value = decision_rule
        
        left_i = np.where(X[:,feature] < split_value)
        right_i = np.where(X[:, feature] >= split_value)
        
        left_subtree = Tree()
        right_subtree = Tree()
        
        return TreeNode(left_subtree.build(X[left_i], y[left_i]),
                        right_subtree.build(X[right_i], y[right_i]), 
                        decision_rule) 
    
    
    def find_decision_rule(self, X, y):
        """
        Input: X - data, y - labels
        Output: A tuple (left, right, decision_rule), left indicies, right indicies and rule. Rule itself is a tuple
        of the index of the feature to split on and the value of where to split.)
        """
        decision_rule = None
        best_info_gain = 0

        for feature in self.get_candidate_columns(X, self.rand):
            values = X[:, feature]
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            for i in range(len(sorted_values) - 1):
                current_info_gain = self.information_gain(y[sorted_indices], np.arange(0,i+1), np.arange(i+1, len(values)))
                
                if(current_info_gain > best_info_gain):
                    split_value = self.midpoint(i, sorted_values)
                    best_info_gain = current_info_gain
                    decision_rule = (feature, split_value)

        return decision_rule
                
    def midpoint(self, index, y):
        """Finds the average value of entires at index i and i+1 in a presumably sorted array."""
        return (y[index] + y[index + 1])/2
    
    def information_gain(self, y , left_partition_indicies, right_partition_indicies):
        """
        Input: Takes an array of labels and the indicies of which belong to the lefr and right partition.
        Output: Returns information gain for this particular split.
        """
        n_left = len(left_partition_indicies)
        n_right = len(right_partition_indicies)
        n = n_left + n_right

        l_weight = n_left/n
        r_weight = n_right/n

        inf_gain = (self.gini_impurity(y) 
                    - self.gini_impurity(y[left_partition_indicies])*l_weight 
                    - self.gini_impurity(y[right_partition_indicies])*r_weight)

        return inf_gain

    def gini_impurity(self, y):
        """
        We can use this simplified version because we are solving a strictly binary classification problem, 
        assume y is a numpy array with values of 0 or 1.
        """

        label_one_probability = sum(y)/len(y)

        return 1 - ((label_one_probability)**2 + (1-label_one_probability)**2)


class TreeNode:
    
    def __init__(self, left, right, decision_rule):
        """Left and right are TreeNode objects. Decision rule is either a tuple with a feature and value 
        to split on or a single value which determines the leaf's predicted label.
        """
        self.left = left
        self.right = right
        self.decision_rule = decision_rule

    def predict(self, X):
        prediction = np.empty(len(X))
        
        if ((self.left is None) and (self.right is None)): # we are in a leaf node
            return self.decision_rule
        
        # get left and right indices
        left_i = np.where(X.T[self.decision_rule[0]] < self.decision_rule[1])
        right_i = np.where(X.T[self.decision_rule[0]] >= self.decision_rule[1])
        
        left_prediction = self.left.predict(X[left_i])
        right_prediction = self.right.predict(X[right_i])
        
        prediction[left_i] = left_prediction
        prediction[right_i] = right_prediction
               
        return prediction


class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(...)  # initialize the tree properly

    def build(self, X, y):
        # ...
        return RFModel(...)


class RFModel:

    def __init__(self, ...):
        # ...

    def predict(self, X):
        # ...
        return predictions

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
