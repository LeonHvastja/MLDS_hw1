import numpy as np


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c = ... # select random columns
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
        # should probably add some checks that the target vector entires are 1 and 0.
        
        # just for robustness and testing
        assert (len(X) == len(y)), "The input data and label vector are not of equal length" 
        
        if (len(y) < self.min_samples): # we are in a leaf node
            return TreeNode(None, None, None)
        if (len(np.unique(y)) == 1): # check if we have a pure node
            return TreeNode(None, None, None)
        
        right_i, left_i, decision_rule = find_decision_rule(X, y)
        
        left_subtree = Tree()
        right_subtree = Tree()
        
        
        return TreeNode(left_subtree.build(X[left_i], y[left_i]),right_subtree.build(X[right_i], y[right_i]), decision_rule) 


    def find_decision_rule(X, y):
        """
        Input: X - data, y - labels
        Output: A tuple (left, right, decision_rule), left indicies, right indicies and rule. Rule itself is a tuple
        of the index of the feature to split on and the value of where to split.)
        """

        decision_rule = None
        best_info_gain = 0 
        
        for i, feature in enumerate(X.T): # we iterate over rows of the transpose so it's actually over columns
            # TODO: better implementation, this just uses the mean to split
            split_value = np.mean(feature) # this is just a number of where to split
            left = np.where(feature < split_value)[0]
            right = np.where(feature >= split_value)[0]
            
            
            current_info_gain = information_gain(y, left, right)
            if (current_info_gain > best_info_gain):
                best_info_gain = current_info_gain
                decision_rule = (i, split_value)
        
        return (left, right, decision_rule) 
    
    def gini_impurity(y):
        """
        We can use this simplified version because we are solving a strictly binary classification problem, 
        assume y is a numpy array with values of 0 or 1.
        """
        
        label_one_probability = sum(y)/len(y)
        
        return 1 - ((label_one_probability)**2 + (1-label_one_probability)**2)
    
    def information_gain(y , left_partition_indicies, right_partition_indicies):
        n_left = len(left_partition_indicies)
        n_right = len(right_partition_indicies)
        n = n_left + n_right
        
        l_weight = n_left/n
        r_weight = n_right/n
        
        inf_gain = (gini_impurity(y[np.concatenate((left_partition_indicies,right_partition_indicies))]) 
                    - gini_impurity(y[left_partition_indicies])*l_weight 
                    - gini_impurity(y[right_partition_indicies])*r_weight)
        
        return inf_gain


class TreeNode:

    def __init__(self, decision_rule, ...):
        self.decision_rule = decision_rule
        # ...

    def predict(self, X):
        return np.ones(len(X))  # make this output a vector of predictions


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
