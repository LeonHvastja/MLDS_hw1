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

    def build(self, X, y):
        return TreeNode(...) # make this output the root node


class TreeNode:

    def __init__(self, parent = None, decision_rule, ...):
        self.parent = parent
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