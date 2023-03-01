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
        return TreeNode(...) # dummy output


class TreeNode:

    def __init__(self, ...):
        # ...

    def predict(self, X):
        return np.ones(len(X))  # dummy output


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

def gini_impurity(data: np.array, label_column: int):
    # we can use this simplified version because we are solving a strictly binary 
    # classification problem
    label_one_probability = sum(data[:,label_column])/len(data)
    return 1 - ((label_one_probability)**2 + (1-label_one_probability)**2)


def information_gain(data, left_partition_indicies, right_partition_indicies, label_column):
    n_left = len(left_partition_indicies)
    n_right = len(right_partition_indicies)
    n = n_left + n_right
    
    l_weight = n_left/n
    r_weight = n_right/n
    
    inf_gain = (gini_impurity(data[np.concatenate((left_partition_indicies,right_partition_indicies))], label_column) 
                - gini_impurity(data[left_partition_indicies], label_column)*l_weight 
                - gini_impurity(data[right_partition_indicies], label_column)*r_weight)
    
    return inf_gain