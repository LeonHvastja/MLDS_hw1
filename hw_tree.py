def gini_impurity(data: np.array, label_column: int):
    # we can use this simplified version because we are solving a strictly binary classification problem
    label_one_probability = sum(data[:,label_column])/len(data)
    return 1 - ((label_one_probability)**2 + (1-label_one_probability)**2)