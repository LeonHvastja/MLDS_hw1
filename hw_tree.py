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