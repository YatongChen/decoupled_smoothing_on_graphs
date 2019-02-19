# modified from Kristen's code
def create_proportion_class_k_friends(adj_matrix, node_id,
                                     y_labels, k_class):
    prop_class_k_friends = []
    total_neighbors = []
    total = np.sum(adj_matrix,1)
    class_k_num = adj_matrix*np.matrix((y_labels==k_class)+0).T
    prop_class_k =class_k_num/total
    return(np.array(prop_class_k).T[0])