# one hop helper function
def one_hop_majority_vote(G, gender_y_update,train_index, test_index):
    A = np.array(nx.adjacency_matrix(G).todense()) 
    D = np.diag(np.sum(A, axis=1))
    d = np.diag(D)
    theta = [None]*len(gender_y_update)
    accuracy_score_benchmark = np.sum(gender_y_update[train_index])/len(train_index)
    accuracy = [] 
    micro_auc = []
    wt_auc = []
    for j in train_index:
        theta[j] = gender_y_update[j]
    for j in test_index:
        #print(j)
        j_neighbor = list(G.neighbors(j))
        j_neighbor_label = list(set(j_neighbor).intersection(train_index))
        if len(j_neighbor_label) ==0:
            theta[j] = accuracy_score_benchmark
        else:
            theta[j] = np.sum(theta[ii] for ii in j_neighbor_label)/len(j_neighbor_label)
    preference_by_class_matrix =np.zeros((len(theta),2))
    preference_by_class_matrix[:,1] = theta
    preference_by_class_matrix[:,0] = np.subtract(1, preference_by_class_matrix[:,1]) 
        
        
    test_index = np.array(test_index)       
    micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='micro'))        
    wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='weighted'))
              

    ## f1-score version
    y_true = label_binarize(gender_y_update[test_index],np.unique(gender_y_update))
    y_pred = ((preference_by_class_matrix[test_index,:][:,1]) >accuracy_score_benchmark)+0
    accuracy.append(f1_score(y_true, y_pred, average='macro'))
    
    return(theta, micro_auc, wt_auc, accuracy)