# the latest version for the iterative method
# first when set t = 1, reduced to 2-hop MV
def iterative_method_test(t, G, gender_y_update, theta0, train_index, test_index, bench_mark):
    A = np.array(nx.adjacency_matrix(G).todense()) 
    D = np.diag(np.sum(A, axis=1))
    d = np.diag(D)
    theta = [None]*len(theta0)
    theta0 = np.array(theta0)
    
    accuracy = [] 
    micro_auc = []
    wt_auc = []
    
    for j in train_index:
        theta[j] = theta0[j]
    if t ==1:
        accuracy_score_benchmark = np.sum(theta0[train_index])/len(train_index)
        #print(accuracy_score_benchmark)
        for j in test_index:
        # if it is the first iteration, which is reduced to the 2-hop method
            # update theta[j] according to the 2-hop update rule
            # get j's neighbor
            denom = 0
            nom = 0
            j_neighbor = list(G.neighbors(j))
            l_two_hop_neighbor = []
            #print(j_neighbor)
            for l in j_neighbor:
                # get l's neighbor
                l_neighbor = list(G.neighbors(l)) 
                # if l's neighbor is labeled then add it to the two hop neighbor set
                l_two_hop_neighbor = l_two_hop_neighbor+[l_neighbor]
        
            l_two_hop_neighbor = list(itertools.chain.from_iterable(l_two_hop_neighbor)) 
            train_set = np.array(train_index)
             
            two_hop_neighbor = []
            two_hop_neighbor = [i for i in l_two_hop_neighbor if i in train_set]
            #print('l_two_hop_neighbor new',two_hop_neighbor)
            # update theta[j]
            if len(two_hop_neighbor) == 0:
                theta[j] = accuracy_score_benchmark
            else:
                denom  = len(two_hop_neighbor) 
                theta[j] = np.sum(theta[ii] for ii in two_hop_neighbor)/denom 
           
            
            
        preference_by_class_matrix =np.zeros((len(theta),2))
        preference_by_class_matrix[:,1] = theta
        preference_by_class_matrix[:,0] = np.subtract(1, preference_by_class_matrix[:,1]) 
        
        
        test_index = np.array(test_index)
        micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='micro'))
        wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),
                                                       preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='weighted'))
              

        ## f1-score version
        y_true = label_binarize(gender_y_update[test_index],np.unique(gender_y_update))
        y_pred = ((preference_by_class_matrix[test_index,:][:,1]) >accuracy_score_benchmark)+0
        accuracy.append(f1_score(y_true, y_pred, average='macro'))#, pos_label=1) )
            
            
        
    else:# when it's not the first iteration: t>=2
            # get the two hop neighbors(no matter labeled or not)
               # update theta[j] according to the 2-hop update rule
        for j in test_index:         # get j's neighbor 
            denom = 0
            nom = 0
            j_neighbor = list(G.neighbors(j))
   
            l_two_hop_neighbor = []
            two_hop_neighbor = []
            for l in j_neighbor:
                # get l's neighbor
                l_neighbor = list(G.neighbors(l)) 
                # if l's neighbor is labeled then add it to the two hop neighbor set
                l_two_hop_neighbor = l_two_hop_neighbor + [l_neighbor]
            #two_hop_neighbor = list(set().union(*l_two_hop_neighbor))
            two_hop_neighbor = list(itertools.chain.from_iterable(l_two_hop_neighbor)) 
            
        
            # average the results for the 2-hop nodes
    
            denom  = len(two_hop_neighbor) 
            theta[j] = np.sum(theta0[ii] for ii in two_hop_neighbor)/denom 
    
      
        # perform accuracy analysis
        preference_by_class_matrix =np.zeros((len(theta),2))
        preference_by_class_matrix[:,1] = theta
        preference_by_class_matrix[:,0] = np.subtract(1, preference_by_class_matrix[:,1]) 
        
        
        test_index = np.array(test_index)
        micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='micro'))
        wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index],np.unique(gender_y_update)),
                                                       preference_by_class_matrix[test_index,:][:,1]-preference_by_class_matrix[test_index,:][:,0],average='weighted'))
              

        ## f1-score version
        y_true = label_binarize(gender_y_update[test_index],np.unique(gender_y_update))
        #y_pred = ((preference_by_class_matrix[test_index,:][:,1]) > bench_mark)+0
        y_pred = ((preference_by_class_matrix[test_index,:][:,1]) > accuracy_score_benchmark)+0
        accuracy.append(f1_score(y_true, y_pred, average='macro'))#, pos_label=1) )
    
    return (theta, micro_auc, wt_auc, accuracy) 
