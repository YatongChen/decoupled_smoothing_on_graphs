def iterative_decoupled_method(t, G, gender_y_update, theta0, train_index, test_index):
    A = np.array(nx.adjacency_matrix(G).todense()) 
    D = np.diag(np.sum(A, axis=1))
    d = np.diag(D)
    theta = [None]*len(theta0)
    theta0 = np.array(theta0)
    
    accuracy = [] 
    micro_auc = []
    wt_auc = []
    ratio  = len(train_index)/len(gender_y_update)
    accuracy_score_benchmark = np.sum(theta0[train_index])/len(train_index)
    
    for j in train_index:
        theta[j] = theta0[j]
    if t ==1:
        # test_index_non_na is used to hold test nodes whose two-hops labeled neighbors are not empty
        #test_index_non_na = []
        for j in test_index:
        # if it is the first iteration, which is reduced to the 2-hop method
            # update theta[j] according to the 2-hop update rule
            # get j's neighbor
            denom = 0
            nom = 0
            j_neighbor = list(G.neighbors(j))
            l_two_hop_neighbor = []
            for l in j_neighbor:
                # get l's neighbor
                l_neighbor = list(G.neighbors(l)) 
                # if l's neighbor is labeled then add it to the two hop neighbor set
                l_two_hop_neighbor = l_two_hop_neighbor + [l_neighbor]
        
            l_two_hop_neighbor = list(itertools.chain.from_iterable(l_two_hop_neighbor)) 
            train_set = np.array(train_index)
             
            two_hop_neighbor = []
            two_hop_neighbor = [i for i in l_two_hop_neighbor if i in train_set]
            # update theta[j]: if a node doesn't have any two hop friends, then just leave it to be None
            # when compute accuracy, leave it
            if len(two_hop_neighbor) != 0:
                #test_index_non_na.append(j)
                denom  = len(two_hop_neighbor) 
                theta[j] = np.sum(theta[ii] for ii in two_hop_neighbor)/denom  
            else:
                theta[j] = accuracy_score_benchmark
                     
        preference_by_class_matrix =np.zeros((len(theta),2))
        preference_by_class_matrix[:,1] = theta
        preference_by_class_matrix[:,0] = np.subtract(1, preference_by_class_matrix[:,1]) 
        
        
        # test_index_non_na = np.array(test_index_non_na)
        # micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update)),preference_by_class_matrix[test_index_non_na,:][:,1]-preference_by_class_matrix[test_index_non_na,:][:,0],average='micro'))
        # wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update)),
        #                                                preference_by_class_matrix[test_index_non_na,:][:,1]-preference_by_class_matrix[test_index_non_na,:][:,0],average='weighted'))
              

        # ## f1-score version
        # y_true = label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update))
        # y_pred = ((preference_by_class_matrix[test_index_non_na,:][:,1]) > accuracy_score_benchmark)+0
        # accuracy.append(f1_score(y_true, y_pred, average='macro'))
        
            
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
        # test_index_non_na is used to hold test nodes whose two-hops labeled neighbors are not empty
        #test_index_non_na = []
        for j in test_index:
        # if it is the first iteration, which is reduced to the 2-hop method
            # update theta[j] according to the 2-hop update rule

            train_set = set(train_index)
            j_neighbor = list(G.neighbors(j))
            l_two_hop_neighbor = []
            for l in j_neighbor:
                # get l's neighbor
                l_neighbor = list(G.neighbors(l)) 
                # if l's neighbor is labeled then add it to the two hop neighbor set
                l_two_hop_neighbor = l_two_hop_neighbor + [l_neighbor]
        
            l_two_hop_neighbor = list(itertools.chain.from_iterable(l_two_hop_neighbor)) 
             
            two_hop_neighbor = []
            #two_hop_neighbor = [i for i in l_two_hop_neighbor if theta0[i]!=None]
            two_hop_neighbor = [i for i in l_two_hop_neighbor]

            
            # break into two parts
            two_hop_labeled_neighbor = []
            two_hop_unlabeled_neighbor = []
            two_hop_labeled_neighbor = [i for i in two_hop_neighbor if i in train_set]
            two_hop_unlabeled_neighbor = [i for i in two_hop_neighbor if i not in train_set]
            
            if len(two_hop_neighbor) != 0:
                #test_index_non_na.append(j)
#                theta[j] = np.sum(theta0[ii] for ii in two_hop_neighbor)/len(two_hop_neighbor)
                lambda_j = len(two_hop_labeled_neighbor)/len(two_hop_neighbor)
                 #lambda_j = (0.85)**(t-1)
                #lambda_j = np.exp(-len(two_hop_labeled_neighbor)/len(two_hop_neighbor))*(len(two_hop_labeled_neighbor)/len(two_hop_neighbor))**(t-1)
#                 #lambda_j = len(two_hop_labeled_neighbor)/len(two_hop_neighbor)
                # with only originally unlabeled neighbors
                if len(two_hop_labeled_neighbor) == 0:
                    theta[j] = (1-lambda_j)* np.sum(theta0[jj] for jj in two_hop_unlabeled_neighbor)/len(two_hop_unlabeled_neighbor)
                    
                elif len(two_hop_unlabeled_neighbor) == 0:
                    theta[j] = lambda_j* np.sum(theta0[jj] for jj in two_hop_labeled_neighbor)/len(two_hop_labeled_neighbor)
                else:   
                    part1 = np.sum(theta0[ii] for ii in two_hop_labeled_neighbor)
                    part2 = np.sum(theta0[jj] for jj in two_hop_unlabeled_neighbor)
                    theta[j] = lambda_j* part1/len(two_hop_labeled_neighbor) + (1-lambda_j)* part2/len(two_hop_unlabeled_neighbor)
                
               

        # perform accuracy analysis
        preference_by_class_matrix =np.zeros((len(theta),2))
        preference_by_class_matrix[:,1] = theta
        preference_by_class_matrix[:,0] = np.subtract(1, preference_by_class_matrix[:,1]) 
        
        
        # test_index_non_na = np.array(test_index_non_na)
        # micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update)),preference_by_class_matrix[test_index_non_na,:][:,1]-preference_by_class_matrix[test_index_non_na,:][:,0],average='micro'))
        # wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update)),
        #                                                preference_by_class_matrix[test_index_non_na,:][:,1]-preference_by_class_matrix[test_index_non_na,:][:,0],average='weighted'))
              

        # ## f1-score version
        # y_true = label_binarize(gender_y_update[test_index_non_na],np.unique(gender_y_update))
        # #print(preference_by_class_matrix[test_index_non_na,:][:,1].size)
        # y_pred = ((preference_by_class_matrix[test_index_non_na,:][:,1]) > accuracy_score_benchmark)+0
        # accuracy.append(f1_score(y_true, y_pred, average='macro'))

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
