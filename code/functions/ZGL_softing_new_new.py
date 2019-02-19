# soft smoothng method: 
# with input w to be the weight
def ZGL_softing_new_new(w, adj_matrix_tmp,gender_dict, attribute, percent_initially_unlabelled, num_iter, cv_setup):
    exec(open("create_graph.py").read())
    #exec(open("/Users/yatong_chen/Google Drive/research/DSG_empirical/code/functions/create_graph.py").read())
    (graph, gender_y)  = create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes')
    percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)    
    mean_accuracy = []
    se_accuracy = []

    mean_micro_auc = []
    se_micro_auc = []

    mean_wt_auc = []
    se_wt_auc = []
    #(graph, gender_y)  = create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes')

    n = len(gender_y)
    keys = list(graph.node)
    for i in range(len(percent_initially_labelled)):
        print(percent_initially_unlabelled[i]) 
        (graph, gender_y)  = create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes')
        if cv_setup=='stratified':
            k_fold = StratifiedShuffleSplit(n_splits=num_iter,test_size=percent_initially_unlabelled[i],
                                                         random_state=1)
    
        else:
            k_fold = cross_validation.ShuffleSplit(n_splits=num_iter,
                                               test_size=percent_initially_unlabelled[i],
                                               random_state=None)
        accuracy = [] 
        micro_auc = []
        wt_auc = []
    
        #w = 10000
        for train_index, test_index in k_fold.split(keys, gender_y):
            (graph, gender_y)  = create_graph(adj_matrix_tmp,gender_dict,'gender',0,None,'yes') 
            # reorder the train_index in ascend order
            #print(train_index)
            original_train_index = list(np.sort(train_index))
            train_index = list(np.sort(train_index))
            len_train_index = len(train_index)
            
            test_index = list(np.sort(test_index))
            
            original_test_index = list(np.sort(test_index))
            #print(original_test_index)
            for j in range(n):
                if j in train_index:
                    new_node_posit = train_index.index(j) + n
                    graph.add_node(new_node_posit)
                    graph.add_edge(j,new_node_posit, weight = w)
                    # add attribute to new node 
                    graph.nodes[new_node_posit][attribute] = graph.nodes[j][attribute] 
            # gender for new graph
            gender_y_update = list(nx.get_node_attributes(graph, attribute).values())
            W_unordered = np.array(nx.adjacency_matrix(graph).todense())
            new_n = n + len(train_index)
              # see how many classes are there and rearrange them
            classes = np.sort(np.unique(gender_y_update))
            class_labels = np.array(range(len(classes)))
            # relabel membership class labels - for coding convenience
            # preserve ordering of original class labels -- but force to be in sequential order now
            gender_y_update = np.array(gender_y_update)
            for k in range(len(classes)):
                gender_y_update[gender_y_update == classes[k]] = class_labels[k]
      
      
            # get new train_index
            train_index = list(range(n, new_n))
            # get new test_index
            test_index = list(range(n))
            idx = np.concatenate((train_index, test_index)) # concatenate train + test = L + U
            # rearrange the column of W matrix to be train + test = L + U 
            W = np.reshape([W_unordered[row,col] for row in np.array(idx) for col in np.array(idx)],(new_n,new_n))

            #fl: L*c(size) label matrix from ZGL paper
            original_test_index_new_position = [x + len_train_index for x in original_test_index]
            original_test_index_new_position = np.array(original_test_index_new_position)
            
            
            train_labels = np.array([np.array(gender_y_update)[id] for id in train_index]) # resort labels to be in same order as training data
            classes_train = np.sort(np.unique(train_labels))
            ##get the approximate ratio of the max class labels
            accuracy_score_benchmark = np.mean(np.array(train_labels) == np.max(class_labels))
            # get the fl label vector from ZGL paper
            fl =np.array(np.matrix(label_binarize(train_labels,list(classes_train) + [np.max(classes_train)+1]))[:,0:(np.max(classes_train)+1)]) 
            
    
            # record testing gender labels for comparing predictions -- ie ground-truth
            true_test_labels = np.array([np.array(gender_y_update)[id] for id in test_index])
            classes_true_test = np.sort(np.unique(true_test_labels))
            fu_truth =np.array(np.matrix(label_binarize(true_test_labels,list(classes_true_test) + [np.max(classes_true_test)+1]))[:,0:(np.max(classes_true_test)+1)])

            
            l = len(train_index) # number of labeled points
            u = len(test_index) # number of unlabeled points

            ## compute Equation (5) in ZGL paper
            W_ll = W[0:l,0:l]
            W_lu = W[0:l,l:(l+u)]
            W_ul = W[l:(l+u),0:l]
            W_uu = W[l:(l+u),l:(l+u)]
            # get the D matrix(numpy/scipy are different)
            D = np.diag(np.sum(W, axis=1))
            D_ll = D[0:l,0:l]
            D_lu = D[0:l,l:(l+u)]
            D_ul = D[l:(l+u),0:l]
            D_uu = D[l:(l+u),l:(l+u)]
            # harmonic_fxn is just fu
            harmonic_fxn =  np.dot(np.dot(np.linalg.inv(np.matrix(np.subtract(D_uu, W_uu))),np.matrix(W_ul)), np.matrix(fl))
            #print(harmonic_fxn)
            #print(harmonic_fxn)
            test_index = np.array(test_index)
            original_test_index_new_position = np.array(original_test_index_new_position)
        
            gender_y_update = np.array(gender_y_update)
            
            
            #print(true_test_labels==gender_y_update[test_index])
            
            # if the classifications are greater than 2
            
            if len(np.unique(gender_y_update)) > 2:
                row_idx = np.array(list(original_test_index))
                a = harmonic_fxn[row_idx[:, None], ]
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update)),a,average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update)),a,average='weighted'))
                accuracy.append(metrics.accuracy_score(label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update)),a))
            # if there are only two types
            else:
                row_idx = np.array(list(original_test_index))
                #print(row_idx)
                a0 = harmonic_fxn[row_idx[:, None], 0]
                a1 = harmonic_fxn[row_idx[:, None], 1]
                #print(len(gender_y_update))
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update)),a1-a0,average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update)),a1-a0,average='weighted'))

                y_true = label_binarize(gender_y_update[original_test_index],np.unique(gender_y_update))
                y_pred = (a1 > accuracy_score_benchmark) + 0
                accuracy.append(f1_score(y_true, y_pred,average='macro', sample_weight=None))
    
        # get the mean and standard deviation
        mean_accuracy.append(np.mean(accuracy))  
        se_accuracy.append(np.std(accuracy)) 
        
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))

    
    return(mean_accuracy,se_accuracy,mean_micro_auc,se_micro_auc,mean_wt_auc,se_wt_auc)
