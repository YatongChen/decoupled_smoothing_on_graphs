def ZGL(adj_matrix_gender,gender_y,percent_initially_unlabelled, num_iter, cv_setup):
    W_unordered = np.array(adj_matrix_gender)

    percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)    
    mean_accuracy = []
    se_accuracy = []

    mean_micro_auc = []
    se_micro_auc = []

    mean_wt_auc = []
    se_wt_auc = []


    n = len(gender_y)
    # see how many classes are there and rearrange them
    classes = np.sort(np.unique(gender_y))
    class_labels = np.array(range(len(classes)))

    # relabel membership class labels - for coding convenience
    # preserve ordering of original class labels -- but force to be in sequential order now
    gender_y_update = np.copy(gender_y)
    for j in range(len(classes)):
        gender_y_update[gender_y_update == classes[j]] = class_labels[j]
        
    for i in range(len(percent_initially_labelled)):
        print(percent_initially_unlabelled[i]) 
        if cv_setup=='stratified':
            k_fold = StratifiedShuffleSplit(n_splits=num_iter,test_size=percent_initially_unlabelled[i],random_state=1)
    
        else:
            k_fold = cross_validation.ShuffleSplit(n_splits=num_iter,test_size=percent_initially_unlabelled[i],random_state=None)
        accuracy = [] 
        micro_auc = []
        wt_auc = []
    
        for train_index, test_index in k_fold.split(W_unordered, gender_y_update):
            X_train, X_test = W_unordered[train_index], W_unordered[test_index]
            y_train, y_test = gender_y_update[train_index], gender_y_update[test_index]
            #print(train_index)
            idx = np.concatenate((train_index, test_index)) # concatenate train + test = L + U
            # rearrange the column of W matrix to be train + test = L + U 
            W = np.reshape([W_unordered[row,col] for row in np.array(idx) for col in np.array(idx)],(n,n))    
        
            #fl: L*c(size) label matrix from ZGL paper
            train_labels = np.array([np.array(gender_y_update)[id] for id in train_index]) # resort labels to be in same order as training data
            classes_train = np.sort(np.unique(train_labels))
            ##get the approximate ratio of the max class labels
            accuracy_score_benchmark = np.mean(np.array(train_labels) == np.max(class_labels))
            # get the fl label vector from ZGL paper
            fl =np.array(np.matrix(label_binarize(train_labels,list(classes_train) + [np.max(classes_train)+1]))[:,0:(np.max(classes_train)+1)]) 
            # record testing gender labels for comparing predictions -- ie ground-truth
            true_test_labels = np.array([np.array(gender_y_update)[id] for id in test_index])
            classes_true_test = np.sort(np.unique(true_test_labels))
            ground_truth =np.array(np.matrix(label_binarize(true_test_labels,list(classes_true_test) + [np.max(classes_true_test)+1]))[:,0:(np.max(classes_true_test)+1)])
    
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
            # if the classifications are greater than 2
            if len(np.unique(gender_y_update))>2:
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn,average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn,average='weighted'))
                accuracy.append(metrics.accuracy_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn))
            # if there are only two types
            else:
                micro_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn[:,1]-harmonic_fxn[:,0],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(gender_y[test_index],np.unique(gender_y)),harmonic_fxn[:,1]-harmonic_fxn[:,0],average='weighted'))

                y_true = label_binarize(gender_y[test_index],np.unique(gender_y))
                y_pred = ((harmonic_fxn[:,1]) > accuracy_score_benchmark)+0
            
                accuracy.append(f1_score(y_true, y_pred,average='macro', sample_weight=None))
        # get the mean and standard deviation
        mean_accuracy.append(np.mean(accuracy))  
        se_accuracy.append(np.std(accuracy)) 
        
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))
    

    return(mean_accuracy,se_accuracy,mean_micro_auc,se_micro_auc,mean_wt_auc,se_wt_auc)