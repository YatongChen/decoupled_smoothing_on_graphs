# modified from Kristen's code
## assumes k=2 class set-up
## p_in = [p_in_1, p_in_2]
def create_affiliation_model_temp(average_node_degree,
                                  lambda_block_parameter,
                                  dispersion_parameter_vect,
                                  class_size_vect):
    # total number of nodes
    N = np.sum(class_size_vect)
    ### BLOCK STRUCTURE
    ## define p_in; p_out： following equation (78) and (79) on page 29 of supplyment materials
    p_in = (lambda_block_parameter * average_node_degree)/N
    #print('p_in: ', p_in)
    #previous parameterization
    denominator = []
    for j in range(len(class_size_vect)):
        denominator.append(class_size_vect[j] * class_size_vect[~j])
    denom = np.sum(denominator)
    p_out = (average_node_degree * N - np.sum(class_size_vect**2 * p_in))/denom
    #print('p_out: ', p_out)
    #print('')

    ## Expected Degree Sequence for nodes in class 1,2,...k
    ## Generates in-class degree sequence and out-class sequence
    in_class_list = []
    out_class_list = []
    for j in range(len(class_size_vect)):
        #intent here is to iterate through each class
        #and important -- assumes a specific data format for input dispersion_parameter_vect
        (in_class, out_class) = create_expected_degree_sequence(class_size_vect[j],p_in,p_out,dispersion_parameter_vect[j][0], dispersion_parameter_vect[j][1])
        in_class_list.append(in_class)
        out_class_list.append(out_class)

    # What is expected prob matrix??
    expected_prob_matrix=np.zeros((N,N))
    for i in range(len(class_size_vect)):
        for j in range(len(class_size_vect)):
            idx = np.sum(class_size_vect[0:i])
            jdx = np.sum(class_size_vect[0:j])
            if i==j:
                expected_prob_matrix[idx:idx+class_size_vect[j],jdx:jdx+class_size_vect[j]] = in_class_matrix(in_class_list[j])/(class_size_vect[j]**2*p_in)
            else:
                out = out_class_matrix(out_class_list[i], out_class_list[j])/(class_size_vect[i]*class_size_vect[j]*p_out)
                if j<i:
                    expected_prob_matrix[idx:idx+class_size_vect[i],jdx:jdx+class_size_vect[j]] = out
                if i<j:
                    expected_prob_matrix[idx:idx+class_size_vect[i],jdx:jdx+class_size_vect[j]] = out
    
    #A_ij_tmp = np.matrix(map(bernoulli.rvs,expected_prob_matrix))
    f = np.vectorize(bernoulli.rvs)
    A_ij_tmp = np.matrix(f(expected_prob_matrix))
    Adj_corrected = np.matrix(np.triu(A_ij_tmp, k=0) + np.transpose(np.triu(A_ij_tmp, k=1)))
  
    f2 = np.vectorize(np.tile)
    Membership = np.concatenate(list(map(np.tile,np.array(range(len(class_size_vect))), class_size_vect)),axis=0)

    return( Adj_corrected, Membership)
