def decoupling_regularization_prepare(graph, sigma_square,lambda_input):
    # get W matrix, Z(for row sum) and Z_prime(for col sum), and A_tilde
    # get matrix W:
    A = np.array(nx.adjacency_matrix(graph).todense()) 
    d = np.sum(A, axis=1)
    D = np.diag(d)
    n = len(D)
    # Alternative way(19): set Sigma_square = sigma_square./d, where sigma_square is fixed
    Sigma_square = np.divide(sigma_square,d)


    Sigma = np.diag(Sigma_square)
    W = np.dot(A,inv(Sigma))
    lambda_diag = lambda_input*np.ones(n)
    lambda_diag_matrix = np.diag(lambda_diag)
    W = W+lambda_diag_matrix
    
    w_col_sum = np.sum(W, axis=0)
    w_row_sum = np.sum(W, axis=1)
    Z_prime = np.diag(w_col_sum)
    Z = np.diag(w_row_sum)

    A_tilde = np.dot(np.dot(W,inv(Z_prime)),np.transpose(W))
    
    # create 

    return (A_tilde)