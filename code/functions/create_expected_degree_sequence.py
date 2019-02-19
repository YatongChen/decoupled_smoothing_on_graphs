# modified from Kristen's code
def create_expected_degree_sequence(class_size_val,
                                    p_in_val, p_out_val,
                                    dispersion_val_in, dispersion_val_out):
    ## in-class 
    # create degree sequence: 
    if(dispersion_val_in != 0):
        alpha_in = p_in_val * (1/dispersion_val_in) * (1-dispersion_val_in)
        beta_in = (1-p_in_val) * (1/dispersion_val_in) * (1-dispersion_val_in)
        p_in_dispersed = np.matrix(np.random.beta(alpha_in, beta_in, size=class_size_val))
        in_class_expected_degree = p_in_dispersed * class_size_val # probability of link * number of possible in-LINKS
    # if there are no dispersion value in: then reduce to a SBM model
    if(dispersion_val_in == 0):
        in_class_expected_degree = np.matrix([class_size_val * p_in_val] * class_size_val)

    ## out-class
    if(dispersion_val_out != 0):
        alpha_out = p_out_val * (1/dispersion_val_out) * (1-dispersion_val_out)
        beta_out = (1-p_out_val) * (1/dispersion_val_out) * (1-dispersion_val_out)
        p_out_dispersed = np.matrix(np.random.beta(alpha_out, beta_out, size=class_size_val))
        out_class_expected_degree = p_out_dispersed * class_size_val # probability of link * number of possible out-LINKS
    if(dispersion_val_out == 0):
        out_class_expected_degree = np.matrix([class_size_val * p_out_val] * class_size_val)
    return(in_class_expected_degree, out_class_expected_degree)