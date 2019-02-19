# decoupled_smoothing_on_graph
## Replication code: "Decoupled smoothings on graphs"


### Documentation

This repository contains all the correponding code to replicate the figures in "Decoupled smoothing on graphs". We provide links to the datasets (Facebook100) in the data sub-folder.


### Directions

This repository set-up assumes that the FB100 (raw .mat files) have been acquired and are saved the data folder. Here are the directions:

1. Save raw files in data

2. Update global file-paths in 'functions/define_paths.py' to your local directory settings. #?????????/
    
3. Run code which is briefly described below:
   * soft_smoothing/ - includes notebooks for code related to simulations for the soft smoothing part (figure 2).
   * decouple_smoothing(compared with other methods)/ - includes all relevant code that compare decoupled smoothing with the other methods (Figure 3)
   * hard_smoothing_regularization/ - includes all relevant code that related to iterative hard smoothing and regularization (Figure 4)
   * decoupled_smoothing_regularization/ - includes all relevant code that related to iterative decoupled smoothing and regularization (Figure 5)
   * functions/ - all helper functions that are required by the main codes.

All random number generators used in the analysis have been seeded deterministically to produce persistent cross-validation folds and thereby consistent results when re-running the analysis. The code for generating random graphs (sampled from the overdispersed stochastic block model) is not deterministically seeded. All code was written and tested for Python 3.6 with versions for the following main Python libraries:  networkx (2.2), numpy (1.15.4), sklearn (0.20.1), matplotlib(3.0.2), scipy(1.1.0), pandas (0.23.4). The code has know incompatibilities with Python 2.x and with networkx 1.x.

### Authors
* Yatong Chen, yatong@stanford.edu
* Kristen M. Altenburger, kaltenb@stanford.edu
* Alex Chin, ajchin@stanford.edu
* Johan Ugander, jugander@stanford.edu

