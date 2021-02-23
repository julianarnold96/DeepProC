# DeepProC - a deep learnig model for proteasomal cleavage prediciton
Implementation of my Master's Thesis - „Elucidation of Proteasomal Cleavage Pattern from HLA peptidome data through Machine Learning”

DeepProC is a deep learnig model for proteasomal cleavage prediciton. Accurate cleavage predictions could further increase the effectiveness of epitope-based vaccines. 

There come several difficulties with peptidome data:
1) negative samples (non-cleavage sites) remain unknown a only cleavage sites can measured -> negative labels need to be generated artificially, leaving us with an uncertainty in the negative labels
2) one measured epitope can stem from multiple different parent proteins leaving us with multiple-instance learning problem 

Solutions:
1) label-flipping noise layer to account for the uncertainty in
negative labels
2) attention-based multiple instance learning to deal with multiple
parent proteins

This results in a network architecture as follows:

