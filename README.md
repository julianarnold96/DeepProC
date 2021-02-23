# DeepProC - a deep learnig model for proteasomal cleavage prediciton
Implementation of my Master's Thesis - „Elucidation of Proteasomal Cleavage Pattern from HLA peptidome data through Machine Learning”

DeepProC is a deep learnig model for proteasomal cleavage prediciton. Accurate cleavage predictions could further increase the effectiveness of epitope-based vaccines. 

## Difficulties with peptidome data:
1) negative samples (non-cleavage sites) remain unknown as only cleavage sites can be measured -> negative labels need to be generated artificially, leaving us with an uncertainty in the negative labels
2) one measured epitope can stem from multiple different parent proteins leaving us with multiple-instance learning problem 

## Solutions:
1) label-flipping noise layer to account for the uncertainty in
negative labels
2) attention-based multiple instance learning to deal with multiple
parent proteins

## Network:
After hyperparamter optimization this results in a network architecture as follows:

<p align="center">
<img width="669" alt="networkarchitecture" src="https://user-images.githubusercontent.com/56801215/108819037-96d76f00-75ba-11eb-88e8-56757a6a3665.png">
</p>

In the first steps, negatives are generated using one of two mechanisms:

<p align="center">
<img width="707" alt="negatives" src="https://user-images.githubusercontent.com/56801215/108819335-051c3180-75bb-11eb-84e8-f3c35f968ca1.png">
</p>

Followed by the length regularization layer scaling the input down to the optimal size needed for accurate predictions, with a CNN encoding the information contained in the peptide sequences next up. Afterwards, the attention weights between the epitope and the mutliple possible parent proteins are computed, and fed into the predictor head, a smaller CNN. The resulting noisy predictions are then possibly flipped in the noise layer, before the final predictions are made:

<p align="center">
<img width="939" alt="noiselayer" src="https://user-images.githubusercontent.com/56801215/108820245-321d1400-75bc-11eb-89ab-152ce33ff3fd.png">
</p>


## Results:

The accuracy of DeepProC was pretty high, especially compared to state-of-the-art proteasomal cleavage predictors and the Ridge classifier fit as a baseline:

<p align="center">
<img width="1009" alt="vivocomp" src="https://user-images.githubusercontent.com/56801215/108821006-4dd4ea00-75bd-11eb-97ac-2ac4df5bdc30.png">
</p>
