# FYS-STK-project3

We have developed several separate codes, which process and analyze the data set via XGBoost, Random Forest and manual code for Neural Network. We have also used Keras implementation of Siamese and Triplet Neural Network. We have unified all the codes in one pipeline, which is intended to be highly configurable. The separate codes can be found in the "code/separately" folder, and the resultant unified code can be found in the "code/unified" folder. To run the unified code, you need to run file main.py, and ParamFile.yaml must be configured beforehand.

Some comments:
- "DataPath" variable inside Parameter File is for the siamese neural network (and its comparison with keras feed forward and triplet); the paths for the main research questions are indicated inside data_processing.py as an urls;
- to run random forest for the main reserach question, please indicate rf_main; to compare results with siamese, rf_side;
- you can run the entire pipeline with by indicating type: 'all' inside parameter file, but do not forget to configure neural network beforehand;
- The codes in R language are for plotting the final comparison between XGBoost and Random Forest and it outputs several plots.
