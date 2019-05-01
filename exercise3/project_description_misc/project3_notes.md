ECG project, done with Keras
=============================

# MIT-BIH arrhytmia database
* eu
* eu


# PTB ECG database
* very high freq measurements


# Data pre-proc by: ECG Deep Transferable Repres
* two tasks: 
    - myocardian infarction

# Project
* make visualization, can you see clusters, etc
* Use RNNs in Keras
* Bidirectional RNN, Transformer
* Eventually do ensamble with weighted average of 2-3 models
* Transfer learning, 3 options
    - 1st try to transfer from bih to ptb
    - 2nd train two together
    - 3rd 
* maybe use masking layer?
* RNN many outputs (at each node) or only output from last, return_sequence=False

Misc:
* Callbacks in Keras, ModelCheckpoint, EarlyStopping -- stop if performance in training is not increasing...
* 
