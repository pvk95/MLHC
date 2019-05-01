ECG project, done with Keras
=============================

# MIT-BIH arrhytmia database
* 5 categories:
    - Classes: [’N’: 0, ‘S’: 1, ‘V’: 2, ‘F’: 3, ‘Q’: 4]
* 110k samples
* CNN trained by kachuee
* Two step-process:
    - train the arrhythmia clf with CNN and 13 weight layers
    - train the MI predictor: FC with 2 layers, 32 units each
* eu


# PTB ECG database
* very high freq measurements
* 2 categories

# Datasets are published on kaggle:
* https://www.kaggle.com/shayanfazeli/heartbeat
* include images and plots
* only the 2nd lead is used, resampled at 125 Hz, whereas the original data is from 12 leads, 360 and 1000 Hz (mit-bih and ptb)

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
    - RNN many outputs (at each node) or only output from last, return_sequence=False

Misc:
* Callbacks in Keras, ModelCheckpoint, EarlyStopping -- stop if performance in training is not increasing...
* 

# Useful to know:
import sys
sys.path.insert(0, '/Users/mfilipav/google_drive/eth_school/19_spring/ml4h/ML4HC/exercise3')
sys.path.insert(0, '/Users/mfilipav/google_drive/eth_school/19_spring/ml4h/ML4HC/exercise_data')

