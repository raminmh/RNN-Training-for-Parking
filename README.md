# Various-RNN-training-for-reproducing-outputs-of-the-WormNets
[Materials and Methods for submission X] This repository contains all the necessary data, for reproducing the RNN training process and monitoring their performance. 


DATA.MAT file contains input output time series for training the time-delayed neural net (TDNN) and non-linear autoregressive network with exogenous input (NARX)

TrainingNARX.m trains a NARX network.

TrainingTDNN.m trains a TDNN network

SimParking.slx contains neural net models that are already trained and can be tested for their accuracy. 


supervise_data folder, contains data you need to train for the lstm cells. 

lstm_parking.py contains a Keras implementation with a TensorFlow backend for LSTM networks. 




