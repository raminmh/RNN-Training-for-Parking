# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:00:11 2017

@author: mathias
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,Convolution1D,MaxPooling1D,Flatten,Activation
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr  

def read_csv(filename):
    f = open(filename, 'r')
    data = []
    for line in f:
        row = line.split(';')
        data.append(row)
    return data
    
    
def enflate(csv, max_len, pos, x,y):
    add_begin = max_len - len(csv)
    
    for i in range(0,len(csv)):
        insert_pos=i+add_begin
        x[pos][insert_pos][0]=csv[i][1]
        x[pos][insert_pos][1]=csv[i][2]
        x[pos][insert_pos][2]=csv[i][3]
        x[pos][insert_pos][3]=csv[i][4]
        x[pos][insert_pos][4]=csv[i][5]
        x[pos][insert_pos][5]=csv[i][6]
        x[pos][insert_pos][6]=csv[i][7]
        x[pos][insert_pos][7]=csv[i][8]
        x[pos][insert_pos][8]=csv[i][9]
        
        y[pos][insert_pos][0]=csv[i][10]
        y[pos][insert_pos][1]=csv[i][11]
        y[pos][insert_pos][2]=csv[i][12]
        
    # integrate start signal
    for i in range(1,max_len):
        x[pos][i][0]+=x[pos][i-1][0]
        

def plot_stuff(filename, times, pred, real,head):
     # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(times, pred,label='Prediction', color='b',linewidth=2.0)
    ax.plot(times, real,label='Real', color='r',linewidth=2.0,linestyle='--')
    
    ax.set(xlabel='Time (s)', ylabel='Output',title=head)
    ax.grid()
    ax.legend()
    
    fig.savefig(filename)
#    plt.show()
    
    
max_len = 0
sv_data =[]
for i in range(1,21):
    filename = 'supervise_data/sv'+str(i)+'.csv'
    csv = read_csv(filename)
    if(len(csv)> max_len):
        max_len = len(csv)
    sv_data.append(csv)
    
eval_data =[]
for i in range(1,5):
    filename = 'supervise_data/eval'+str(i)+'.csv'
    csv = read_csv(filename)
    if(len(csv)> max_len):
        max_len = len(csv)
    eval_data.append(csv)
    
test_data =[]
for i in range(1,6):
    filename = 'supervise_data/test'+str(i)+'.csv'
    csv = read_csv(filename)
    if(len(csv)> max_len):
        max_len = len(csv)
    test_data.append(csv)
    
x_train = np.zeros([20,max_len,9],dtype=np.float32)
y_train = np.zeros([20,max_len,3],dtype=np.float32)

x_test = np.zeros([5,max_len,9],dtype=np.float32)
y_test = np.zeros([5,max_len,3],dtype=np.float32)

x_eval = np.zeros([4,max_len,9],dtype=np.float32)
y_eval = np.zeros([4,max_len,3],dtype=np.float32)

for i in range(0,20):
    enflate(sv_data[i],max_len,i,x_train,y_train)
    
for i in range(0,4):
    enflate(eval_data[i],max_len,i,x_eval,y_eval)
    
for i in range(0,5):
    enflate(test_data[i],max_len,i,x_test,y_test)
    
""" Reading one, start learning """


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(100,
#model.add(LSTM(80, return_sequences=True,
               input_shape=(max_len, 9),return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.3))
#model.add(LSTM(10, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(80, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(3, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(5, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(Dense(5, activation='tanh'))
#model.add(Dense(50, activation='tanh'))

#model.add(Dense(3, activation='tanh'))

print('Compiling computation graph ...')
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=10, epochs=500,shuffle=True,validation_data=(x_eval, y_eval))

#validation_data=(processed_valid_data, valid_y_one_hot)
y_predict = model.predict(x_test, batch_size=1)


for i in range(0,5):
    ts = np.zeros(max_len,dtype=np.float32)
    mlinear = np.zeros(max_len,dtype=np.float32)
    mlinear_ = np.zeros(max_len,dtype=np.float32)
    mangular = np.zeros(max_len,dtype=np.float32)
    mangular_ = np.zeros(max_len,dtype=np.float32)
    mdone = np.zeros(max_len,dtype=np.float32)
    mdone_ = np.zeros(max_len,dtype=np.float32)
    
    tm=0
    for t in range(0,max_len):
        ts[t]=tm
        tm+=0.1
        mlinear[t]=y_predict[i][t][0]
        mangular[t]=y_predict[i][t][1]
        mdone[t]=y_predict[i][t][2]
        
        mlinear_[t]=y_test[i][t][0]
        mangular_[t]=y_test[i][t][1]
        mdone_[t]=y_test[i][t][2]
    
    fn1 = 'pred/test_'+str(i)+'_linear.png'
    fn2 = 'pred/test_'+str(i)+'_angular.png'
    fn3 = 'pred/test_'+str(i)+'_done.png'
    
    plot_stuff(fn1, ts, mlinear, mlinear_,'Linear Motor Command')
    plot_stuff(fn2, ts, mangular, mangular_,'Angular Motor Command')
    plot_stuff(fn3, ts, mdone, mdone_,'Parking Finished indicator')
    
""" Compute MSE over test files """
Vmse = 0
Wmse = 0
Bmse = 0
#VCor = 0
#WCor = 0
#BCor = 0
for i in range(0,4):
    for t in range(0,max_len):
       Vmse += np.square(y_predict[i][t][0]-y_test[i][t][0]).mean()
       Wmse += np.square(y_predict[i][t][1]-y_test[i][t][1]).mean()
       Bmse += np.square(y_predict[i][t][2]-y_test[i][t][2]).mean()
       #VCor += np.corrcoef(y_predict[i][t][0],y_test[i][t][0])[0,1]
       #WCor += np.corrcoef(y_predict[i][t][1],y_test[i][t][1])[0,1]
       #BCor += np.corrcoef(y_predict[i][t][2],y_test[i][t][2])[0,1]
        #mse += mean_squared_error(y_predict[i][t][0],y_test[i][t][0])
        #mse += mean_squared_error(y_predict[i][t][1], y_test[i][t][1])
        #mse += mean_squared_error(y_predict[i][t][2], y_test[i][t][2])

print('VMSE: '+str(Vmse))
print('WMSE: '+str(Wmse))
print('BMSE: '+str(Bmse))
#print('VCor: '+str(VCor))
#print('WCor: '+str(WCor))
#print('BCor: '+str(BCor))

""" Print result of test5.csv into a file """

of = open('out_trace.csv','w')
for t in range(0,max_len):
    of.write(str(x_test[0][t][0])+';')
    of.write(str(x_test[0][t][1])+';')
    of.write(str(x_test[0][t][2])+';')
    of.write(str(x_test[0][t][3])+';')
    of.write(str(x_test[0][t][4])+';')
    of.write(str(x_test[0][t][5])+';')
    of.write(str(x_test[0][t][6])+';')
    of.write(str(x_test[0][t][7])+';')
    of.write(str(x_test[0][t][8])+';')

    of.write(str(y_test[0][t][0])+';')
    of.write(str(y_test[0][t][1])+';')
    of.write(str(y_test[0][t][2])+';')

    of.write(str(y_predict[4][t][1])+';')
    of.write(str(y_predict[4][t][1])+';')
    of.write(str(y_predict[4][t][2])+'\n')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
