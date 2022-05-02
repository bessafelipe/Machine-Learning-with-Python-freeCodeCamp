# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
#RPS.py

import keras
import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from numpy import array
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM,GRU,Dropout
from keras.layers import Dense
      
def player(prev_play, opponent_history=[],hist=[],tent=[0]):
  ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

  if prev_play=="":
    prev_play='R'
  
  if hist ==[]:
      previa='R'
      hist.append(previa)

  def trans(vetor):
      for index, value in enumerate(vetor):
        if value == 'R':
          vetor[index] = 0
        if value == 'P':
          vetor[index] = 1
        if value == 'S':
          vetor[index] = 2
      return vetor
        
  def split_x(sequences, n_steps):
    X = list()
    for i in range(len(sequences)):
      end_ix = i + n_steps
      if end_ix > len(sequences):
        break
      seq_x = sequences[end_ix-1, :-1]
      X.append(seq_x)
    return array(X)
      
  def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
      end_ix = i + n_steps
      if end_ix > len(sequences):
        break
      seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
      X.append(seq_x)
      y.append(seq_y)
    return array(X), array(y)

  opponent_history.append(prev_play)
  
  guess = np.random.choice(['R','S','P'],1)[0]

  if len(opponent_history) >= 30 :

    if (opponent_history[len(hist)-1]== ideal_response[hist[len(hist)-1]] and opponent_history[len(hist)-2]== ideal_response[hist[len(hist)-2]]) or (opponent_history[len(hist)-1]== hist[len(hist)-1] and opponent_history[len(hist)-2]== hist[len(hist)-2]):
      tent[0]=1
    else:
      tent[0]=0
    
    in_seq1 = array(trans(opponent_history[1:]))
    in_seq2 = array(trans(hist[1:]))
    out_seq = array((trans(opponent_history[1:])))
  
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1)) 
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    dataset = hstack((in_seq1[:-1], in_seq2[:-1], out_seq[1:]))  
 
    n_steps = 10 
 
    X, y = split_sequences(dataset, n_steps)
    y=np_utils.to_categorical(y)

    n_features = X.shape[2]
    
    test=hstack((in_seq1, in_seq2, out_seq))
    X_test,y_test=split_sequences(test, n_steps)

    if tent[0]==1 or len(opponent_history)%15 ==0 :   
      model = tf.keras.Sequential()
      model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
      model.add(layers.Dense(50, activation='relu',))
      model.add(layers.Dense(3, activation='softmax'))
      simple_adam = keras.optimizers.Adam(0.01)
      model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
      if len(X)>=30:
        X=X[-20:]    
        y=y[-20:]
      model.fit(X, y, epochs=80, verbose=0)
      model.save('gfgModel.h5')
    
    model=load_model('gfgModel.h5')
    test_data = np.array(X_test)[len(X_test)-1][-(n_steps):]
    test_data = test_data.reshape((1, n_steps, n_features))
    predictNextNumber = model.predict(test_data, verbose=0)
    predict=np.argmax(predictNextNumber[0])
    resp=''
    
    if predict == 0:
      resp = "P"
    if predict == 1:
      resp = "S"
    if predict == 2:
      resp = "R"

    guess=resp
    
    if opponent_history[len(hist)-1]== ideal_response[hist[len(hist)-1]] and hist[len(hist)-2]== ideal_response[opponent_history[len(hist)-2]]:
      guess=ideal_response[opponent_history[len(hist)-1]]
  
  hist.append(guess)
  
  return guess
