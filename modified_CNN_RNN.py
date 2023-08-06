import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling1D, RepeatVector, SimpleRNN,
                          TimeDistributed)
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("indian_liver_patient.csv")

print(data.describe().T)  #Values need to be normalized before fitting. 


print(data.isnull().sum())
#data = data.dropna()

print(data['Albumin_and_Globulin_Ratio'].mean())
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0.947)
print(data.isnull().sum())

data.describe()

data = data.drop('Gender',axis=1)

from sklearn.utils import resample
print(data['Dataset'].value_counts())

#Separate majority and minority classes
data_majority = data[data['Dataset'] == 1]
data_minority = data[data['Dataset'] == 2]

# Upsample minority class and other classes separately

data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=416,    # to match average class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
print(data_upsampled['Dataset'].value_counts())  


X = data_upsampled.drop('Dataset',axis=1)
Y = data_upsampled['Dataset']

from sklearn.model_selection import train_test_split

X0=[]
X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
X7=[]
X8=[]
Y=[]

for i in range(0, data.shape[0]-10):
    X0.append(data_upsampled.iloc[i:i + 10,0])
    X1.append(data_upsampled.iloc[i:i + 10,1])
    X2.append(data_upsampled.iloc[i:i + 10,2])
    X3.append(data_upsampled.iloc[i:i + 10,3])
    X4.append(data_upsampled.iloc[i:i + 10,4])
    X5.append(data_upsampled.iloc[i:i + 10,5])
    X6.append(data_upsampled.iloc[i:i + 10,6])
    X7.append(data_upsampled.iloc[i:i + 10,7])
    X8.append(data_upsampled.iloc[i:i + 10,8])
    Y.append(data_upsampled.iloc[i:i + 10,9])

print(type(X5))

X0, X1, X2, X3, X4, X5, X6, X7, X8, Y = np.array(X0), np.array(X1), np.array(X2),np.array(X3),np.array(X4),np.array(X5),np.array(X6),np.array(X7),np.array(X8),np.array(Y)

print(Y.shape)
#(573,10)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
X0 = sc.fit_transform(X0)
X1 = sc.fit_transform(X1)
X2 = sc.fit_transform(X2)
X3 = sc.fit_transform(X3)
X4 = sc.fit_transform(X4)
X5 = sc.fit_transform(X5)
X6 = sc.fit_transform(X6)
X7 = sc.fit_transform(X7)
X8 = sc.fit_transform(X8)
Y = sc.fit_transform(Y)

#Till now we don't have 3D input for LSTM layer, so we are going to stack it in axis 2
X = np.stack([X0, X1, X2, X3, X4, X5, X6, X7, X8], axis = 2)

print(X.shape)
#(573,10,9)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

print(x_train.shape[1])
print(x_train.shape[2])

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Conv1D(filters=64, kernel_size=6, activation='relu'))
model.add(MaxPooling1D(2))

model.add((SimpleRNN(100, return_sequences=True,activation='swish')))

model.add(Dropout(0.5))

#Final layers
model.add(Dense(100, activation='swish'))
model.add(Dense(1, activation='swish'))

model.summary()

model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=31, batch_size=16, verbose=1, shuffle=False)



