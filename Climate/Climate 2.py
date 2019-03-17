#Jena Climate GRU Reference Model by Anubhav Srivastava
#GRU Based Model based on 14 dimensions

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ProgbarLogger
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

#Global Variables
#Lookback = Number of timesteps in 1 sliding window. Interval between each timestep is 10 mins, so 720 timesteps is 5 days worth of data in 1 window
#Step = Determines how much to shift between each window. 6 timesteps is 60 minutes.
#Delay = Determines how many timesteps in the future we want to predict. 144 timestep = 24 hours in future
#Future = Determines the width of the future window i.e. also 144 timesteps. So we want to predict 24 hours worth of predictions 24 hours from now
lookback = 720
shift = 6
delay = 144
dim=14
epochs=10
steps_per_epoch=10
val_steps=10
future = 144

jena=pd.read_csv('jena_climate_2009_2016/jena_climate_2009_2016.csv')
dataset = jena.iloc[:,1:15].values
dates_data=jena.iloc[:,0].values.reshape(-1,1)

#Normalize Data
data_mean=dataset.mean(axis=0)
data_std=dataset.std(axis=0)
target_mean=data_mean[1]
target_std=data_std[1]
dataset = (dataset-data_mean)/data_std

# For debugging,
# print("Target Mean: ", target_mean)
# print("Target Std: ", target_std)

#Main function
def main():
    inputs, outputs, dates = build_data(dataset, lookback=lookback, steps=shift, dates_data=dates_data)
    x_train,y_train,date_train=create_model_inputs(inputs=inputs,outputs=outputs,dates=dates,min_index=0,max_index=10000)
    x_test, y_test, date_test = create_model_inputs(inputs=inputs, outputs=outputs, dates=dates, min_index=10000,max_index=10002)

    # For debugging
    # print("x_train Shape: ", x_train.shape)  # (10000 samples, 720 timesteps, 14 dim)
    # print("y_train Shape: ", y_train.shape)  # (100000 samples, 720 timesteps, 14 dim)
    # print("date_train Shape: ", date_train.shape)  # (100000 samples, 720 timesteps, 14 dim)
    # print("x_test Shape: ", x_test.shape) #(100 samples, 720 timesteps, 14 dim)
    # print("y_test Shape: ", y_test.shape) #(100 samples, 1 dim)
    # print("date_test Shape: ", date_test.shape)  # (100000 samples, 720 timesteps, 14 dim)

    history,y_pred=build_model(x_train,y_train,x_test,y_test)

    plot(history,y_test,y_pred,date_test)

#Data function loads the data into Pandas dataframe - non generator version
#Every 6 timesteps, select 720 timesteps in the past and slit into the features vector.
#Then select 144 timesteps in the future and slot into target vector
def build_data(data,lookback,steps,dates_data):

    inputs=[]
    outputs=[]
    dates=[]

    targets = data[:,1] #Extracts temperature column from data
    targets = targets.reshape(-1, 1)
    features= data #Dataset renamed to features. This is a (420551, 14) matrix
    window_start=lookback
    window_end=len(features)-delay-lookback

    #For debugging
    # print("Window Start ",window_start)
    # print("Window End ", window_end)

    for i in range(window_start,window_end,shift):
        feature_slice=features[i-lookback:i,:]
        targets_slice=targets[i+delay:i+delay+lookback,:]
        date_slice=dates_data[i+delay:i+delay+lookback,:]
        inputs.append(feature_slice)
        outputs.append(targets_slice)
        dates.append(date_slice)

    #Creates Input array of features and output array of all targets
    inputs=np.array(inputs)
    outputs = np.array(outputs)
    dates = np.array(dates)

    #For debugging
    # print("Inputs Shape: ", inputs.shape)
    # print("Outputs Shape: ", outputs.shape)
    # print("Dates Shape: ",dates.shape)

    return inputs,outputs,dates

# Creates selected input and output x_train and y_train vectors
def create_model_inputs(inputs,outputs,dates,min_index,max_index):
    x_train = inputs[min_index:max_index, :]
    y_train = outputs[min_index:max_index, :]
    date_train = dates[min_index:max_index]

    return x_train,y_train,date_train

#Build GRU Model
def build_model(x_train,y_train,x_test,y_test):

    model=Sequential()
    model.add(layers.GRU(16,input_shape=(lookback,dim),dropout=0.1,return_sequences=True))  # Recall GRU layer input shape does not need batch_size definition. The input_shape is simply (timesteps, dim)
    # model.add(layers.GRU(16, dropout=0.1,return_sequences=True))  # Recall GRU layer input shape does not need batch_size definition. The input_shape is simply (timesteps, dim)
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mae')
    print(model.summary())

    history=model.fit(x_train,y_train,validation_split=0.2,verbose=1,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=val_steps)
    y_pred = model.predict(x_test)

    return history,y_pred

#Plot Model
def plot(history,y_test,y_pred,date_test):

    y_test = y_test.flatten().reshape(-1, 1)
    y_pred = y_pred.flatten().reshape(-1, 1)
    date_test = date_test.flatten().reshape(-1, 1)
    y_test_denorm = y_test * target_std + target_mean
    y_pred_denorm = y_pred * target_std + target_mean

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    #For debugging
    # print("y_test_denorm Shape ", y_test_denorm.shape)
    # print("y_pred_denorm Shape ", y_pred_denorm.shape)
    # print("date_test_denorm Shape ", date_test.shape)
    print('Training loss (Denormalized)',loss[-1]*target_std)
    print('Validation loss (Denormalized)', val_loss[-1]*target_std)

    plt.figure()
    plt.subplot(211)
    plt.plot(range(1, epochs + 1), loss, 'bo', label='Training loss')
    plt.plot(range(1, epochs + 1), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(y_test_denorm, 'g', label='Actuals')
    plt.plot(y_pred_denorm, 'b', label='Predictions')
    plt.legend()
    plt.title('Actuals vs Predicted')
    plt.show()

if __name__ == '__main__':
    main()

