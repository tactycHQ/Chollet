from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np


(train_data, train_targets), (test_data, test_targets)=boston_housing.load_data()

#Normalizing Data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

print(train_data.shape)
print(train_targets.shape)

#Build model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#Production Model
model = build_model()
print(model.summary())
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print(test_mae_score)
