import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

#Define variables
step_radians = 0.01
timesteps = 100
steps_future = 2
epochs =5
x = np.arange(0,4*math.pi,step_radians)
future_x=np.arange(4*math.pi,6*math.pi,step_radians)
y=np.sin(x)+5
future_y=np.sin(future_x)+5
mean = y.mean(axis=0)
std = y.std(axis=0)
print(mean)
print(std)
y=(y-mean)/std
future_y=(future_y-mean)/std

def create_data(data,timesteps=100):
    x_train=[]
    y_train=[]
    for i in range(timesteps+1,len(data)):
        x_train.append(data[i-timesteps:i])
        y_train.append(data[i])
    x_train=np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], timesteps, 1))
    print('X_train shape is',x_train.shape)

    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (x_train.shape[0], 1))
    print('y_train shape is', y_train.shape)

    batch_size=x_train.shape[1]
    return x_train,y_train,batch_size



def create_model(x_train,y_train,timesteps,batch_size):
    network = Sequential()
    network.add(layers.GRU(32,return_sequences=False,input_shape=(timesteps,1)))
    network.add(layers.Dense(1))
    network.compile(optimizer='adam',
                    loss='mse',
                    )
    print(network.summary())
    history = network.fit(x_train,y_train,validation_split=0.2,epochs=epochs,batch_size=batch_size,verbose=True)
    return network, history


def plot_results(network,history):
    y_pred = network.predict(x_test)
    y_pred_denorm = y_pred*std+mean


    plt.figure()
    plt.subplot(211)
    plt.plot(range(1, epochs + 1), loss, 'bo', label='Training loss')
    plt.plot(range(1, epochs + 1), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(y_test, 'g', label='Actuals')
    plt.plot(y_pred, 'b', label='Predictions')
    plt.legend()
    plt.title('Actuals vs Predicted')
    plt.show()

if __name__ == '__main__':
    x_train,y_train,batches =create_data(y)

    network,history=create_model(x_train,y_train,timesteps,batches)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("Loss on training data is",loss[-1])
    print("Loss on val data is",val_loss[-1])

    x_test, y_test, test_batch_size = create_data(future_y)
    eval_results = network.evaluate(x_test, y_test)

    print("Loss on test data is",eval_results)

    plot_results(network,history)




