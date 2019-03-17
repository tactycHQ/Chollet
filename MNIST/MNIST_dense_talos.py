#First keras program
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
import talos as ta

#Loading Data
#Train_images has shape (60000,28,28)
#Train_labels has shape (60000)
def data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images,train_labels,test_images,test_labels

#Params
p={'lr':[0.1,0.01]}

#Defining Network and adding Dense Layers
#Compiling Network
def create_model(train_images,train_labels,test_images,test_labels,params):
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    rms=optimizers.RMSprop(lr=params['lr'],rho=0.9,epsilon=0.9,decay=0.0)
    network.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['acc'])

    history=network.fit(train_images,
                        train_labels,
                        validation_split=0.33,
                        epochs=5,
                        batch_size=128)

    return history,network


x_train,y_train,x_test,y_test=data()
#mnist_model=create_model(x_train,y_train,x_test,y_test,params)
t=ta.Scan(x=x_train,
          y=y_train,
          model=create_model,
          params=p,
          dataset_name='mnist',
          experiment_no='3')
r=ta.Reporting('mnist_3.csv')
