#First keras program
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe


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

#Defining Network and adding Dense Layers
#Compiling Network
def create_model(train_images,train_labels,test_images,test_labels):

    layer_1_size ={{choice([1024,256,512])}}
    opt_function = {{choice(['adam','rmsprop','sgd'])}}

    network = models.Sequential()
    network.add(layers.Dense(layer_1_size, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer=opt_function,loss='categorical_crossentropy',metrics=['accuracy'])
    network.fit(train_images,
                train_labels,
                validation_split=0.33,
                epochs=5,
                batch_size=128)

    score,acc = network.evaluate(train_images,train_labels,verbose=0)
    print('Test accuracy:',acc)
    out={'loss':-acc,'score':score,'status':STATUS_OK, 'model': network}
    return out

if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          verbose=True,
                                          trials=Trials(),
                                          eval_space=True,
                                          return_space=False)
    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test,verbose=0))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)







