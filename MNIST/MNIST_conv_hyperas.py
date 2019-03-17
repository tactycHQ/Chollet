#MNIST Conv with Hyperas and Plotting
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim



def data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


def create_model(train_images,train_labels,test_images,test_labels):
    opt_function = {{choice(['adam', 'rmsprop', 'sgd'])}}
    a = {{choice([32,64,128])}}
    b = {{choice([32,64,128])}}
    c = {{choice([32,64,128])}}
    d = {{choice([32,64,128])}}

    network = models.Sequential()
    network.add(layers.Conv2D(a,(3,3), activation='relu', input_shape=(28,28,1)))
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(b,(3,3),activation='relu'))
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(c,(3,3),activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(d, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer=opt_function,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = network.fit(train_images, train_labels,validation_split=0.33, epochs=5, batch_size=64,verbose=False)
    print('train_acc:',history.history['acc'])
    score, acc = network.evaluate(test_images, test_labels)
    print('test_acc:', acc)
    out = {'loss': -acc, 'score': score, 'status': STATUS_OK, 'model': network}
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

#Plotting
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(acc)+1)
#
# plt.figure()
# plt.subplot(211)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.subplot(212)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
