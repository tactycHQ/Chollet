#MNIST CNN Reference Model by Anubhav Srivastava
#Layer size optimized  by Hyperas separately
#Optimizer selected as RMSprop by Hyperas
#To do:

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ProgbarLogger
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

#Global Variables
#Hyperas Flag 1 implies model is running in hyperparameter optimization mode. 0 implies model running optimized variables already
hyperas_flag=0

#Data function loads the data into data arrays
#60,000 Trainng images reshaped into 28 x 28 x 1 channel of 10 digits
#10,000 Test images reshaped into 28 x 28 x 1 channel of 10 digits
def data():
    (train_images, train_labels),(test_images, test_labels)=mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    # The function to_Categorical converts vector of integers into one-hot encoded matrixes. So a 4 becomes [0 0 0 0 1 0 0 0 0 0 0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

#Create Model function takes in data and outputs the final model
def create_model(train_images,train_labels,test_images,test_labels):

    network = models.Sequential()
    network.add(layers.Conv2D(128,(3,3), activation='relu', input_shape=(28,28,1)))
    network.add(layers.BatchNormalization())
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(64,(3,3),activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(64,(3,3),activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    #Callbacks instance initiated
    tensorboard = TensorBoard(log_dir='mnist_logs', histogram_freq=0, embeddings_freq=0, write_graph=True)
    progbar = ProgbarLogger()

    #Fit method returns history object. Applied a validation split of 0.33 and Tensorboard callback
    history = network.fit(train_images, train_labels,validation_split=0.33, epochs=5, batch_size=64,verbose=True,callbacks=[tensorboard,progbar])

    #Evaluate method returns loss and accuracy on Test data. Score is the evaluation of loss function. The lower the score the more accurate the prediction
    score, acc = network.evaluate(test_images, test_labels)

    #Prints the accuracy and losses across training, validation and test datasets
    print('train_acc:',history.history['acc'][-1])
    print('val_acc:', history.history['val_acc'][-1])
    print('test_acc:', acc)
    print('train_loss:', history.history['loss'][-1])
    print('val_loss:', history.history['val_loss'][-1])
    print('test_loss:', score)

    #Saving Model
    network.save('MNIST.h5')

    return network, history

#Hyperas model wrapper create function. Returns dictionary of output variables
def create_model_hyperas(train_images,train_labels,test_images,test_labels):
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

#Plotting Function for Validation and Training Datasets
def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)

    plt.figure()
    plt.subplot(211)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    return None

#Main function to initiate the model
if __name__ == '__main__':

#Running optimized model
    if hyperas_flag == 0:
        x_train, y_train, x_test, y_test = data()
        network, history = create_model(x_train, y_train, x_test, y_test)
        plot(history)

#Running in Hyperparameter Optimization mode i.e. find the best hyperparamers
    if hyperas_flag==1:
        print("Running Hyperparameter Optimization with Hyperas")
        best_run, best_model = optim.minimize(model=create_model_hyperas,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              verbose=True,
                                              trials=Trials(),
                                              eval_space=True,
                                              return_space=False)
        print("Evaluation of best performing model:")
        print(best_model.evaluate(x_test, y_test,verbose=0))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)


