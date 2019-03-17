from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import metrics
from keras import losses
from keras import regularizers
from keras import optimizers
import matplotlib.pyplot as plt


(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#Define Vectorize Sequences to convert list of data into arrays
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,s in enumerate(sequences):
        results[i,s]=1
    return results

#Data Manipulation
x_train=vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#Validation sets
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

#Model Creation
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#Model Compile
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=['acc'])
print(model.summary())

#Model Fit
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#Model Predict
predict = model.predict(x_test)
print(predict)

#Model Evaluate
eval_results = model.evaluate(x_test,y_test)
print(eval_results)

#Plotting
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
