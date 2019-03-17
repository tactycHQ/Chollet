#Cat and Dog Recognition Program

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def PrepData():
    original_dataset_dir = "C:\\Users\\anubhav\\Desktop\\Projects\\chollet\\cat_dog_data\\train\\train"
    base_dir ="C:\\Users\\anubhav\\Desktop\\Projects\\chollet\\cat_dog_data_small"
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir,'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir,'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir,'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir,'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir,'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir,'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir,'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir,'dogs')
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range (1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir,fname)
        dst=os.path.join(train_cats_dir,fname)
        shutil.copyfile(src,dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1400, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1400, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print('Total training cat images:', len(os.listdir(train_cats_dir)))
    print('Total training dog images:', len(os.listdir(train_dogs_dir)))
    print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('Total test cat images:', len(os.listdir(test_cats_dir)))
    print('Total test dog images:', len(os.listdir(test_dogs_dir)))


def ProcessData():
    train_dir = "C:\\Users\\anubhav\\Desktop\\Projects\\chollet\\cat_dog_data_small\\train"
    validation_dir = "C:\\Users\\anubhav\\Desktop\\Projects\\chollet\\cat_dog_data_small\\test"
    train_datagen= ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./ 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size= 20,
        class_mode = 'categorical')

    print("Data has been processed and rescaled")

    return train_generator,validation_generator


def create_model(training_set, validation_set):

    network = models.Sequential()
    network.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(64,(3,3),activation='relu'))
    network.add(layers.MaxPooling2D(2,2))
    network.add(layers.Conv2D(128,(3,3),activation='relu'))
    network.add(layers.MaxPooling2D(2, 2))
    network.add(layers.Conv2D(128, (3, 3), activation='relu'))
    network.add(layers.MaxPooling2D(2, 2))
    network.add(layers.Flatten())
    network.add(layers.Dense(512, activation='relu'))
    network.add(layers.Dense(2, activation='softmax'))

    network.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                    loss='binary_crossentropy',
                    metrics=['acc'])

    history = network.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_set,
        validation_steps=50)

    network.save('cats_and_dogs_small_1.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


#PrepData()
training_set,validation_set=ProcessData()
create_model(training_set,validation_set)
