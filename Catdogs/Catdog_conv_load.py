from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



network = load_model('cats_and_dogs_small_1.h5')

validation_dir = "C:\\Users\\anubhav\\Desktop\\Projects\\chollet\\cat_dog_data_small\\test"
test_datagen = ImageDataGenerator(rescale=1./ 255)
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size= 20,
        class_mode = 'binary')

loss,acc = network.evaluate_generator(validation_generator)
print("Loss is: ",loss)
print("Accuracy is: ",acc)




