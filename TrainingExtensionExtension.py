import numpy as np
import random
from random import choice
import tensorflow as tf
import math
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.utils import to_categorical
import pickle
from skimage.transform import resize  # Import resize function from skimage.transform


# Load CIFAR-10 dataset from cifar-10-batches-py directory
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_train_data(data_dir):
    data_batches = []
    labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch = load_cifar_batch(batch_file)
        data_batches.append(batch[b'data'])
        labels += batch[b'labels']
    return np.concatenate(data_batches), np.array(labels)

def load_cifar10_test_data(data_dir):
    test_batch_file = os.path.join(data_dir, 'test_batch')
    test_batch = load_cifar_batch(test_batch_file)
    return test_batch[b'data'], np.array(test_batch[b'labels'])

data_dir = '/home/chance/cifarDataStuff/cifar-10-python/cifar-10-batches-py/'  # Replace with the path to the extracted CIFAR-10 dataset directory
x_train, y_train = load_cifar10_train_data(data_dir)
x_test, y_test = load_cifar10_test_data(data_dir)




#Chance's mislabeling implementation
print("input the percent of the training data set you want to mislabel, 0.xx: ")
percent_to_mislabel = float(input())
print("input file name yo uwant model saved to, xxxx.h5: ")
fileSaveName = str(input())

y_train_len = y_train.shape[0]
totalNumToMislabel = math.floor(y_train_len * percent_to_mislabel)

#indices_to_change = np.random.choice(y_train_len, totalNumToMislabel, replace=False)

#for idx in indices_to_change:
    # Get the current value at the index
    #current_value = y_train[idx]
    
    #new_value = choice([i for i in range(0,9) if i not in [current_value]])
    
    # Update the array with the new value
    #y_train[idx] = new_value
#End of Chance's mislabeling



########Ignore all of this garbage code we aren't using the actual values just the shap of the arrays
print("NUMBER 1________________")
# Resize CIFAR-10 images to meet the minimum input size requirement of InceptionV3 (75x75 pixels) and convert to RGB
x_train_resized = np.array([resize(image, (75, 75)) for image in x_train])
x_train_resized = np.stack((x_train_resized,) * 3, axis=-1)
x_test_resized = np.array([resize(image, (75, 75)) for image in x_test])
x_test_resized = np.stack((x_test_resized,) * 3, axis=-1)
########Ok the code is probably less garbage now

print(x_train_resized.shape)
print(x_test_resized.shape)

#Tucker's resizing implementation
for i, image_array in enumerate(x_train):

    image_tensor = tf.reshape(image_array, (32, 32, 3))
    image_tensor = tf.cast(image_tensor, tf.float32)
    resized_image = tf.image.resize(image_tensor, (75, 75), method=tf.image.ResizeMethod.BILINEAR)
    x_train_resized[i] = resized_image				#<----- This is dumb but I don't know how to initialize an array properly
    
for i, image_array in enumerate(x_test):

    image_tensor = tf.reshape(image_array, (32, 32, 3))
    image_tensor = tf.cast(image_tensor, tf.float32)
    resized_image = tf.image.resize(image_tensor, (75, 75), method=tf.image.ResizeMethod.BILINEAR)
    x_test_resized[i] = resized_image				#<----- This is dumb but I don't know how to initialize an array properly
#End of Tucker's code


print("NUMBER 2________________")
# Normalize pixel values to between 0 and 1
#x_train_resized = x_train_resized.astype('float32') / 255.0
#x_test_resized = x_test_resized.astype('float32') / 255.0

# Normalize pixel values to between -1 and 1
x_train_resized = (x_train_resized.astype('float32') / 255.0) * 2 - 1.0
x_test_resized = (x_test_resized.astype('float32') / 255.0) *2 - 1.0

# Normalize pixel values to between -1 and 1
#x_train_resized = tf.keras.applications.inception_v3.preprocess_input(x_train_resized)
#x_test_resized = tf.keras.applications.inception_v3.preprocess_input(x_test_resized)

print("NUMBER 3________________")
# Convert labels to one-hot encoding using to_categorical from tensorflow.keras.utils
num_classes = 10
#y_train = to_categorical(y_train, num_classes)   <----skipping for now because it interferes with the scrabling code
y_test = to_categorical(y_test, num_classes)

print("NUMBER 4________________")
# Define InceptionV3 model
base_model = InceptionV3(weights= 'imagenet', include_top=False, input_shape=(75, 75, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("NUMBER 5________________")
# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False
print("NUMBER 6________________")
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Tucker and Chance's fancy shmancy epochlabelscramblinator
print("NUMBER 7________________")
actualEpochs = 10;
for i in range(actualEpochs):
    y_train_scrambled = y_train[:]
    
    indices_to_change = np.random.choice(y_train_len, totalNumToMislabel, replace=False)
    for idx in indices_to_change:
 
    	# Get a new value that is different from the original value
    	current_value = y_train[idx]
    	new_value = random.choice(y_train)
    	while (new_value == current_value):
    	    new_value = random.choice(y_train)
    
    	# Update the scramled array with the new value
    	y_train_scrambled[idx] = new_value
    
    #Repeating step 3 for scrambled data
    y_train_scrambled = to_categorical(y_train_scrambled, num_classes)
    
    model.fit(x_train_resized, y_train_scrambled, batch_size=32, epochs=1, validation_data=(x_test_resized, y_test))
# End of the fancy shmancy epochlabelscramlinator


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_resized, y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save(fileSaveName)
print("Model saved as " + fileSaveName)
