import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.utils import to_categorical
import pickle
from skimage.transform import resize  # Import resize function from skimage.transform


# Chance's resizing function
def convert_to_image(rgb_data):
    """
    Converts a (3072,) ndarray containing RGB data into a (32, 32, 3) ndarray representing a 32 x 32 image.

    Parameters:
    rgb_data (ndarray): A (3072,) ndarray containing RGB data.

    Returns:
    ndarray: A (32, 32, 3) ndarray representing a 32 x 32 image.
    """
    # Ensure the input array has the correct shape
    if rgb_data.shape != (3072,):
        raise ValueError("Input array must have shape (3072,)")

    # Split the input array into three equal parts for red, green, and blue data
    red_data = rgb_data[:1024]
    green_data = rgb_data[1024:2048]
    blue_data = rgb_data[2048:]

    # Reshape each color data into (32, 32)
    red_data = red_data.reshape((32, 32))
    green_data = green_data.reshape((32, 32))
    blue_data = blue_data.reshape((32, 32))

    # Stack the red, green, and blue data along the third axis
    image = np.stack([red_data, green_data, blue_data], axis=2)

    return image


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


########Ignore all of this garbage code
print("NUMBER 1________________")
# Resize CIFAR-10 images to meet the minimum input size requirement of InceptionV3 (75x75 pixels) and convert to RGB
x_train_resized = np.array([resize(image, (75, 75)) for image in x_train])
x_train_resized = np.stack((x_train_resized,) * 3, axis=-1)  # Convert grayscale to RGB
x_test_resized = np.array([resize(image, (75, 75)) for image in x_test])
x_test_resized = np.stack((x_test_resized,) * 3, axis=-1)  # Convert grayscale to RGB
########Ok the code is probably less garbage now


#Tucker's resizing implementation
print("STUFF")
print(x_train[1].shape)
print(type(x_train[1]))

for i, image_array in enumerate(x_train):

    image_tensor = tf.reshape(image_array, (32, 32, 3))
    resized_image = tf.image.resize(image_tensor, (75, 75), method=tf.image.ResizeMethod.BILINEAR)
    x_train_resized[i] = resized_image				#<----- This is dumb but I don't know how to initialize an array properly
    
for i, image_array in enumerate(x_test):

    image_tensor = tf.reshape(image_array, (32, 32, 3))
    resized_image = tf.image.resize(image_tensor, (75, 75), method=tf.image.ResizeMethod.BILINEAR)
    x_test_resized[i] = resized_image				#<----- This is dumb but I don't know how to initialize an array properly
#End of Tucker's code


print("NUMBER 2________________")
# Normalize pixel values to between 0 and 1
x_train_resized = x_train_resized.astype('float32') / 255.0
x_test_resized = x_test_resized.astype('float32') / 255.0

print("NUMBER 3________________")
# Convert labels to one-hot encoding using to_categorical from tensorflow.keras.utils
num_classes = 10
y_train = to_categorical(y_train, num_classes)
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
print("NUMBER 7________________")
# Train the model
model.fit(x_train_resized, y_train, batch_size=32, epochs=100, validation_data=(x_test_resized, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_resized, y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save("funk.h5")
print("Model saved aszackSaidNone.h5")
