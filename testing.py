from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the pre-trained InceptionV3 model trained on CIFAR-10
model = load_model('inceptionv3_cifar10_model.h5')

# Define function to preprocess input image
def preprocess_input(img_path):
    # Resize the image to the expected input shape (75, 75)
    img = image.load_img(img_path, target_size=(75, 75))
    img_array = image.img_to_array(img)
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values to [0, 1]
    img_array /= 255.0
    return img_array

# Load an example image
img_path = "dog.jpg"  # Change this to the path of your image
img = preprocess_input(img_path)

# Perform inference
predictions = model.predict(img)

# Read the CIFAR-10 labels
cifar10_labels = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# Print top predicted category
# Get the indices of the top 5 predicted classes
top5_indices = np.argsort(predictions)[0][-5:][::-1]

# Print the top 5 predicted classes and their probabilities
print("Top 5 predicted classes:")
for i in top5_indices:
    predicted_label = cifar10_labels[i]
    probability = predictions[0][i]
    print(predicted_label, ":", probability)
