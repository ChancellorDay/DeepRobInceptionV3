
# Inceptionv3 Extension for DeepRob
This repsoitory was created for the final project of Robotics 498 - DeepRob. It is designed to emulate the InceptionV3 architecture as well as an algorithmic extension designed to increase noise to prevent overfitting. 


## Authors

- [Chancellor Day](dchance@umich.edu)
- [Zack Vega](zvega@umich.edu)
- [Tucker Moffat](moffatuc@umich.edu)
- [Meha Goya](mehag@umich.edu)






## Requirements
All code was run on Ubuntu 24.04 and Python 3.13.
- Numpy
- Tensorflow
- Keras
- skimage
- pickle

## Deployment


# Important
Remember to replace this line with your saved CIFAR-10 dataset. 
```
data_dir = '/home/chance/cifarDataStuff/cifar-10-python/cifar-10-batches-py/' 
```

To emulate model 1 extension run
```bash
    python trainingExtension.py
```

To emulate our extended architecture with 
```bash
    python TrainingExtensionExtension.py
```

To emulate the base inceptionv3 architecture 
```bash
    python trainingExtension.py
```
or
```bash
    python TrainingExtensionExtension.py
```
and when it asks you for what percent of the data you want to mislabel, enter 0.0

If using the model 1 or model 2 architecture it will be your choice for the name when you run it.

## Testing
To test the trained architecture, in test.py locate line 6
```python
model = load_model('inceptionv3_cifar10_model.h5')
```

replace the model name with the file you have created and the architecture would would like to use. 

To test an image replace line 20
```python
model = load_model('img_path = "dog.jpg")
```
with the image of your choosing that fits with the CIFAR 10 dataset.
From there, you must run the file and it will output the top 5 scores.  


