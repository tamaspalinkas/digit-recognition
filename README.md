# Digit recognition from video

Simple implementation of a neural network to predict handwritten digits from video using webcam.

# Description
The model uses a softmax layer to solve the multi-class problem.
Using l2 kernel regularization dropped the accuracy on the training data by ~ 5% but allowed the model to generalize better on the user generated input.

OpenCV was used to detect the contour of the image and make image transformations. Keras normalization utility is then used to feed the data to the model.

# Execution
To run the code use
```python digit_recognizer.py```

# Details

### Data
MNIST handwritten digits dataset [(link)](http://yann.lecun.com/exdb/mnist/)

### Libraries
- tensorflow
- keras
- opencv
- glob

# Demo
<img src="https://github.com/tamaspalinkas/digit-recognition/blob/master/demo.gif">
