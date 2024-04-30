# auto_encoder_colorizationV2

 Train auto-encoder model to colorize grayscale images.
This code defines a convolutional neural network (CNN) using Keras for image processing tasks. Here's a breakdown of the code:

1. Import necessary libraries:
   - `keras.layers`: Contains various layers used to build the model.
   - `keras.models`: Contains the Model class for creating a neural network model.

2. Define two functions for building the encoder and decoder parts of the network:
   - `down`: Builds the downsampling layers (encoder) using Conv2D.
   - `up`: Builds the upsampling layers (decoder) using Conv2DTranspose.

3. Define the main function `create_model()` to create the entire model:
   - Define the input layer with shape (128, 128, 1).
   - Encoding layers (downsampling) using the `down` function.
   - Decoding layers (upsampling) using the `up` function.
   - Concatenate the feature maps from the corresponding encoder layers with the decoder layers.
   - The final output layer is a Conv2D layer with 3 filters (assuming RGB images) and a kernel size of (2, 2).

# Training

1. Import necessary libraries:
   - `from up_down_functions import create_model`: Import the `create_model` function from a module named `up_down_functions`, which likely contains the model architecture definition.
   - `from keras.callbacks import ModelCheckpoint`: Import the `ModelCheckpoint` callback from Keras, which will save the model weights during training.
   - `import numpy as np`: Import NumPy for numerical operations.

2. Load data:
   - Load grayscale images (`x_gray`) and corresponding color images (`x_color`) from specified file paths.
   - Print the shapes of the loaded data arrays.

3. Split the data into training and testing sets:
   - Split the grayscale and color images into training and testing sets. It seems that the first 5500 samples are used for training, and the rest are used for testing.

4. Create the model:
   - Call the `create_model` function to instantiate the neural network model.

5. Compile the model:
   - Compile the model using mean absolute error (MAE) as the loss function, accuracy as the metric, and the Adam optimizer.

6. Set up a ModelCheckpoint callback:
   - Configure a ModelCheckpoint callback to save the model's weights during training. The `filepath` parameter specifies the directory and prefix for the saved model files, and the `monitor` parameter specifies the metric to monitor for saving the best model.

7. Train the model:
   - Fit the model to the training data (`x_train_gray`, `x_train_color`) using the validation data (`x_test_gray`, `x_test_color`) for validation.
   - Train for 50 epochs with a batch size of 16.
   - Pass the ModelCheckpoint callback to the `callbacks` parameter to save the model weights.

Overall, this code trains a neural network model to colorize grayscale images using a dataset of paired grayscale and color images. It uses the MAE loss function, accuracy as a metric, and the Adam optimizer for training. During training, it saves the best model weights based on the validation accuracy.
