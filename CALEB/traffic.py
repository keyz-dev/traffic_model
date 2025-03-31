import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

EPOCHS = 25
IMG_WIDTH= 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, random_state=42
    )

    # Get a compiled neural network
    model = get_model()
    
    # use Early Stopping to stop if no improvement in training
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping])

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    
    try:
    # Loop through category directories
        for category in range(NUM_CATEGORIES):
            category_path = os.path.join(data_dir, str(category))
            
            if not os.path.exists(category_path):
                continue

            # Loop through each image in the category directory
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                # Read image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    continue
                
                # Resize image
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                image = image / 255.0
                # Append images and labels to the respective list
                images.append(image)
                labels.append(category)
                
        return np.array(images), np.array(labels)
    
    except:
        raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    try:
        # Create a sequential model
        model = tf.keras.models.Sequential()

        # Add a convolutional layer with 32 filters, a 3x3 kernel, ReLU activation, and input shape
        model.add(tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", kernel_regularizer=l2(0.001), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ))

        # Add a max-pooling layer with a 2x2 pool size
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer to reduce overfitting
        model.add(tf.keras.layers.Dropout(0.3))
        
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.001)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer to reduce overfitting
        model.add(tf.keras.layers.Dropout(0.3))
        
        model.add(tf.keras.layers.Flatten())
        
        # Add a hidden layer with 128 units and ReLU activation
        model.add(tf.keras.layers.Dense(128, activation="relu"))

        # Add a dropout layer to reduce overfitting
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
    except:    
        raise NotImplementedError

if __name__ == "__main__":
    main()
