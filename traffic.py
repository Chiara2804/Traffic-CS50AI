import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


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
    #initiate lists for images and labels
    images = []
    labels = []

    #define path to file folder
    path_to_folder = os.path.join(".", f"{data_dir}")

    #save list of labels
    categories = [f for f in os.listdir(path_to_folder)]

    #iterate over different categories
    for category in categories:
        #get category directory path
        path_to_category = os.path.join(path_to_folder, f"{category}")
        #get all files in folder
        dirs = os.listdir(path_to_category)

        #iterate over files in category, convert them to numpy-array
        for file in dirs:
            #print progress
            print(f'Loading category {category}: {file}', end="\r", flush=True)

            #get file path
            path_to_file = os.path.join(path_to_folder, f"{category}", f"{file}")

            #read image and resize
            img = cv2.imread(path_to_file)
            img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))

            #check image snape and return ERROR if not (30, 30, 3)
            if img.shape != (30, 30, 3):
                print(f'Category {category}, File {file}: size ERROR!')

            #append category-file data to images and labels
            labels.append(category)
            images.append(img)

        #print loaded message
        print(f'Category {category} loaded!' + ' ' * 20)

    #print number of files that were loaded or error message
    if len(images) == len(labels):
        print(f'\nSuccessfully loaded {len(labels)} pictures from {len(categories)} categories.')
    else:
        print('ERROR! "Images" and "labels"-list are of different lenght!')
    
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    #initiate sequential model
    model = tf.keras.models.Sequential()

    #convolutional - pooling layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(64,(3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128,(3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    #flatten
    model.add(tf.keras.layers.Flatten())

    #add dropout
    model.add(tf.keras.layers.Dropout(0.4))

    #output layer with NUM_CATEGORIES outputs
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    #compile model with default optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return(model)


if __name__ == "__main__":
    main()
