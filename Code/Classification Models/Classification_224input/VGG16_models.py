
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNet

np.random.seed(123)
tf.random.set_seed(123)

def preprocess_grayscale_to_rgb(x):
    # Replicate the grayscale values across the last dimension to mimic RGB
    return tf.tile(x, [1, 1, 1, 3])

def data_generator(image_paths, labels, batch_size=32, seed=123):
    rng = np.random.default_rng(seed)  # Using the new numpy Generator for random number generation
    while True:
        indices = np.arange(len(image_paths))
        rng.shuffle(indices)  # Shuffling with a specific seed

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            images_batch = np.zeros((len(batch_indices), 224, 224, 1))
            labels_batch = labels[batch_indices]

            for i, idx in enumerate(batch_indices):
                img = load_img(image_paths[idx], color_mode='grayscale', target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                images_batch[i] = img_array

            yield images_batch, labels_batch

def preprocess_grayscale_to_rgb2(x):
    # Ensure that x is a tensor with shape [height, width, 1]
    # Replicate the grayscale values across the third dimension to mimic RGB
    return tf.tile(x, [1, 1, 3])

def data_generator2(image_paths, labels, batch_size=32, seed=123):
    rng = np.random.default_rng(seed)
    while True:
        indices = np.arange(len(image_paths))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            # Initialize the batch array for RGB images
            images_batch = np.zeros((len(batch_indices), 224, 224, 3))
            labels_batch = labels[batch_indices]

            for i, idx in enumerate(batch_indices):
                img = load_img(image_paths[idx], color_mode='grayscale', target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                # Reshape the image to ensure it is [224, 224, 1]
                img_array = np.reshape(img_array, (224, 224, 1))
                # Convert grayscale to RGB
                img_rgb = preprocess_grayscale_to_rgb2(img_array)
                images_batch[i] = img_rgb

            yield images_batch, labels_batch
            
            
          
            
def prepare_data_for_folds(dataset_dir, class_labels):
    X, y = [], []
    for class_index, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            X.append(os.path.join(class_dir, image_name))
            y.append(class_index)

    return np.array(X), np.array(y)


############# prepare train and val data for train ###
import os
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_train_val_data(dataset_dir, class_labels, seed=123, batch_size = 32):
    """
    Loads images from a directory, assigns labels based on folder names, and splits the data into training and validation sets.

    Parameters:
    - path_dir: The base directory where the dataset is located.

    Returns:
    - train_gen: A data generator for the training set.
    - val_gen: A data generator for the validation set.
    - batch_size: The batch size used for the data generators.
    """
    
    X, y = [], []

    for class_index, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            X.append(os.path.join(class_dir, image_name))
            y.append(class_index)

    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)  # Using the same seed
#     batch_size = 32
    train_gen = data_generator(np.array(X_train), y_train, batch_size, seed)
    val_gen = data_generator(np.array(X_val), y_val, batch_size, seed)

    return train_gen, val_gen, X_train, X_val, batch_size

# Example usage:
# train_gen, val_gen, X_train, X_val, batch_size = prepare_train_val_data(dataset_dir)


############# VGG16 ################

def VGG_model(input_shape, num_classes):
    # Load VGG16 without pretrained weights
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    
    # Define the input layer for grayscale images
    input_layer = Input(shape=(224, 224, 1), name='grayscale_input')
    
    x = Lambda(preprocess_grayscale_to_rgb)(input_layer)  # This line is just a placeholder for your preprocessing    
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


####### Pretrained VGG16 ###########
def VGG_imagenet(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Define the input layer for grayscale images
    input_layer = Input(shape=(224, 224, 1), name='grayscale_input')
    # Apply the revised preprocessing function within a Lambda layer
    x = Lambda(preprocess_grayscale_to_rgb)(input_layer)
    # Use the preprocessed input with the base model
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


####### Pretrained MobileNet ###########

def MobileNet_imagenet(input_shape, num_classes):
    # Load MobileNet with pretrained ImageNet weights
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Set the base model layers to not be trainable
    for layer in base_model.layers:
        layer.trainable = False
    
    # Define the input layer for grayscale images
    input_layer = Input(shape=(224, 224, 1), name='grayscale_input')
    
    # Replicate the grayscale image to have 3 channels
    x = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(input_layer)
    
    # Pass the replicated grayscale images through the base model
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    
    # Add custom layers on top of the base model
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

############### Predict Valedation Set ###
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score

def evaluate_validation_set(val_gen, model, X_val, class_labels, plot_confusion_matrix, title="Confusion Matrix"):
    """
    Evaluates model performance on a validation set, plots a confusion matrix,
    and prints a classification report.

    Parameters:
    - val_gen: A generator that yields batches of images and labels (validation set).
    - model: The trained model for prediction.
    - X_val: List of validation images used to determine when to stop predictions.
    - class_labels: List of class labels corresponding to the dataset.
    - plot_confusion_matrix: Function to plot the confusion matrix.

    Returns:
    - y_true: True labels of the validation set.
    - y_pred: Predicted labels of the validation set.
    """
    y_pred = []
    y_true = []
    for images, labels in val_gen:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels)
        
        if len(y_pred) >= len(X_val):
            break

    y_pred = np.array(y_pred)[:len(X_val)]
    y_true = np.array(y_true)[:len(X_val)]
    

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(cm, classes=class_labels, title=title)
    plt.show()

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_labels))
    # For binary classification
    binary_accuracy = accuracy_score(y_true, y_pred)
    binary_recall = recall_score(y_true, y_pred)
    binary_f1 = f1_score(y_true, y_pred)

    print("Binary Classification:")
    print(f"Accuracy: {binary_accuracy}")
    print(f"Recall: {binary_recall}")
    print(f"F1 Score: {binary_f1}")
    

    return #y_true, y_pred


############### Predict Test Set #########
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_test_set(dataset_dir, model, batch_size, data_generator, plot_confusion_matrix, class_labels=None, title="Confusion Matrix"):
    """
    Loads test images, generates predictions, plots a confusion matrix,
    and prints a classification report.

    Parameters:
    - dataset_dir: Directory containing the test dataset with class subdirectories.
    - model: The trained model for prediction.
    - batch_size: The size of the batches to generate with the data_generator.
    - data_generator: A function that yields batches of data (images and labels).
    - plot_confusion_matrix: A function that plots the confusion matrix.
    - class_labels: Optional. A list of class labels. If None, class labels will be inferred from directory names.

    Returns:
    - y_true_test: The true labels.
    - y_pred_test: The predicted labels.
    """
    X_test, y_test = [], []
    
    
    # Load images and labels
    for class_index, class_name in enumerate(class_labels):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            X_test.append(os.path.join(class_dir, image_name))
            y_test.append(class_index)

    y_test = np.array(y_test)
    test_gen = data_generator(np.array(X_test), y_test, batch_size)

    # Generate predictions
    y_pred_test = []
    y_true_test = []
    for images, labels in test_gen:
        preds = model.predict(images, verbose=0)
        y_pred_test.extend(np.argmax(preds, axis=1))
        y_true_test.extend(labels)

        if len(y_pred_test) >= len(X_test):
            break

    y_pred_test = np.array(y_pred_test)[:len(X_test)]
    y_true_test = np.array(y_true_test)[:len(X_test)]

    # Plot confusion matrix
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plt.figure()
    plot_confusion_matrix(cm_test, classes=class_labels, title=title)
    plt.show()

    # Print classification report
    print(classification_report(y_true_test, y_pred_test, target_names=class_labels))
    # For binary classification
    binary_accuracy = accuracy_score(y_true_test, y_pred_test)
    binary_recall = recall_score(y_true_test, y_pred_test)
    binary_f1 = f1_score(y_true_test, y_pred_test)

    print("Binary Classification:")
    print(f"Accuracy: {binary_accuracy}")
    print(f"Recall: {binary_recall}")
    print(f"F1 Score: {binary_f1}")
    
    return #y_true_test, y_pred_test



