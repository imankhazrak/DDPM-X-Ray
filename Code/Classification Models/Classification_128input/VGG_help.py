import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score


def prepare_dataset(dataset_dir, class_labels, target_size=(128, 128)):
    X = []
    y = []

    for label_index, class_label in enumerate(class_labels):
        label_folder = os.path.join(dataset_dir, class_label)
        image_files = os.listdir(label_folder)

        for img_file in image_files:
            img_path = os.path.join(label_folder, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.vgg16.preprocess_input(img)
            X.append(img)
            y.append(label_index)

    X = np.array(X)
    y = np.array(y)

    return X, y

#################################
def vgg_model(input_shape=(128, 128, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes
    ])
    
    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

#############################

def cv_train_vgg_model(X, y):
    # Initialize K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # Initialize lists to store training history across folds
    fold_metrics = []
    best_model = None
    best_val_accuracy = 0.0  # Track the best validation accuracy

    fold = 0
    for train_index, val_index in kf.split(X, y):
        fold += 1
        print(f"Fold {fold}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create and compile VGG model with input shape (128, 128, 3)
        model = vgg_model(input_shape=(128, 128, 3))

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {val_accuracy}")

        # Round the values before storing in fold_metrics
        rounded_train_loss = [round(loss, 2) for loss in history.history['loss']]
        rounded_train_accuracy = [round(acc, 2) for acc in history.history['accuracy']]
        rounded_val_loss = [round(loss, 2) for loss in history.history['val_loss']]
        rounded_val_accuracy = [round(acc, 2) for acc in history.history['val_accuracy']]
        rounded_val_accuracy_mean = round(val_accuracy, 2)

        # Append rounded metrics to fold_metrics
        fold_metrics.append({
            'Fold': fold,
            'Train Loss': rounded_train_loss,
            'Train Accuracy': rounded_train_accuracy,
            'Val Loss': rounded_val_loss,
            'Val Accuracy': rounded_val_accuracy,
            'Val Accuracy Mean': rounded_val_accuracy_mean
        })

        # Update best model if current model is better
        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy
            
    fold_metrics_df = pd.DataFrame(fold_metrics)
    
    return fold_metrics_df, best_model


    
    ######################################
def plot_train_history(fold_metrics_df, title, file_name):
    # Initialize figure and axes for subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns of subplots

    # Plot training loss vs validation loss for each fold (1st row)
    for i in range(5):  # Iterate over each fold
        axes[0, i].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
        axes[0, i].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
        axes[0, i].set_title(f"Fold {i+1}: Train vs Val Loss")
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()

    # Plot training accuracy vs validation accuracy for each fold (2nd row)
    for i in range(5):  # Iterate over each fold
        axes[1, i].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
        axes[1, i].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
        axes[1, i].set_title(f"Fold {i+1}: Train vs Val Accuracy")
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].legend()

    # Set general title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    
    ######################################

#######################################################
def evaluate_test_set(model, X_test, y_test):
    """
    Evaluate the trained model on a test set and return performance metrics as a dictionary with two-digit precision.

    Args:
    - model: Trained VGG model.
    - X_test (numpy.ndarray): Test set features.
    - y_test (numpy.ndarray): Test set labels.
    
    Returns:
    - metrics_dict (dict): Dictionary containing performance metrics.
    """
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    test_loss = round(test_loss, 4)
    test_accuracy = round(test_accuracy, 4)

    # Get model predictions on test data
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Calculate additional performance metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    sensitivity = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Round metrics to two-digit precision
    f1 = round(f1, 2)
    sensitivity = round(sensitivity, 2)
    precision = round(precision, 2)

    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Create dictionary to store metrics
    metrics_dict = {
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'F1 Score': f1,
        'Sensitivity (Recall)': sensitivity,
        'Precision': precision,      
    }

    return metrics_dict, conf_matrix
############################
def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix as a heatmap.

    Args:
    - conf_matrix (numpy.ndarray): Confusion matrix array.
    - class_names (list): List of class names (labels).
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()