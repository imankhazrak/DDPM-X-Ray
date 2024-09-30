import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# import tensorflow_addons as tfa  # To access the Swin Transformer


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
def vgg_model(input_shape, weights):
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def resnet_model(input_shape, weights):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


from transformers import SwinModel, SwinConfig
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def swin_transformer_model(input_shape=(224, 224, 3), num_classes=2):
    # Load the pretrained Swin Transformer model
    swin_config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224")
    swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", config=swin_config)
    
    # Define input layer
    inputs = Input(shape=input_shape)
    
    # Use the Swin Transformer as a feature extractor
    swin_features = swin_model(inputs)
    
    # Flatten the output from the Swin Transformer model
    x = Flatten()(swin_features.last_hidden_state)
    
    # Add a fully connected layer with 512 units and ReLU activation
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Add the final output layer with softmax activation for classification
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model



# def swin_transformer_model(input_shape=(128, 128, 3)):
#     # Load Swin Transformer from tensorflow_addons
#     base_model = tfa.layers.SwinTransformer2D(input_shape=input_shape, patch_size=4, window_size=7, num_heads=4, embed_dim=96, depths=(2, 2, 6, 2))
    
#     base_model.trainable = False  # Freeze the base model layers
    
#     model = Sequential([
#         base_model,
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(2, activation='softmax')  # 2 output classes
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])cv_train_model
    
#     return model

################### Function to call model and run cv cross validation #####################
def cv_train_model(model_fn, X, y, cv, epochs, batch_size):
    # Initialize K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
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
        model = model_fn(input_shape=(128, 128, 3)) #model = model_fn(input_shape=(128, 128, 3))

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

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




def cv_train_and_evaluate_model(dataset_dir, test_dataset_dir, class_labels, model, weights, input_shape, title, file_name, cv, epochs, batch_size):
    print('Class labels: ', class_labels)

    # Prepare dataset with resized images
    X, y = prepare_dataset(dataset_dir, class_labels, target_size=input_shape[:2])
    
    # Train the model
#     model_history, trained_model = cv_train_model(model, X, y, cv)
    model_history, trained_model = cv_train_model_v2(model, X, y, cv, epochs, batch_size, input_shape, weights)
    
    # Plot training history
    plot_train_history(model_history, title, file_name, cv)
    
    
    # Test on data
    test_metrics, confusion_matrix = test_on_data(test_dataset_dir, trained_model)
    
    # Print test metrics
    print(test_metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, class_labels, f'{title} on Test Data')

    return trained_model, test_metrics, confusion_matrix


def imbalanced_cv_train_and_evaluate_model(dataset_dir, test_dataset_dir1, test_dataset_dir2, test_dataset_dir3, class_labels, model, weights, input_shape, title, file_name, cv, epochs, batch_size):
    print('Class labels:', class_labels)

    # Prepare dataset with resized images
    X, y = prepare_dataset(dataset_dir, class_labels, target_size=input_shape[:2])
    
    # Train the model
#     model_history, trained_model = cv_train_model(model, X, y, cv, epochs, batch_size)
    model_history, trained_model = cv_train_model_v2(model, X, y, cv, epochs, batch_size, input_shape, weights)
#     model_history, trained_model = cv_train_model_v3(model, X, y, cv, epochs, batch_size, input_shape, weights) # for resnet


    # Plot training history
    plot_train_history(model_history, title, file_name, cv)
    
    # Test on first test dataset
    print("Testing on Dataset 1")
    test_metrics1, confusion_matrix1 = test_on_data(test_dataset_dir1, trained_model)
    print("Test Metrics for Dataset 1:", test_metrics1)
    plot_confusion_matrix(confusion_matrix1, class_labels, f'{title} on Test Dataset 1')
    
    # Test on second test dataset
    print("Testing on Dataset 2")
    test_metrics2, confusion_matrix2 = test_on_data(test_dataset_dir2, trained_model)
    print("Test Metrics for Dataset 2:", test_metrics2)
    plot_confusion_matrix(confusion_matrix2, class_labels, f'{title} on Test Dataset 2')
    
    # Test on third test dataset
    print("Testing on Dataset 3")
    test_metrics3, confusion_matrix3 = test_on_data(test_dataset_dir3, trained_model)
    print("Test Metrics for Dataset 3:", test_metrics3)
    plot_confusion_matrix(confusion_matrix3, class_labels, f'{title} on Test Dataset 3')
    
    return trained_model, (test_metrics1, test_metrics2, test_metrics3), (confusion_matrix1, confusion_matrix2, confusion_matrix3)

#############################

def cv_train_vgg_model(X, y, cv):
    # Initialize K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
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
# def plot_train_history(fold_metrics_df, title, file_name, cv):
#     # Initialize figure and axes for subplots
#     fig, axes = plt.subplots(2, cv, figsize=(20, 10))  # 2 rows, 5 columns of subplots
    
#     y_ticks = np.arange(0.5, 1.1, 0.1) ####### Added

#     # Plot training loss vs validation loss for each fold (1st row)
#     for i in range(cv):  # Iterate over each fold
#         axes[0, i].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
#         axes[0, i].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
#         axes[0, i].set_title(f"Fold {i+1}: Train vs Val Loss")
#         axes[0, i].set_xlabel('Epoch')
#         axes[0, i].set_ylabel('Loss')
# #         axes[0, i].set_ylim(0, 20) ####### Added
#         axes[0, i].legend()

#     # Plot training accuracy vs validation accuracy for each fold (2nd row)
#     for i in range(cv):  # Iterate over each fold
#         axes[1, i].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
#         axes[1, i].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
#         axes[1, i].set_title(f"Fold {i+1}: Train vs Val Accuracy")
#         axes[1, i].set_xlabel('Epoch')
#         axes[1, i].set_ylabel('Accuracy')
#         axes[1, i].set_ylim(0.5, 1.1) ####### Added
#         axes[1, i].set_yticks(y_ticks)
#         axes[1, i].legend()

#     # Set general title for the entire figure
#     fig.suptitle(title, fontsize=16)

#     # Adjust layout and display the plots
#     plt.tight_layout()
#     plt.savefig(file_name)
#     plt.show()
    
# def plot_train_history(fold_metrics_df, title, file_name, cv):
#     if cv == 1:
#         fig, axes = plt.subplots(1, 2, figsize=(6, 6))  # 2 rows, 1 column of subplots for cv=1
#         axes = np.array(axes).reshape(2, 1)  # Ensure axes are always 2D for consistent indexing
#     else:
#         fig, axes = plt.subplots(cv, 2, figsize=(20, 10))  # 2 rows, cv columns of subplots

#     y_ticks = np.arange(0.5, 1.1, 0.1)

#     # Plot training loss vs validation loss for each fold (1st row)
#     for i in range(cv):  # Iterate over each fold
#         axes[i, 0].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
#         axes[i, 0].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
# #         axes[0, i].set_title(f"Fold {i+1}: Train vs Val Loss")
#         axes[i, 0].set_xlabel('Epoch')
#         axes[i, 0].set_ylabel('Loss')
#         axes[i, 0].legend()

#     # Plot training accuracy vs validation accuracy for each fold (2nd row)
#     for i in range(cv):  # Iterate over each fold
#         axes[i, 1].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
#         axes[i, 1].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
# #         axes[1, i].set_title(f"Fold {i+1}: Train vs Val Accuracy")
#         axes[i, 1].set_xlabel('Epoch')
#         axes[i, 1].set_ylabel('Accuracy')
#         axes[i, 1].set_ylim(0.5, 1.1)
#         axes[i, 1].set_yticks(y_ticks)
#         axes[i, 1].legend()

#     # Set general title for the entire figure
#     fig.suptitle(title, fontsize=16)

#     # Adjust layout and display the plots
#     plt.tight_layout()
#     plt.savefig(file_name)
#     plt.show()
    
def plot_train_history(fold_metrics_df, title, file_name, cv):
    if cv == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns of subplots for cv=1
        axes = np.array(axes).reshape(1, 2)  # Ensure axes are always 2D for consistent indexing
    else:
        fig, axes = plt.subplots(cv, 2, figsize=(20, 5 * cv))  # cv rows, 2 columns of subplots

    y_ticks = np.arange(0.5, 1.1, 0.1)

    # Plot training loss vs validation loss and training accuracy vs validation accuracy for each fold
    for i in range(cv):  # Iterate over each fold
        axes[i, 0].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
        axes[i, 0].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()

        axes[i, 1].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
        axes[i, 1].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Accuracy')
        axes[i, 1].set_ylim(0.5, 1.1)
        axes[i, 1].set_yticks(y_ticks)
        axes[i, 1].legend()

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
def test_on_data(dataset_dir, model):
    class_labels = ['NORMAL', 'PNEUMONIA']
    X_test, y_test = prepare_dataset(dataset_dir, class_labels, target_size=(128, 128))
    metrics_dict, cf_matrix = evaluate_test_set(model, X_test, y_test)
    return metrics_dict, cf_matrix
############################
def plot_confusion_matrix(conf_matrix, class_names, title):
    """
    Plot the confusion matrix as a heatmap.

    Args:
    - conf_matrix (numpy.ndarray): Confusion matrix array.
    - class_names (list): List of class names (labels).
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    
def plot_train_history2(fold_metrics_df, title, file_name):
    # Initialize figure and axes for subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 2 rows, 5 columns of subplots

    # Plot training loss vs validation loss for each fold (1st row)
#     for i in range(5):  # Iterate over each fold
#         axes[0, i].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
#         axes[0, i].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
#         axes[0, i].set_title(f"Fold {i+1}: Train vs Val Loss")
#         axes[0, i].set_xlabel('Epoch')
#         axes[0, i].set_ylabel('Loss')
#         axes[0, i].legend()

    # Plot training accuracy vs validation accuracy for each fold (2nd row)
    for i in range(5):  # Iterate over each fold
        axes[i].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
        axes[i].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
        axes[i].set_title(f"Fold {i+1}: Train vs Val Accuracy")
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()

    # Set general title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    
################### Train model without cross validation ########
# def train_model(model, X, y, validation_split=0.2, epochs=10, batch_size=32, title=None, file_name=None):
#     # Split the dataset into training and validation sets
#     split_index = int(len(X) * (1 - validation_split))
#     X_train, X_val = X[:split_index], X[split_index:]
#     y_train, y_val = y[:split_index], y[split_index:]

#     # Train the model
#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

#     # Evaluate the model on validation data
#     val_loss, val_accuracy = model.evaluate(X_val, y_val)
#     print(f"Validation Accuracy: {val_accuracy}")

#     # Round the values before storing them
#     rounded_train_loss = [round(loss, 2) for loss in history.history['loss']]
#     rounded_train_accuracy = [round(acc, 2) for acc in history.history['accuracy']]
#     rounded_val_loss = [round(loss, 2) for loss in history.history['val_loss']]
#     rounded_val_accuracy = [round(acc, 2) for acc in history.history['val_accuracy']]
#     rounded_val_accuracy_mean = round(val_accuracy, 2)

#     # Store metrics in a dictionary
#     metrics = {
#         'Train Loss': rounded_train_loss,
#         'Train Accuracy': rounded_train_accuracy,
#         'Val Loss': rounded_val_loss,
#         'Val Accuracy': rounded_val_accuracy,
#         'Val Accuracy Mean': rounded_val_accuracy_mean
#     }
    
#     # Plot training history
#     if title and file_name:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns of subplots

#         # Plot training loss vs validation loss
#         axes[0].plot(metrics['Train Loss'], label='Train Loss')
#         axes[0].plot(metrics['Val Loss'], label='Val Loss')
#         axes[0].set_title("Train vs Val Loss")
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('Loss')
#         axes[0].legend()

#         # Plot training accuracy vs validation accuracy
#         axes[1].plot(metrics['Train Accuracy'], label='Train Accuracy')
#         axes[1].plot(metrics['Val Accuracy'], label='Val Accuracy')
#         axes[1].set_title("Train vs Val Accuracy")
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('Accuracy')
#         axes[1].set_ylim(0.3, 1.1) ####### Added
#         axes[1].legend()

#         # Set general title for the entire figure
#         fig.suptitle(title, fontsize=16)

#         # Adjust layout and display the plots
#         plt.tight_layout()
#         plt.savefig(file_name)
#         plt.show()
    
#     return metrics, model
def cv_train_model_v2(model_fn, X, y, cv, epochs, batch_size, input_shape, weights):
    # Create and compile the model with the given input shape
    model = model_fn(input_shape, weights) #model = model_fn(input_shape=(128, 128, 3))

    # Split the data into training and validation sets manually (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy}")

    # Round the values before storing in metrics
    rounded_train_loss = [round(loss, 2) for loss in history.history['loss']]
    rounded_train_accuracy = [round(acc, 2) for acc in history.history['accuracy']]
    rounded_val_loss = [round(loss, 2) for loss in history.history['val_loss']]
    rounded_val_accuracy = [round(acc, 2) for acc in history.history['val_accuracy']]
    rounded_val_accuracy_mean = round(val_accuracy, 2)

    # Store the metrics in a dictionary
    metrics = {
        'Train Loss': rounded_train_loss,
        'Train Accuracy': rounded_train_accuracy,
        'Val Loss': rounded_val_loss,
        'Val Accuracy': rounded_val_accuracy,
        'Val Accuracy Mean': rounded_val_accuracy_mean
    }

    metrics_df = pd.DataFrame([metrics])

    return metrics_df, model

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def cv_train_model_v3(model_fn, X, y, cv, epochs, batch_size, input_shape, weights):
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    fold_metrics = []
    best_model = None
    best_val_accuracy = 0.0

    fold = 0
    for train_index, val_index in kf.split(X, y):
        fold += 1
        print(f"Training on Fold {fold}/{cv}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create and compile the model
        model = model_fn(input_shape, weights)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        # Train the model with class weights
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_val, y_val), class_weight=class_weights)

        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print(f"Fold {fold} Validation Accuracy: {val_accuracy}")

        # Predict on the validation set and generate the classification report
        y_pred = np.argmax(model.predict(X_val), axis=1)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        print(f"Fold {fold} Classification Report:\n", report)
        print(f"Fold {fold} Confusion Matrix:\n", cm)

        # Store metrics for this fold
        fold_metrics.append({
            'Fold': fold,
            'Train Loss': history.history['loss'],
            'Train Accuracy': history.history['accuracy'],
            'Val Loss': history.history['val_loss'],
            'Val Accuracy': history.history['val_accuracy'],
            'Val Accuracy Mean': val_accuracy
        })

        # Track the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy

    # Convert metrics to DataFrame for analysis
    fold_metrics_df = pd.DataFrame(fold_metrics)

    return fold_metrics_df, best_model


def train_and_evaluate_model(dataset_dir, test_dataset_dir, class_labels, model, input_shape, title, file_name):
    print('Class labels: ', class_labels)

    # Prepare dataset with resized images
    X, y = prepare_dataset(dataset_dir, class_labels, target_size=input_shape[:2])
    
    # Train the model
    model_history, trained_model = train_model(model, X, y, title=title, file_name=file_name)
    
    # Test on data
    test_metrics, confusion_matrix = test_on_data(test_dataset_dir, trained_model)
    
    # Print test metrics
    print(test_metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, class_labels, f'{title} on Test Data')

    return trained_model, test_metrics, confusion_matrix

    
################## DEEP Custom CNN ###################
    
# import tensorflow as tf
from tensorflow.keras.models import Model


def deep_custom_cnn(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial Convolutional Block
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # First Dense Block
    for filters in [64, 64]:
        y = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        y = BatchNormalization()(y)
        x = Concatenate()([x, y])

    # Transition Layer
    x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Dense Block
    for filters in [128, 128]:
        y = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        y = BatchNormalization()(y)
        x = Concatenate()([x, y])

    # Transition Layer
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third Dense Block
    for filters in [256, 256]:
        y = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        y = BatchNormalization()(y)
        x = Concatenate()([x, y])

    # Bridge Layer
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    if num_classes == 2:
        output = Dense(1, activation='sigmoid')(x)
    else:
        output = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


#######################ResNet Models #########################
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

def build_resnet_model(model_type='ResNet50', input_shape=(224, 224, 3), num_classes=2):
    if model_type == 'ResNet50':
        base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == 'ResNet101':
        base_model = ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == 'ResNet152':
        base_model = ResNet152(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

####################### DenseNet Models############################
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

def build_densenet_model(model_type='DenseNet121', input_shape=(224, 224, 3), num_classes=2):
    if model_type == 'DenseNet121':
        base_model = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == 'DenseNet169':
        base_model = DenseNet169(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == 'DenseNet201':
        base_model = DenseNet201(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

###################### Inception Models ##############################
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

def build_inception_model(model_type='InceptionV3', input_shape=(299, 299, 3), num_classes=2):
    if model_type == 'InceptionV3':
        base_model = InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == 'InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

####################### EfficientNet Models ##########################
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

def build_efficientnet_model(model_type='EfficientNetB0', input_shape=(224, 224, 3), num_classes=2):
    model_class = globals()[model_type]
    base_model = model_class(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

######################## Xception Model ##############################
from tensorflow.keras.applications import Xception

def build_xception_model(input_shape=(299, 299, 3), num_classes=2):
    base_model = Xception(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

######################################################

def cv_train_deep_cnn_model(X, y):
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
        model = deep_custom_cnn((128, 128, 3), 2)

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

######################################################

def cv_train_deep_cnn_model(X, y):
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
        model = deep_custom_cnn((128, 128, 3), 2)

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


######################### Fine tuning ##################
from tensorflow.keras.models import load_model
def fine_tune_model_on_dataset_B(X_B, y_B, best_model_path='best_vgg_model_on_dataset_A.h5'):
    # Load the best model trained on Dataset A
    model = load_model(best_model_path)

    # Fine-tune on Dataset B
    history = model.fit(X_B, y_B, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the validation set of Dataset B
    val_loss, val_accuracy = model.evaluate(X_B, y_B)
    print(f"Fine-tuned Model Validation Accuracy on Dataset B: {val_accuracy}")

    # Return the fine-tuned model and history for further analysis if needed
    return model, history

# # Example usage:
# # Train and get the best model from Dataset A
# fold_metrics_A, best_model_A = cv_train_vgg_model(X_A, y_A)

# # Fine-tune the best model on Dataset B
# fine_tuned_model_B, history_B = fine_tune_model_on_dataset_B(X_B, y_B)


###################################################


###########################################

def custom_cnn(input_shape=(256, 256, 3), num_classes=2 ):
    model = Sequential([
    Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')  # Changed to 2 for the two classes
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
