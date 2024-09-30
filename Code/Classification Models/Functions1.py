
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import clone_model, Sequential
from tensorflow.keras.applications import ResNet50, ResNet101

from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import itertools

np.random.seed(123)
tf.random.set_seed(123)

def preprocess_grayscale_to_rgb(x):
    # Replicate the grayscale values across the last dimension to mimic RGB
    return tf.tile(x, [1, 1, 1, 3])

def data_generator(image_paths, labels, img_size, batch_size=32, seed=123):
    rng = np.random.default_rng(seed)  # Using the new numpy Generator for random number generation
    while True:
        indices = np.arange(len(image_paths))
        rng.shuffle(indices)  # Shuffling with a specific seed

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            images_batch = np.zeros((len(batch_indices), img_size, img_size, 1))
            labels_batch = labels[batch_indices]

            for i, idx in enumerate(batch_indices):
                img = load_img(image_paths[idx], color_mode='grayscale', target_size=(img_size, img_size))
                img_array = img_to_array(img) / 255.0
                images_batch[i] = img_array

            yield images_batch, labels_batch

def preprocess_grayscale_to_rgb2(x):
    # Ensure that x is a tensor with shape [height, width, 1]
    # Replicate the grayscale values across the third dimension to mimic RGB
    return tf.tile(x, [1, 1, 3])

def data_generator2(image_paths, labels, img_size, batch_size=32, seed=123):
    rng = np.random.default_rng(seed)
    while True:
        indices = np.arange(len(image_paths))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            # Initialize the batch array for RGB images
            images_batch = np.zeros((len(batch_indices), img_size, img_size, 3))
            labels_batch = labels[batch_indices]

            for i, idx in enumerate(batch_indices):
                img = load_img(image_paths[idx], color_mode='grayscale', target_size=(img_size, img_size))
                img_array = img_to_array(img) / 255.0
                # Reshape the image to ensure it is [img_size, img_size, 1]
                img_array = np.reshape(img_array, (img_size, img_size, 1))
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


def split_dataset(ds, train_ratio=0.8, val_ratio=0.2, shuffle=True):
    # Get dataset size
    dataset_size = len(ds)

    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    # Shuffle dataset if required
    if shuffle:
        ds = ds.shuffle(dataset_size)

    # Split dataset
    train_dataset = ds.take(train_size)
    val_dataset = ds.skip(train_size).take(val_size)

    return train_dataset, val_dataset

def load_and_filter_images(directory, class_labels, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {class_name: index for index, class_name in enumerate(class_labels)}

    for class_name in class_labels:
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
            labels.append(label_map[class_name])

    images = np.array(images)
    labels = np.array(labels)
    
    return tf.data.Dataset.from_tensor_slices((images, labels))

def load_and_preprocess_data(dataset_dir, class_labels, img_size=(256, 256), batch_size=32, buffer_size=1024):
    """
    Load, preprocess, and split dataset into training and validation sets.
    """
    # Load and filter images
    dataset = load_and_filter_images(dataset_dir, class_labels, img_size=img_size)
    
    # Shuffle, batch, and prefetch dataset
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Split dataset into training and validation sets
    train_ds, val_ds = split_dataset(dataset)
    
    return train_ds, val_ds

def build_and_compile_model(input_shape=(256, 256, 3), num_classes=2):
    """
    Build and compile the CNN model.
    """
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
        Dense(num_classes, activation='softmax')  # Adjust for the number of classes
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_ds, val_ds, model, epochs=20, batch_size=32):
    """
    Train the model and return the training history.
    """
    model_clone = clone_model(model)
    model_clone.set_weights(model.get_weights())
    model_clone.compile(optimizer=model.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model_clone.fit(train_ds, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val_ds)
    return history, model_clone

def evaluate_model(model, test_ds, class_labels, model_name="CNN", title="Model Evaluation"):
    """
    Evaluate the model and plot confusion matrix and classification report.
    """
    true_labels = np.array([])
    pred_labels = np.array([])

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        true_labels = np.concatenate([true_labels, labels.numpy()])
        pred_labels = np.concatenate([pred_labels, np.argmax(preds, axis=1)])
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels,
                          title=f'{model_name} \n Test: {title}',
                          figsize=(4, 4),
                          title_fontsize=10,
                          tick_labelsize=8,
                          colorbar_size={'shrink': 0.75, 'aspect': 20, 'pad': 0.04})

    # Print classification report
    print(classification_report(true_labels, pred_labels, target_names=class_labels))
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(4, 4),
                          title_fontsize=10,  
                          tick_labelsize=8,
                          colorbar_size={'shrink': 0.75, 'aspect': 20, 'pad': 0.04}):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar(im, **colorbar_size)  # Apply colorbar size adjustments
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=tick_labelsize)
    plt.yticks(tick_marks, classes, fontsize=tick_labelsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=tick_labelsize)
    plt.xlabel('Predicted label', fontsize=tick_labelsize)

############# prepare train and val data for train ###
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def prepare_train_val_data(dataset_dir, class_labels, img_size, seed=123, batch_size = 32):
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
    # Perform one-hot encoding on the labels after the split
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    
    train_gen = data_generator(np.array(X_train), y_train, img_size, batch_size, seed)
    val_gen = data_generator(np.array(X_val), y_val, img_size, batch_size, seed)

    return train_gen, val_gen, X_train, X_val, batch_size

# Example usage:
# train_gen, val_gen, X_train, X_val, batch_size = prepare_train_val_data(dataset_dir)


######################################################## VGG16 & ResNet ##################################

def VGG_model(input_shape, num_classes=2):
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# def vgg_model_v2(input_shape=(128, 128, 3), num_classes=2):
# #     base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#     base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
#     base_model.trainable = False
    
#     model = Sequential([
#         base_model,
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')  # 2 output classes
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     return model

# def vgg_model_v2(input_shape=(224, 224, 1), num_classes=2):
#     # Check if the input shape is grayscale (1 channel)
#     if input_shape[-1] == 1:
#         # Define the input layer for grayscale images
#         input_layer = Input(shape=input_shape, name='grayscale_input')
#         # Convert grayscale to RGB by duplicating the single channel 3 times
#         x = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_layer)
#     else:
#         # If the input shape is already RGB (3 channels), use it directly
#         input_layer = Input(shape=input_shape, name='rgb_input')
#         x = input_layer
    
#     # Load VGG16 base model without pretrained weights
#     base_model = VGG16(weights=None, include_top=False, input_tensor=x)
#     base_model.trainable = False  # Freeze the base model
    
#     # Add additional layers
#     x = Flatten()(base_model.output)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
    
#     # Define the complete model
#     model = Model(inputs=input_layer, outputs=predictions)
    
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     return model


def vgg_model_v2(input_shape=(224, 224, 3), num_classes=2):
    # Define the input layer for RGB images
    input_layer = Input(shape=input_shape, name='rgb_input')
    
    # Load VGG16 base model without pretrained weights
    base_model = VGG16(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False  # Freeze the base model
    
    # Add additional layers
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Define the complete model
    model = Model(inputs=input_layer, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model




def ResNet50_model(input_shape, num_classes=2):
    # Load ResNet50 without pretrained weights
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    
    # Define the input layer for grayscale images
    input_layer = Input(shape=(224, 224, 1), name='grayscale_input')
    
    x = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_layer)  # Convert grayscale to RGB
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def resnet50_model_v2(input_shape=(128, 128, 3)):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
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

def ResNet101_model(input_shape, num_classes=2):
    # Load ResNet101 without pretrained weights
    base_model = ResNet101(weights=None, include_top=False, input_shape=input_shape)
    
    # Define the input layer for grayscale images
    input_layer = Input(shape=(224, 224, 1), name='grayscale_input')
    
    x = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_layer)  # Convert grayscale to RGB
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def resnet101_model_v2(input_shape=(128, 128, 3)):
    base_model = ResNet101(weights=None, include_top=False, input_shape=input_shape)
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


################ General training and evaluation function
# def train_and_evaluate_model_vgg_resnet(dataset_dir_org, test_dataset_dir, model_fn, 
#                                         img_size=256, num_classes=2, epochs=10, batch_size = 32, title, model_name):
#     # Define the model using the provided model function
#     input_shape = (img_size, img_size, 3)
#     model = model_fn(input_shape, num_classes=num_classes)
    
#     # Get the class labels
#     class_labels = sorted(os.listdir(dataset_dir_org))
#     print("class_labels: ", class_labels)
    
#     # Prepare training and validation data
#     train_gen, val_gen, X_train, X_val, batch_size = prepare_train_val_data(dataset_dir_org, class_labels, img_size, batch_size)
#     steps_per_epoch = len(X_train) // batch_size
#     validation_steps = len(X_val) // batch_size
    
#     # Train the model
#     history = model.fit(
#         train_gen,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=val_gen,
#         validation_steps=validation_steps,
#         epochs=epochs
#     )
    
#     # Plot accuracy and loss
#     title = f"{model_name} - {data_name}"
#     plot_training_history(history, title, model_name, figsize=(14, 5))
    
#     # Evaluate the model on the test set
#     evaluation_title = f"{model_name} \n Test: {title}"
#     evaluate_test_set(test_dataset_dir, model, img_size, batch_size, data_generator, plot_confusion_matrix, class_labels, evaluation_title)

# def train_and_evaluate_model_vgg_resnet(dataset_dir_org, test_dataset_dirs, model_fn, model_name, data_name, test_name, 
#                                         img_size=256, num_classes=2, epochs=10, batch_size=32):
#     # Define the model using the provided model function
#     input_shape = (img_size, img_size, 3)
#     model = model_fn(input_shape, num_classes=num_classes)
    
#     # Get the class labels
#     class_labels = sorted(os.listdir(dataset_dir_org))
#     print("class_labels: ", class_labels)
    
#     # Prepare training and validation data
#     train_gen, val_gen, X_train, X_val, batch_size = prepare_train_val_data(dataset_dir_org, class_labels, img_size, batch_size)
#     steps_per_epoch = len(X_train) // batch_size
#     validation_steps = len(X_val) // batch_size
    
#     # Train the model
#     history = model.fit(
#         train_gen,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=val_gen,
#         validation_steps=validation_steps,
#         epochs=epochs
#     )
    
#     # Plot accuracy and loss
    
#     plot_training_history(history, data_name, model_name, figsize=(14, 5))
    
#     # Evaluate the model on each test set
#     for idx, test_dataset_dir in enumerate(test_dataset_dirs):
# #         evaluation_title = f"{model_name} - Test Set {idx + 1}: {title}"
#         test_name_1 = f"{test_name}_{idx}"
#         evaluate_test_set(test_dataset_dir, model, img_size, batch_size, data_generator, plot_confusion_matrix, class_labels, model_name, test_name_1)


def train_and_evaluate_model_vgg_resnet(dataset_dir_org, test_dataset_dirs, model_fn, model_name, data_name, test_name, 
                                        img_size=256, num_classes=2, epochs=10, batch_size=32):
    # Define the model using the provided model function
    input_shape = (img_size, img_size, 3)
    model = model_fn(input_shape, num_classes=num_classes)
    
    # Get the class labels
    class_labels = sorted(os.listdir(dataset_dir_org))
    print("class_labels: ", class_labels)
    
    # Prepare training and validation data
    train_gen, val_gen, X_train, X_val, batch_size = prepare_train_val_data(dataset_dir_org, class_labels, img_size, batch_size)
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs
    )
    
    # Plot accuracy and loss
    plot_training_history(history, data_name, model_name, figsize=(14, 5))
    
    # Evaluate the model on each test set
    for idx, test_dataset_dir in enumerate(test_dataset_dirs):
        test_name_1 = f"{test_name}_{idx + 1}"
        evaluate_test_set(test_dataset_dir, model, img_size, batch_size, data_generator, plot_confusion_matrix, class_labels, model_name, data_name, test_name_1)




####### Pretrained VGG16 ###########
def VGG_imagenet(input_shape, num_classes=2, shape= (224, 224, 1)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Define the input layer for grayscale images
    input_layer = Input(shape=shape, name='grayscale_input')
    # Apply the revised preprocessing function within a Lambda layer
    x = Lambda(preprocess_grayscale_to_rgb)(input_layer)
    # Use the preprocessed input with the base model
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
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

def evaluate_test_set(dataset_dir, model, img_size, batch_size, data_generator, plot_confusion_matrix, class_labels, model_name, data_name, test_name):
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
    test_gen = data_generator(np.array(X_test), y_test, img_size, batch_size)

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
    plot_confusion_matrix(cm_test, classes=class_labels, title=f"{model_name} Train: {data_name} \n Test: {test_name}")
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



#########################################
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D, Dense, Dropout


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
#     if num_classes == 2:
#         output = Dense(1, activation='sigmoid')(x)
#     else:
#         output = Dense(num_classes, activation='softmax')(x)
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

####################### EfficientNet Models ##########################
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

def build_efficientnet_model(model_type='EfficientNetB0', input_shape=(224, 224, 3), num_classes=2):
    model_class = globals()[model_type]
    base_model = model_class(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

######################## Xception Model ##############################
from tensorflow.keras.applications import Xception

def build_xception_model(input_shape=(299, 299, 3), num_classes=2):
    base_model = Xception(include_top=False, input_shape=input_shape, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

######################################################
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add, Activation
from tensorflow.keras.regularizers import l2

def deep_custom_cnn2(input_shape, num_classes):
    def res_block(x, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
        # Residual block helper function
        shortcut = x
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)

        if use_dropout:
            x = Dropout(0.3)(x)

        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    # Input Layer
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # First Set of Residual Blocks
    for _ in range(3):
        x = res_block(x, 64)

    # Transition Layer
    x = Conv2D(128, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)

    # Second Set of Residual Blocks
    for _ in range(4):
        x = res_block(x, 128)

    # Transition Layer
    x = Conv2D(256, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)

    # Third Set of Residual Blocks
    for _ in range(6):
        x = res_block(x, 256, use_dropout=True)  # Using dropout in deeper layers

    # Transition Layer
    x = Conv2D(512, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)

    # Fourth Set of Residual Blocks
    for _ in range(3):
        x = res_block(x, 512)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)

    # Output Layer
    output = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

###############################################
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dense, GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def deep_custom_cnn3(input_shape, num_classes):
    def res_block(x, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False, dropout_rate=0.3):
        # Residual block helper function
        shortcut = x
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)

        if use_dropout:
            x = Dropout(dropout_rate)(x)

        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.005))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    for _ in range(2):
        x = res_block(x, 64)

    x = Conv2D(128, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.005))(x)

    for _ in range(3):
        x = res_block(x, 128)

    x = Conv2D(256, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.005))(x)

    for _ in range(4):
        x = res_block(x, 256, use_dropout=True)

    x = Conv2D(512, (1, 1), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.005))(x)

    for _ in range(2):
        x = res_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def plot_training_history(history, data_name, model_name, figsize=(14, 5)):
    # Number of epochs
    epochs = len(history.history['loss'])
    epoch_list = list(range(1, epochs + 1))

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=figsize)

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epoch_list, train_accuracy, label='Train Accuracy')
    plt.plot(epoch_list, val_accuracy, label='Validation Accuracy')

    # Adding text annotation for each point
    for i, txt in enumerate(train_accuracy):
        plt.text(epoch_list[i], train_accuracy[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')
    for i, txt in enumerate(val_accuracy):
        plt.text(epoch_list[i], val_accuracy[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')

    plt.title(f'{model_name}::{data_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epoch_list)  # Set x-ticks to be the epoch numbers
    plt.legend()

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epoch_list, train_loss, label='Train Loss')
    plt.plot(epoch_list, val_loss, label='Validation Loss')

    # Adding text annotation for each point
    for i, txt in enumerate(train_loss):
        plt.text(epoch_list[i], train_loss[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')
    for i, txt in enumerate(val_loss):
        plt.text(epoch_list[i], val_loss[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')

    plt.title(f'{model_name}::{data_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epoch_list)
    plt.legend()

    plt.show()

