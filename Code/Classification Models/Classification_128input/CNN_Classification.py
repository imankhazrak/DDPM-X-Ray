import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold

def fit_classification_model_cv(X, y, n_splits=5):
    """
    Fit a classification model using stratified K-fold cross-validation.

    Parameters:
    - X (numpy.ndarray): Input data (features).
    - y (numpy.ndarray): Target labels.
    - n_splits (int): Number of splits for stratified K-fold cross-validation.

    Returns:
    - pd.DataFrame: DataFrame containing metrics for each fold.
    - tf.keras.models.Sequential: Best model found during cross-validation.
    """
    # Initialize stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    # Define fixed model parameters
    optimizer = Adam(learning_rate=0.0001)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    epochs = 10
    batch_size = 32
    
    # Initialize lists to store metrics across folds
    fold_metrics = []
    best_model = None
    best_val_accuracy = 0.0
    
    fold = 0
    for train_index, val_index in skf.split(X, y):
        fold += 1
        print(f"Processing Fold {fold}...")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Define the model architecture
        model = Sequential([
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
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
            Dense(2, activation='softmax')  # Changed to 2 for the two classes
        ])
        
        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val))
        
        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        
        # Round the metrics for storage
        rounded_train_loss = [round(loss, 2) for loss in history.history['loss']]
        rounded_train_accuracy = [round(acc, 2) for acc in history.history['accuracy']]
        rounded_val_loss = [round(loss, 2) for loss in history.history['val_loss']]
        rounded_val_accuracy = [round(acc, 2) for acc in history.history['val_accuracy']]
        rounded_val_accuracy_mean = round(val_accuracy, 2)        
        # Append metrics to fold_metrics
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
    
    # Create DataFrame from fold_metrics
    fold_metrics_df = pd.DataFrame(fold_metrics)
    
    return fold_metrics_df, best_model
