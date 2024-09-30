import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(4, 4),
                          title_fontsize=14,  
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




def plot_training_history(history, title, model_name, figsize=(14, 5)):
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

    plt.title(f'{model_name}::{title}')
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

    plt.title(f'{model_name}::{title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epoch_list)
    plt.legend()

    plt.show()



def plot_training_history2(history, title, model_name, figsize=(10, 5)):
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

    # # Adding text annotation for each point
    # for i, txt in enumerate(train_accuracy):
    #     plt.text(epoch_list[i], train_accuracy[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')
    # for i, txt in enumerate(val_accuracy):
    #     plt.text(epoch_list[i], val_accuracy[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')

    plt.title(f'{model_name}::{title}')
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

    # # Adding text annotation for each point
    # for i, txt in enumerate(train_loss):
    #     plt.text(epoch_list[i], train_loss[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')
    # for i, txt in enumerate(val_loss):
    #     plt.text(epoch_list[i], val_loss[i], f"{txt:.2f}", fontsize=8, ha='center', va='bottom')

    plt.title(f'{model_name}::{title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epoch_list)
    plt.legend()

    plt.show()
