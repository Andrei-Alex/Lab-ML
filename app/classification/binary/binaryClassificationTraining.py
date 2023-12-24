import itertools
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

from app.utils.constants import binaryClassificationModelFilepath

# Set the level of logging for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Print out the number of available GPUs for TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Define file paths for the model and data
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# Setup the ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=10.0,
    fill_mode='nearest'
)

# Generate batches of tensor image data for training, validation and testing
train_batch = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(256, 256),
    classes=['correct', 'incorrect'],
    batch_size=10
)
valid_batch = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=valid_path,
    target_size=(256, 256),
    classes=['correct', 'incorrect'],
    batch_size=10
)
test_batch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(256, 256), classes=['correct', 'incorrect'], batch_size=10,
                         shuffle=False)

imgs, labels = next(train_batch)
print(train_batch)

# Function to plot and save a batch of images
def plotImages(images):
    output_folder = 'debug_plot_folder'
    output_image_path = os.path.join(output_folder, 'output_image.png')
    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax, in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_image_path)

plotImages(imgs)
print(labels)

# Build the model if it doesn't exist
if os.path.isfile(binaryClassificationModelFilepath) is False:
    input_shape = (256, 256, 3)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(2, activation='sigmoid', name='output_layer'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.save(binaryClassificationModelFilepath)

# Load and summarize the model
model = load_model(binaryClassificationModelFilepath)
model.summary()
print(model.optimizer)

# Define model training parameters
batch_size = 16
epochs = 200
model_checkpoint_callback = ModelCheckpoint(
    './binary_classification_model/checkpoint.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)

# Train the model
model.fit(
    x=train_batch,
    validation_data=valid_batch,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    shuffle=True,
    callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping]
)
model.save(binaryClassificationModelFilepath)

# Predict and generate confusion matrix
predictions = model.predict(x=test_batch, verbose=0)
cm = confusion_matrix(y_true=test_batch.classes, y_pred=np.argmax(predictions, axis=-1))

# Function to plot and save the confusion matrix
def plot_CM(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists('cm'):
        os.makedirs('cm')

    image_path = os.path.join('cm', 'confusion_matrix.png')
    plt.savefig(image_path)
    print(f"Confusion matrix saved as: {image_path}")

print(test_batch.class_indices)
cm_plot_labels = ['correct', 'incorrect']
plot_CM(cm=cm, classes=cm_plot_labels, title='Confusion_Matrix')
