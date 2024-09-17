import numpy as np
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
import tensorflow as tf
import os

# Define dataset directories
dataset_dir = 'D:/sens-II/dataset'

# Hyperparameters
batch_size = 8
epochs = 20
initial_learning_rate = 0.0001

# Load VGG16 pre-trained model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the first few layers
for layer in base_model.layers[:15]:
    layer.trainable = False

# Add custom top layers for skin cancer classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)  # Three classes with softmax activation

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
              loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              metrics=['accuracy'])

# Print model summary
model.summary()

# Create ImageDataGenerators for train, validation, and test sets
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of data for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for test data

# Load and augment training images
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'
)

# Load validation images
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='validation'
)

# Load testing images (no augmentation)
test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model without early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Define Google Drive paths
drive_path = '/content/drive/MyDrive/skin_cancer_detection'
if not os.path.exists(drive_path):
    os.makedirs(drive_path)

h5_model_path = os.path.join(drive_path, 'skin_cancer_detection_model.h5')
tflite_model_path = os.path.join(drive_path, 'skin_cancer_detection_model.tflite')

# Save the model
model.save(h5_model_path)
print("Model saved successfully to Google Drive as .h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to Google Drive
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved successfully to Google Drive as .tflite")
