import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 30,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.3,
    zoom_range=0.3,
    fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='grayscale',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='grayscale',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_generator.__next__()
num_classes = 7

model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation = 'relu', padding='same', input_shape=(64,64,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Block 2
model.add(Conv2D(128, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))


# Block 3
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Block 4
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Block 5
model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


# Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

model.summary()
no_of_train_images = 0
for rrot, dirs, files in os.walk(TRAIN_DIR):
    no_of_train_images += len(files)
    
    
no_of_test_images = 0
for rrot, dirs, files in os.walk(TEST_DIR):
    no_of_test_images += len(files)
    

model.fit(train_generator,
          steps_per_epoch=no_of_train_images//32,
          epochs=150,
          validation_data=validation_generator,
          validation_steps=no_of_test_images//32,
          callbacks=[reduce_lr, early_stop])
model.save("model_weights.h5")