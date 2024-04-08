from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
from tensorflow import keras
import nvitop

batch_size = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256

datagen = ImageDataGenerator(
    rescale=1.0/255,
    # rotation_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2)

train_generator = datagen.flow_from_directory("./data",subset='training',batch_size=batch_size,target_size=(IMG_WIDTH,IMG_WIDTH))
validation_generator = datagen.flow_from_directory("./data",subset='validation',batch_size=batch_size,target_size=(IMG_WIDTH,IMG_WIDTH))


num_classes=90
input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
#Build the model
model = keras.Sequential(
    [
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(128,activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

train_step_per_epoch = train_generator.samples//batch_size
validation_steps = validation_generator.samples//batch_size

history = model.fit(train_generator,steps_per_epoch=train_step_per_epoch,
                    epochs=100,
                    validation_data=validation_generator,validation_steps=validation_steps)