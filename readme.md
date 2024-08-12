\- N. Abinash, Partheban KV

### Dataset: 
- Taken from kaggle (https://www.kaggle.com/datasets/enalis/tomatoes-dataset)
- Contains 7226 images of tomatoes in four different states: Unripe, Ripe, Old, and Damaged.
- For this model only images in the dataset within the subdirectories "Ripe" and "Unripe" are used for training the model.
- 90/5/5 Split (total: 3949)
	Ripe: 2195
	  Train: 1975
	  Test: 110
	  Validation: 110
	  
	Unripe: 1754
	 Train: 1585
	 Test: 84
	 Validation: 85

### Model:
-  The model is VGG16 connected with a Multilayer Perceptron block which consists of a fully connected layer and an output layer.
- Images of size 224 x 224 with 3 channels are fed into the frozen convolutional layers of VGG16. This is then fed into the last two unfrozen convolutional layers for fine tuning. This is connected to a MLP block.
- The output layer has two nodes (for ripe and unripe) with softmax activation function.
- Hyperparameters:
	- Drop Out: 0.3
	- Total Epochs:100
	- Early Stopping: 20
	- Optimizer: Adam
	- BatchSize: 16
	- Initial Learning Rate: 1e-3
- This model is based on this existing paper: https://www.ijisae.org/index.php/IJISAE/article/view/2538/1121

### Source Code:
```import tensorflow as tf

from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

  

# Define the input shape

input_shape = (224, 224, 3)

num_classes = 2  

learning_rate = 0.001

batch_size = 16

epochs = 100

early_stopping_patience = 20

  

# Load pre-trained VGG16 model without top classification layers

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

  

# Build the classification head on top of the base model

model = Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

  

# Freeze the convolutional layers of the VGG16 model

for layer in base_model.layers:

    layer.trainable = False

  

# Compile the model

model.compile(optimizer=Adam(lr=learning_rate),#, momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['accuracy'])

  

# Early stoppage

early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1, restore_best_weights=True)

  

# Model summary to review architecture

model.summary()

  

# Define paths to your dataset (replace these with your actual dataset paths)

train_data_dir = r'C:\Users\lenovo\Documents\Tomat\tomat_train\train'

validation_data_dir = r'C:\Users\lenovo\Documents\Tomat\tomat_train\validation'

test_data_dir = r'C:\Users\lenovo\Documents\Tomat\tomat_train\test'

  

# Data augmentation for the training dataset

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

  

# Data augmentation for the validation and test datasets

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

  

# Data generators

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='categorical'

)

  

validation_generator = validation_datagen.flow_from_directory(

    validation_data_dir,

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='categorical'

)

  

test_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='categorical'

)

  

# Train the model

model.fit(

    train_generator,

    steps_per_epoch=train_generator.samples // batch_size,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples // batch_size,

    epochs=epochs

)

  

# Unfreeze the last two layers of the base model for fine-tuning

for layer in base_model.layers[-2:]:

    layer.trainable = True

  

# Compile the model again after unfreezing

model.compile(optimizer=Adam(lr=learning_rate),#, momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['accuracy'])

  

# Continue training with fine-tuning

model.fit(

    train_generator,

    steps_per_epoch=train_generator.samples // batch_size,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples // batch_size,

    epochs=epochs,

    callbacks=[early_stopping]

)

  

# Evaluate the model

test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

  

# Save the trained model

model.save('tomato_ripeness_model.h5')
```
### Next Step/ Suggestions:
- Variation in the model to make it a novel approach.
