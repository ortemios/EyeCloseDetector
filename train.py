import os
import random
import os.path
import tensorflow as tf
import numpy as np
import cv2
from keras import backend as K
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import ResNet50
from matplotlib.pyplot import imshow
from PIL import Image


random.seed()
root = os.path.dirname(os.path.abspath(__file__))
train_dir = root + "\\dataset\\train\\"
validation_dir = root + "\\dataset\\validation\\"
weights_file = root + "\\model\\weights.hdf5"
model_file = root + "\\model\\model.h5"
trained_model_file = root + "\\model\\trained_model.h5"
input_width = 64
input_height = 64
input_channels = 3
color_mode = 'rgb'

# Configure the TF backend session
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
K.set_session(tf.Session(config=tf_config))

model_input = layers.Input(shape=(input_width, input_height, input_channels))
x = layers.Conv2D(32, (5, 5), activation='relu')(model_input)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
model_output = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(model_input, model_output)
model.compile(
	optimizer=optimizers.RMSprop(lr=1e-4), 
	loss='binary_crossentropy',
	metrics=['acc'])
model.summary()
model.save(model_file)
	
model.summary()

def apply_blur(img):
	img = img.astype(np.uint8)
	# crop
	side = int(img.shape[1] * (1 - random.random()*0.6))
	x = int((img.shape[1] - side) * random.random())
	y = int((img.shape[0] - side) * random.random())
	img = img[x:x+side, y:y+side]
	img = cv2.resize(img, (input_width, input_height))
	
	# blur
	size = int(random.random()*side*0.03)+1
	kernel_motion_blur = np.zeros((size, size))
	kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
	kernel_motion_blur = kernel_motion_blur / size
	#img = cv2.filter2D(img, -1, kernel_motion_blur)
	
	# display
	if not True:
		x = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imshow("Image", x)
		cv2.waitKey()
		
	return img.astype(np.float32) / 255

# Load dataset generators
train_datagen = ImageDataGenerator(
	rotation_range=20,
	#width_shift_range=0.2,
	#height_shift_range=0.2,
	#shear_range=0.2,
	#zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest',
	preprocessing_function=apply_blur)
train_generator = train_datagen.flow_from_directory(
	train_dir,
	interpolation="bilinear",
	target_size=(input_width, input_height),
	batch_size=20,
	color_mode=color_mode,
	class_mode='binary')
validation_datagen = ImageDataGenerator(
	preprocessing_function=apply_blur)
validation_generator = validation_datagen.flow_from_directory(
	validation_dir,
	interpolation="bilinear",
	target_size=(input_width, input_height),
	batch_size=20,
	color_mode=color_mode,
	class_mode='binary')
	
checkpoint = ModelCheckpoint(weights_file, save_best_only=True, mode='max')

prompt = 'Enter epochs count: '
epochs = input(prompt)
while(epochs != '0'):
	history = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=int(epochs),
		validation_data=validation_generator,
		validation_steps=50,
		verbose=2,
		callbacks=[checkpoint])
	epochs = input(prompt)
model.save(trained_model_file)
print("Training complete!")