# Machine Learning Homework 4 - Image Classification

__author__ = 'tcs9pk'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


### Already implemented
def get_data(datafile):
	dataframe = pd.read_csv(datafile)
	dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	#normalizes.
	images /= 255
	return images, labels


def get_test_data(datafile):
	dataframe = pd.read_csv(datafile)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	# normalizes.
	images /= 255
	return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=10, save=True, hot=True):
	layer1 = trained_model.layers[0]
	weights = layer1.get_weights()[0]
	# Feel free to change the color scheme
	colors = 'hot' if hot else 'binary'

	for i in range(num_to_display):
		wi = weights[:, i].reshape(28, 28)
		#wi = np.tile(np.array(weights[:, i]), (28,28))
		plt.imshow(wi, cmap=colors, interpolation='nearest')
		if save:
			plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
		else:
			plt.show()

### Already implemented
def output_predictions(predictions):

	filename = 'predictions.txt'
	with open(filename, 'w+') as f:
		for pred in predictions:
			f.write(str(pred) + '\n')


def plot_history(history):

	train_loss_history = history.history['loss']
	val_loss_history = history.history['val_loss']

	train_acc_history = history.history['acc']
	val_acc_history = history.history['val_acc']

	epochs = np.arange(0, 4, 1)

	plt.plot(train_loss_history)
	plt.xticks(epochs, ([str(thing) for thing in epochs]))
	plt.xlabel("Epoch")
	plt.ylabel("Training Loss")
	plt.title("Training Loss History")
	# plt.show()
	plt.savefig('CNN_Train_Loss.png')
	plt.gcf().clear()

	plt.plot(val_loss_history)
	plt.xticks(epochs, ([str(thing) for thing in epochs]))
	plt.xlabel("Epoch")
	plt.ylabel("Validation Loss")
	plt.title("Validation Loss History")
	#plt.show()
	plt.savefig('CNN_Val_Loss.png')
	plt.gcf().clear()

	plt.plot(train_acc_history)
	plt.xticks(epochs, ([str(thing) for thing in epochs]))
	plt.xlabel("Epoch")
	plt.ylabel("Training Accuracy")
	plt.title("Training Accuracy History")
	#plt.show()
	plt.savefig('CNN_Train_Acc.png')
	plt.gcf().clear()

	plt.plot(val_acc_history)
	plt.xticks(epochs, ([str(thing) for thing in epochs]))
	plt.xlabel("Epoch")
	plt.ylabel("Validation Accuracy")
	plt.title("Validation Accuracy History")
	#plt.show()
	plt.savefig('CNN_Val_Acc.png')
	plt.gcf().clear()

def create_mlp(args=None):
	# You can use args to pass parameter values to this method
	# make the model.
	# Define model architecture
	# Define model architecture
	model = Sequential()
	# edit
	# model.add(Dense(units=10, activation='relu', input_shape=(28, 28, 1)))
	model.add(Dense(units=10, activation='relu', input_dim=28 * 28))
	# edit
	# model.add(MaxPooling2D(pool_size=(2, 1)))
	model.add(keras.layers.Dropout(0.3))
	# edit
	# model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(10, activation='softmax'))
	sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
	# print(model.summary())

	return model

def train_mlp(x_train, y_train, x_vali, y_vali, args=None):
	# You can use args to pass parameter values to this method
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	#y_train = y_train.squeeze()
	print ' shape '
	print(np.shape(x_train))
	print ' y shape'
	print np.shape(y_train)

	model = create_mlp(args)

	history = model.fit(x_train, y_train,
			  epochs=10, validation_split=.10, batch_size=128)
	return model, history


def create_cnn(args=None):
	# You can use args to pass parameter values to this method

	# 28x28 images with 1 color channel
	input_shap = (28, 28, 1)

	# Define model architecture
	model = Sequential()

	model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape = input_shap))
	model.add(keras.layers.Dropout(0.3))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Conv2D(32, kernel_size=4, activation='relu'))
	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))
	sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	x_train = x_train.reshape(-1, 28, 28, 1)
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_cnn(args)
	history = model.fit(x_train,
						y_train,
						nb_epoch=4,
						validation_split=.10,
						batch_size=128)

	return model, history


def train_and_select_model(train_csv):
	"""Optional method. You can write code here to perform a 
	parameter search, cross-validation, etc. """

	x_train, y_train = get_data(train_csv)

	x_valid = []
	y_valid = []

	# y is a 1 d array
	# x is a 2 d array

	args = {
		'learning_rate': 0.01,
	}
	#if type == 'CNN':
	model, history = train_cnn(x_train, y_train, x_valid, y_valid, args=None)
	#else:
	#	model, history = train_mlp(x_train, y_train, x_valid, y_valid, args=None)

	return model, history


def test_model(model_type):

	if model_type == 'CNN':

		train_file, test_file = 'fashion_train.csv', 'fashion_test.csv'

		# train your best model
		model, the_history = train_and_select_model(train_file)

		x_test, y_test = get_test_data(test_file)
		x_test = x_test.reshape(10000, 28, 28, 1)
		y_test = keras.utils.to_categorical(y_test, num_classes=10)
		# prediction = best_model.evaluate(x_test, y_test, verbose=0)
		# use your best model to generate predictions for the test_file
		predictions = model.predict_classes(x_test)
		output_predictions(predictions)
		score = model.evaluate(x_test, y_test, verbose=0)
		print model.summary()
		#print(score)

		val_loss_history = the_history.history['val_loss']
		val_acc_history = the_history.history['val_acc']

		print('Val loss:', sum(val_loss_history) / len(val_loss_history))
		print('Val accuracy:', sum(val_acc_history) / len(val_acc_history))


	else:
		train_file, test_file = 'fashion_train.csv', 'fashion_test.csv'

		# train your best model
		model, the_history = train_and_select_model(train_file, 'MLP')
		x_test, y_test = get_data(test_file)
		#x_test = x_test.reshape(10000, 28, 28, 1)
		y_test = keras.utils.to_categorical(y_test, num_classes=10)
		# prediction = best_model.evaluate(x_test, y_test, verbose=0)
		# use your best model to generate predictions for the test_file
		predictions = model.predict_classes(x_test)
		output_predictions(predictions, 'MLP')
		#score = model.evaluate(x_test, y_test, verbose=0)
		print model.summary()
		val_loss_history = the_history.history['val_loss']
		val_acc_history = the_history.history['val_acc']

		print('Val loss:', sum(val_loss_history)/len(val_loss_history))
		print('Val accuracy:', sum(val_acc_history) / len(val_acc_history))

	return None
if __name__ == '__main__':
	### Before you submit, switch this to grading_mode = False and rerun ###

	grading_mode = False
	if grading_mode:
		if (len(sys.argv) != 3):
			print("Usage:\n\tpython3 fashion.py train_file test_file")
			exit()
		train_file, test_file = sys.argv[1], sys.argv[2]

		# train your best model

		best_model, best_history = train_and_select_model(train_file)

		# use your best model to generate predictions for the test_file
		x_test, y_test = get_test_data(test_file)

		x_test = x_test.reshape(10000, 28, 28, 1)
		y_test = keras.utils.to_categorical(y_test, num_classes=10)

		predictions = best_model.predict_classes(x_test)

		output_predictions(predictions)
		# Include all of the required figures in your report. Don't generate them here.

	else:
		train_file = 'fashion_train.csv'
		test_file = 'fashion_test.csv'
		print 'Set grading_mode to true to evaluate.'
		# MLP
		#mlp_model, mlp_history = train_and_select_model(train_file, 'MLP')

		#val_loss_history = mlp_history.history['val_loss']
		#val_acc_history = mlp_history.history['val_acc']

		#print('Val loss:', sum(val_loss_history) / len(val_loss_history))
		#print('Val accuracy:', sum(val_acc_history) / len(val_acc_history))

		# plot acc plot loss
		#plot_history(mlp_history)
		#test_model('MLP')
		# FIX THIS
		#visualize_weights(mlp_model)
		#print mlp_model.summary()

		# okay make preds.

		# CNN
		#cnn_model, cnn_history = train_and_select_model(train_file)
		#print(cnn_model.summary())
		#plot_history(cnn_history)
		#layer1 = cnn_model.layers[0]
		#print(layer1.get_config())
		#layer3 = cnn_model.layers[2]
		#print(layer3.get_config())

		#val_loss_history = cnn_history.history['val_loss']
		#val_acc_history = cnn_history.history['val_acc']

		#print('Val loss:', sum(val_loss_history) / len(val_loss_history))
		#print('Val accuracy:', sum(val_acc_history) / len(val_acc_history))

		#test_model('CNN')
		#print(cnn_model.summary())
