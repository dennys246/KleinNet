from __future__ import absolute_import, division, print_function, unicode_literals
import os, shutil, time, time, random, csv, tqdm, atexit, json	
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Ask tensorflow to not use GPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel as nib
from glob import glob
from numpy import asarray
from nilearn import plotting, datasets, surface
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
from random import randint, randrange

class KleinNet:
	def __init__(self):
		self.config = configuration() # Load default configuration

		self.model = None # Initialize model variable

		atexit.register(self.save) # Set up force save model before exiting

		print("\n - KleinNet Initialized -\n - Process PID - " + str(os.getpid()) + ' -\n')

	def orient(self, bids_dir, bold_identifier, label_identifier, exclude_trained = False):
		self.create_dir()

		# Attach orientation variables to object for future use
		self.bids_dir = bids_dir
		self.bold_identifier = bold_identifier
		self.label_identifier = label_identifier
		if os.path.exists(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/subjects_run.txt"):
			with open(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/subjects_run.txt", 'r') as file:
				self.previously_run = file.read().split('\n')
		else:
			self.previously_run = []
		print(f"\nOrienting and generating KleinNet lexicons for bids directory {bids_dir}...")
		# Grab all available subjects with fMRIPrep data
		self.subject_pool = []
		lexicon = [item for item in glob(f"{bids_dir}/derivatives/{self.config.tool}/sub-*") if os.path.isdir(item)]
		for subject in lexicon:
			subject_id = subject.split('/')[-1]

			if exclude_trained == True and subject_id in self.previously_run: # If subject was previously run, exclude from subject pool
				continue
			# Grab all available sessions

			sessions = glob(f"{subject}/ses-*/")
			if len(sessions) == 0:
				print(f"No sessions found for {subject_id}")
			for session in sessions:
				# Look if the subject has usable bold file
				bold_filename = glob(f"{session}/func/{bold_identifier}")
				if len(bold_filename) == 0:
					continue
				if len(bold_filename) > 1:
					bold_filename = '\n'.join(bold_filename)
					print(f"Multiple bold files found for {subject_id}...\n{bold_filename}")
					continue

				# Look if the subject has labels (regressors/classifiers)
				label_filename = glob(f"{session}/func/{label_identifier}")
				if len(label_filename) == 0:
					print(f"No labels found for {subject_id}, excluding from analysis...")
					continue
				if len(label_filename) > 1:
					label_filename = '\n'.join(label_filename)
					print(f"Multiple label files found for {subject_id}...\n{label_filename}")
					continue

				self.subject_pool.append(subject_id)

		subject_pool = '\n'.join(self.subject_pool)
		print(f"\n\nSubject pool available for use...\n{subject_pool}")

	def wrangle(self, subjects = [], count = 0, session = '*', activation = 'linear', shuffle = False, jackknife = None, exclude_trained = False):
		
		subject_count = 0
		if count == 0 and len(subjects) < 3:
			print("Not enough subjects passed into wrangle, must pass in at least 3 subjects to evenly split between test and training")
			return
		if count == 0:
			count = len(subjects)

		
		# Run through each subject and load data
		self.current_batch = subjects
		if os.path.exists(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/subjects_run.txt"):
			with open(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/subjects_run.txt", 'r') as file:
				self.previously_run = file.read().split('\n')
		else:
			self.previously_run = []
		
		images = np.array([])

		self.test_indices = []
		self.train_indices = []
		train_test_mod = 0

		while subject_count < count:
			if subjects == []: # If we've run out of subjects but need more
				subject = self.subject_pool.pop(0)
			else:
				subject = subjects.pop(0)
			print(subject)
			if subject in self.config.excluded_subjects: # Check if subject is in excluded list
				print(f'Excluding subject {subject} from run...')
				continue
			if exclude_trained == True and subject in self.previously_run: # check is subject previously run
				print(f"Subject {subject} previously run, skipping subject...")
				continue
			if subject != jackknife:
				image, label = self.load_subject(subject, session, activation, shuffle)
				if len(label) == 0: # If no images available
					print(f"Skipping {subject}, no labels...")
					continue # Skip subjects
				try:
					images = np.append(images, image, axis = 0)
					labels = np.append(labels, label)
				except:
					images = image
					labels = label

				if train_test_mod % 3 < 2:
					self.train_indices += [ind for ind in range(len(self.train_indices), len(self.train_indices) + len(image) - 1 )]
				else:
					self.test_indices += [ind for ind in range(len(self.test_indices), len(self.test_indices) + len(image) - 1)]
				train_test_mod += 1
			subject_count += 1
		if images.shape[0] == 0:
			return False
		if jackknife == None:
			print(f'Max test indice {max(self.test_indices)}\nMax train indices {max(self.train_indices)}\nLength of images {images.shape}\nLength of labels: {len(labels)}\nTraining Indices: {len(self.train_indices)}\nTesting Indices: {len(self.test_indices)}')
			self.x_train = images[self.train_indices,:,:,:]
			self.y_train = labels[self.train_indices]
			self.x_test = images[self.test_indices,:,:,:]
			self.y_test = labels[self.test_indices]
		else:
			self.x_train = images
			self.y_train = labels
			self.x_test, self.y_test = self.load_subject(jackknife, session, activation, shuffle)
		print(f"X train shape: {self.x_train.shape}")
		return True

	def load_subject(self, subject, session, activation = 'linear', shuffle = False, load_affine = False, load_header = False):
			print(subject)
			def empty_exit(load_affine, load_header):
				if load_affine == False and load_header == False:
					return [], []
				if load_header == False and load_affine == True:
					return [], [], []
				if load_header == True and load_affine == False:
					return [], [], []
				if load_header == True and load_affine == True:
					return [], [], [], []

			image_filenames = glob(f"{self.bids_dir}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.bold_identifier}")
			if len(image_filenames) == 0:
				print(f'No images found for {subject} ses-{session}')
				return empty_exit(load_affine, load_header)
			else:
				image_filename = image_filenames[0]
				image_file = nib.load(image_filename) # Load images

			header = image_file.header # Grab images header

			# Grab image shape and affine from header
			image_shape = header.get_data_shape() 

			# Reshape image to have time dimension as first dimension and add channel dimension
			image = image_file.get_fdata().reshape(image_shape[3], image_shape[0], image_shape[1], image_shape[2], 1)
			if self.config.data_shape == None:
				self.config.data_shape = image.shape[1:-1]
				print(f"Data shape: {self.config.data_shape}")

			# Normalize data
			image = self.normalize(image)
			# Grab fMRI affine transformation matrix
			affine = image_file.affine 

			# Load labels
			label_filenames = glob(f"{self.bids_dir}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.label_identifier}")
			if len(label_filenames) != 1:
				print(f"Multiple/no label files found for subject {subject}: {label_filenames}")
				empty_exit(load_affine, load_header)
			else:
				label_filename = label_filenames[0]
			labels = []

			print(label_filename)
			with open(label_filename, 'r') as label_file:
				if label_filename[-4:] == '.txt':
					labels = label_file.readlines()
					labels = [float(label) for label in ''.join(labels).split('\n') if label != '']
				if label_filename[-4:] == '.csv':
					labels = []
					csv_reader = csv.reader(label_filename)
					for row in csv_reader:
						labels.append(float(row))
				if label_filename[-4:] == '.tsv':
					tsv_reader = csv.reader(label_filename, '\t')
					for row in tsv_reader:
						labels.append(float(row))
				labels = np.array(labels)

			print(f'Subject {subject} image shape: {labels.shape}')
			if labels.shape[0] == 0 or labels.shape[0] != image.shape[0]:
				print(f"Labels are empty or labels length does not match image length...\n Image shape - {image.shape}\n Label shape - {labels.shape})")
				return empty_exit(load_affine, load_header)
			
			labels = self.normalize(labels)

			if activation != 'linear': # If not a regression problem
				labels = [int(label) for label in labels] # Convert labels to integers for classifing
			
			if self.config.shuffle == True or shuffle == True:
				image, labels = self.shuffle(image, labels)

			if load_affine == False and load_header == False:
				return image, labels
			if load_header == False and load_affine == True:
				return image, labels, affine
			if load_header == True and load_affine == False:
				return image, labels, header
			if load_header == True and load_affine == True:
				return image, labels, header, affine

	def shuffle(self, images, labels):
		indices = np.arange(images.shape[0])
		np.random.shuffle(indices)
		images = images[indices, :, :, :, :]
		labels = labels[indices]
		return images, labels
	
	def normalize(self, array):
		array_min = np.min(array)
		array_max = np.max(array)
		return (array - array_min) / (array_max - array_min)

	def mean_filter(image, filter_size):
		return
	
	def median_filter(image, filter_size):
		return

	def gaussian_filter(image, filter_size, sigma = 1):
		return
	
	def bilateral_filter(image, filter_size):
		return
	
	def laplacian_filter(image, filter_size, sign = 'positive'):
		return

	def dilation_filter(image, filter_size):
		return

	def erosion_filter(image, filter_size):
		return

	def plan(self):
		print("\nPlanning KleinNet model structure")
		self.filter_counts = []
		convolution_size = self.config.init_filter_count
		for depth in range(self.config.convolution_depth*2):
			self.filter_counts.append(convolution_size)
			convolution_size = convolution_size*2

		self.layer_shapes = []
		self.output_layers = []
		conv_shape = [self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2]]
		conv_layer = 1
		print(f"Convolution Shape: {conv_shape}")
		for depth in range(self.config.convolution_depth):
			if depth > 0:
				conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 3
			conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 4
			if depth < self.config.convolution_depth - 1:
				conv_shape = self.calcMaxPool(conv_shape)

		self.new_shapes = []
		print(f'Layer shapes...\n{self.layer_shapes}')
		for layer_ind, conv_shape in enumerate(self.layer_shapes):
			new_shape = self.calcConvTrans(conv_shape)
			for layer in range(layer_ind,  0, -1):
				new_shape = self.calcConvTrans(new_shape)
				if layer % 2 == 1 & layer != 1:
					new_shape = self.calcUpSample(new_shape)
			self.new_shapes.append(new_shape)

		for layer, plan in enumerate(zip(self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes)):
			print(f"Layer {layer + 1} ({plan[0]}) | Filter count: {plan[1]} | Layer Shape: {plan[2]} | Deconvolution Output: {plan[3]}")

	def calcConv(self, shape):
		return [(input_length - filter_length + (2*pad))//stride + 1 for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calcMaxPool(self, shape):
		return [(input_length - pool_length + (2*pad))//stride + 1 for input_length, pool_length, stride, pad in zip(shape, self.config.pool_size, self.config.pool_stride, self.config.padding)]

	def calcConvTrans(self, shape):
		if self.config.zero_padding == 'valid':
			return [(input_length - 1)*stride + filter_length for input_length, filter_length, stride in zip(shape, self.config.kernel_size, self.config.kernel_stride)]
		else:
			return [(input_length - 1)*stride + filter_length - 2*pad for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calcUpSample(self, shape):
		return [input_length * self.config.pool_stride[0] for input_length in shape]

	def build(self, load = False):
		# Plan out model structure
		if self.config.data_shape == None:
			self.wrangle(count = 3, session = 0)
		self.plan()

		self.checkpoint_path = f"{self.config.result_directory}{self.config.run_directory}/KleinNet/ckpt.weights.h5"
		print('\nConstructing KleinNet model')
		self.model = tf.keras.models.Sequential() # Create first convolutional layer
		for layer in range(1, self.config.convolution_depth + 1): # Build the layer on convolutions based on config convolution depth indicated
			# Removed -> input=tf.keras.Input((self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2], 1), self.config.batch_size)
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 2], self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.zero_padding, input_shape = (self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2], 1), use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias)))
				
			self.model.add(LeakyReLU(negative_slope = self.config.negative_slope))
			self.model.add(tf.keras.layers.BatchNormalization())
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 1], self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.zero_padding, use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias)))
			self.model.add(LeakyReLU(negative_slope = self.config.negative_slope))
			self.model.add(tf.keras.layers.BatchNormalization())
			if layer < self.config.convolution_depth:
				self.model.add(tf.keras.layers.MaxPooling3D(pool_size = self.config.pool_size, strides = self.config.pool_stride, padding = self.config.zero_padding, data_format = "channels_last"))
		if self.config.density_dropout[0] == True: # Add dropout between convolution and density layer
			self.model.add(tf.keras.layers.Dropout(self.config.dropout))
		self.model.add(tf.keras.layers.Flatten()) # Create heavy top density layers
		for density, dense_dropout in zip(self.config.top_density, self.config.density_dropout[1:]):
			self.model.add(tf.keras.layers.Dense(density, use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias))) # Density layer based on population size of V1 based on Full-density multi-scale account of structure and dynamics of macaque visual cortex by Albada et al.
			self.model.add(LeakyReLU(negative_slope = self.config.negative_slope))
			if dense_dropout == True:
				self.model.add(tf.keras.layers.Dropout(self.config.dropout))
		self.model.add(tf.keras.layers.Dense(1, activation=self.config.output_activation)) #Create output layer

		self.model.build()
		self.model.summary()

		if self.config.optimizer == 'Adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate, epsilon = self.config.epsilon, amsgrad = self.config.use_amsgrad)
		if self.config.optimizer == 'SGD':
			optimizer = tf.keras.optimizers.SGD(learning_rate = self.config.learning_rate, momentum = self.config.momentum, nesterov = self.config.use_nestrov)
		
		if self.config.output_activation == 'linear': # Compile model for regression task
			self.config.loss = 'mse'
			self.config.history_types = ['loss']
			self.model.compile(optimizer = optimizer, loss = self.config.loss) # Compile model
		else: # Else compile model for classification
			self.config.loss = 'binary_crossentropy'
			self.config.history_types = ['accuracy', 'loss']
			self.model.compile(optimizer = optimizer, loss = self.config.loss, metrics = ['accuracy']) # Compile mode

		print(f'\nKleinNet model compiled using {self.config.optimizer}')

		# Check if a model already exists and load	
		if self.load():
			print('KleinNet weights and history loaded...')
		else: # Else save new weights to checkpoint path
			if os.path.exists(self.checkpoint_path) == False or self.config.rebuild == True:
				self.model.save_weights(self.checkpoint_path)
			self.model_history = {}
			for history_type in self.config.history_types:
				self.model_history[history_type] = [] 
				self.model_history[f"val_{history_type}"] = [] 
			print(f"Model weights reinitialized")

		# Create a callback to the saved weights for saving model while training
		self.callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path + '',
											save_weights_only=True,
											verbose=1)]

	def train(self):
		print(f"\nx-train: {self.x_train.shape}\ny-train: {self.y_train.shape}\n\nx_test: {self.x_test.shape}\ny_test: {self.y_test.shape}")
		self.history = self.model.fit(self.x_train, self.y_train, epochs = self.config.epochs, batch_size = self.config.batch_size, validation_data = (self.x_test, self.y_test), callbacks = self.callbacks)
		print(f"Train history: {self.history}")
		for history_type in self.config.history_types: # Save training history
			self.model_history[history_type] += self.history.history[history_type]
			self.model_history[f'val_{history_type}'] += self.history.history[f'val_{history_type}']

		self.previously_run += self.current_batch
		with open(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/subjects_run.txt", "w") as output:
			output.write('\n'.join(self.previously_run))

	def test(self):
		self.history = self.model.evaluate(self.x_test,  self.y_test, verbose=2)
		print(f"Test History: {self.history}")
		for history_type in self.config.history_types: # Save test history
			self.model_history[f"val_{history_type}"] += self.history.history[f"val_{history_type}"]

	def save(self):
		if self.model != None:
			self.model.save_weights(self.checkpoint_path) # Save model
			with open(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/history.json", 'w') as file:
				json.dump(self.model_history, file) # Save model history
	
	def load(self):
		if os.path.exists(self.checkpoint_path):
			if len(os.listdir('/'.join(self.checkpoint_path.split('/')[:-1]))) > 0:
				try:
					self.model.load_weights(self.checkpoint_path)
					with open(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/history.json", 'r') as file:
						self.model_history = json.load(file)
					print('KleinNet loaded successfully')
					return True
				except:
					print('KleinNet weights and history failed to load...')
					return False
			else:
				print('KleinNet not found...')
				return False

	def plot_accuracy(self, i = 1):
		print("\nEvaluating KleinNet model accuracy & loss...")
		for history_type in self.config.history_types:		# Evaluate the model accuracy and loss
			plt.plot(self.model_history[history_type], label=history_type)
			plt.plot(self.model_history[f"val_{history_type}"], label = f'validation {history_type}')
			plt.xlabel('Epoch')
			plt.ylabel(history_type)
			plt.legend(loc='upper right')
			plt.ylim([0, 1])
			title = f"~learnig rate: {str(self.config.learning_rate)} ~negative_slope: {str(self.config.negative_slope)} ~bias: {str(self.config.bias)} ~optimizer: {self.config.optimizer}"
			if self.config.optimizer == 'SGD':
				title = f"{title} ~epsilon: {str(self.config.epsilon)}"
			else:
				title = f"{title} ~momentum: {str(self.config.momentum)}"
			plt.title(title)
			plt.savefig(f"{self.config.result_directory}{self.config.run_directory}/KleinNet/Model_{str(i)}_{history_type}.png")
			plt.close()

	def observe(self, interest):
		self.images = []
		ind = 0
		while self.images == []: # Iterate till you find a subject
			self.images, self.labels, self.header, self.affine = self.load_subject(self.subject_pool[ind], session = '1', load_affine = True, load_header = True)
			ind += 1

		self.sample_label = -2
		while self.sample_label <= interest - 0.25 or self.sample_label >= interest + 0.25: # Grab next sampsle that is the other category
			rand_ind = random.randint(0, self.images.shape[0] - 1)
			self.sample_label = self.labels[rand_ind] # Grab sample label
		self.sample = self.images[rand_ind, :, :, :, :] # Grab sample volume
#		self.sample = self.sample.reshape((1, self.sample.shape[0], self.sample.shape[1], self.sample.shape[2], self.sample.shape[3]))

		for category, label in zip(self.config.outputs_category, self.config.outputs):
			if interest == label:
				self.category = category

		print(f"\nObserving {self.category} outcome structure")

		print(f"\nExtracting {interest} answer features from KleinNet convolutional layers...")
		self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes
		layer_outputs = [layer.output for layer in self.model.layers[:]]
		layer_names = [layer.name for layer in self.model.layers if layer.name[:6] == 'conv3d']
		print(f'layer names: {layer_names}')
		for self.layer in range(1, (self.config.convolution_depth*2 + 1)): # Build deconvolutional models for each layer
			self.model(tf.keras.Input(self.sample.shape))
			print(f"Model new input: {self.model.input}")
			print(f"Model layer output to be applied to activation {self.model.get_layer(layer_names[self.layer - 1]).output}")
			
			#self.model.input
			self.activation_model = tf.keras.models.Model(inputs = tf.keras.Input(self.sample.shape), outputs = [self.model.get_layer(layer_names[self.layer - 1]).output]) 
			print(f"Model outputs - {self.activation_model.output} \n\n Layer outputs - {layer_outputs[self.output_layers[self.layer - 1]]}")

			self.deconv_model = tf.keras.models.Sequential() # Create first convolutional layer
			print(f"Deconv model shape - {self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], self.layer_shapes[self.layer - 1][2]}")
			self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, kernel_size = self.config.kernel_size, strides = self.config.kernel_stride, input_shape = (self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], self.layer_shapes[self.layer - 1][2], 1), kernel_initializer = tf.keras.initializers.Ones()))
			for deconv_layer in range(self.layer - 1, 0, -1): # Build the depths of the deconvolution model
				if deconv_layer % 2 == 1 and deconv_layer != 1:
					self.deconv_model.add(tf.keras.layers.UpSampling3D(size = self.config.pool_size, data_format = 'channels_last'))
				self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, self.config.kernel_size, strides = self.config.kernel_stride, kernel_initializer = tf.keras.initializers.Ones()))
			print(f'Summarizing layer {self.layer} deconvolution model')
			self.deconv_model.build()
			self.deconv_model.summary()
			print(f"Sample shape {self.sample.shape}")
			self.activation_model.summary()
			self.sample = self.sample.reshape((1, self.sample.shape[0], self.sample.shape[1], self.sample.shape[2], self.sample.shape[3]))
			self.feature_maps, predictions = self.activation_model.predict(self.sample) # Grab feature map using single volume
			self.feature_maps = self.feature_maps[0, :, :, : ,:].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2], self.current_shape[3])

			for map_index in range(self.feature_maps.shape[3]): # Save feature maps in glass brain visualization pictures
				feature_map = (self.feature_maps[:, :, :, map_index].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2])) # Grab Feature map
				deconv_feature_map = self.deconv_model.predict(self.feature_maps[:, :, :, map_index].reshape(1, self.current_shape[0], self.current_shape[1], self.current_shape[2], 1)).reshape(self.new_shape[0], self.new_shape[1], self.new_shape[2])
				self.plot_all(deconv_feature_map, 'DeConv_Feature_Maps', map_index)
			print(f"\n\nExtracting KleinNet model class activation maps for layer {self.layer}")
			
			with tf.GradientTape() as gtape: # Create CAM
				conv_output, predictions = self.activation_model(self.sample)
				loss = predictions[:, np.argmax(predictions[0])]
				grads = gtape.gradient(loss, conv_output)
				pooled_grads = K.mean(grads, axis = (0, 1, 2, 3))

			self.heatmap = tf.math.reduce_mean((pooled_grads * conv_output), axis = -1)
			self.heatmap = np.maximum(self.heatmap, 0)
			max_heat = np.max(self.heatmap)
			if max_heat == 0:
				max_heat = 1e-10
			self.heatmap /= max_heat

			# Deconvolute heatmaps and visualize
			self.heatmap = self.deconv_model.predict(self.heatmap.reshape(1, self.current_shape[0], self.current_shape[1], self.current_shape[2], 1)).reshape(self.new_shape[0], self.new_shape[1], self.new_shape[2])
			self.plot_all(self.heatmap, 'CAM', 1)


	# Feature output linear correlation analysis
	# Within this function we will correlated feature node 
	# activity to their output value within each layer and 
	# feature.
	def folc(self):
		# For each layer
		
		# For each feature map

		# For each node

		# Iterate through a subjects bold images
		# Collect each feature map activity

		# Correlate the feature node activity to the output (NN output or label?)

		# Output 3D r coefficient matrix to their T1/T2 and plot
			# Which slices to use until we build GUI?

		# Convolve feature maps with r coefficient matrix and plot
		return

# How do we incorperate the unique attributes and personality of a child?
# ElasticNet and MVPA integration


	def plot_all(self, data, data_type, map_index):
		self.surf_stat_maps(data, data_type, map_index)
		#self.glass_brains(data, data_type, map_index)
		#self.stat_maps(data, data_type, map_index)

	def prepare_plots(self, data, data_type, map_index, plot_type):
		affine = self.header.get_best_affine()
		max_value, min_value, mean_value, std_value = describe_data(data)
		#-Thresholding could take some more consideration-#
		threshold = 0
		intensity = 0.5
		data = data * intensity
		# ---------------------------------------------- #
		data = nib.Nifti1Image(data, affine = self.affine, header = self.header) # Grab feature map
		title = f"{layer} {data_type} Map {str(map_index)} for  {self.category} Answer"
		output_folder = f"{self.config.result_directory}{self.config.run_directory}{self.catergory}/Layer_{self.layer}/{data_type}/{plot_type}/"
		return data, title, threshold, output_folder

	def glass_brains(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Glass_Brain")
		plotting.plot_glass_brain(stat_map_img = data, black_bg = True, plot_abs = False, display_mode = 'lzry', title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) + '-' + self.category + '_category.png')) # Plot feature map using nilearn glass brain - Original threshold = (mean_value + (std_value*2))

	def stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Stat_Maps")
		for display, midfix, cut_coord in zip(['z', 'x', 'y'], ['-zview-', '-xview-', '-yview-'], [6, 6, 6]):
			plotting.plot_stat_map(data, bg_img = self.anatomy, display_mode = display, cut_coords = cut_coord, black_bg = True, title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) +  midfix + self.category + '_category.png')) # Plot feature map using nilearn glass brain

	def surf_stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Surf_Stat_Maps")
		fsaverage = datasets.fetch_surf_fsaverage()

		texture = surface.vol_to_surf(data, fsaverage.pial_left)
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-medial-' + self.category + '_category.png'))

		texture = surface.vol_to_surf(data, fsaverage.pial_right)
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-medial-' + self.category + '_category.png'))

	def jack_knife(self, Range = None):
		for self.jackknife in self.subject_pool:
			print(f"Running Jack-Knife on Subject {str(self.jackknife)}")
			self.wrangle(self.subject_pool, self.jackknife)
			self.build()
			self.train()
			self.plot_accuracy()
			self.ROC()

	def ROC(self):
		self.probabilities = self.model.predict(self.x_test).ravel()
		np.save(f"{self.config.result_directory}{self.config.run_directory}/Jack_Knife/Probabilities/Sub-{str(self.jackknife)}_Volumes_Prob.np", self.probabilities)
		fpr, tpr, threshold = roc_curve(self.y_test, self.probabilities)
		predictions = np.argmax(self.probabilities, axis=-1)
		AUC = auc(fpr, tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr, tpr, label = 'RF (area = {:.3f})'.format(AUC))
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(f'Subject {str(self.jackknife)} ROC Curve')
		plt.legend(loc = 'best')
		plt.savefig(f"{self.config.result_directory}{self.config.run_directory}/Jack_Knife/Sub_{str(self.jackknife)}_ROC_Curve.png")
		plt.close()

	def create_dir(self):
		first_dir = self.config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = [f'Layer_{str(layer)}' for layer in range(1, self.config.convolution_depth*2 + 1)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		if self.config.rebuild == True:
			if os.path.exists(f'{self.config.result_directory}{self.config.run_directory}/') == True:
				print(f'\nRun directory {self.config.result_directory}{self.config.run_directory} already exists, clearing directory...')
				shutil.rmtree(f'{self.config.result_directory}{self.config.run_directory}')
				time.sleep(1)
		else: # If not resetting model
			if os.path.isdir(f'{self.config.result_directory}{self.config.run_directory}/') == True: # If model exists
				print(f"Run directory already exists for {self.config.run_directory}, consider altering run directory or deleting model (i.e. setting config.reset_model = True)")
				return

		os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/')
		os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/KleinNet')
		os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/Jack_Knife')
		os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/Jack_Knife/Probabilities')
		for first in first_dir:
			os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/{first}')
			for second in second_dir:
				os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/{first}/{second}')
				for third in third_dir:
					os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/{first}/{second}/{third}')
					for fourth in fourth_dir:
						os.mkdir(f'{self.config.result_directory}{self.config.run_directory}/{first}/{second}/{third}/{fourth}')
		print(f'\nResult directories generated for {self.config.run_directory}\n')
		

class configuration:

	def __init__(self):
		self.build() # Build configuration

	def build(self):

		#-------------------------------- Model Set-Up -------------------------------#
		#These initial variables are used by KleinNet and won't need to be set to anything
		self.data_shape = None
		self.checkpoint_path = None
		self.history_types = ['Accuracy', 'Loss']
		self.excluded_subjects = []

		#-------------------------------- Shuffle Data ------------------------------##
		# Shuffle will zip all images and labels and shuffle the data before assigning
		# them to training and testing sets. Defaults to true, however you might want
		# to change this to false if you think your question has a time dimension to it
		self.shuffle = False


		#--------------------------------- Rebuild Model ----------------------------##
		# This variable defines wether a new model will be build each time KleinNet is 
		# called. It can be useful to set this to True when initially setting up the model
		# so the model will be rebuilt with new configurations.
		self.rebuild = False

		#------------------------------- Run parameters ------------------------------#
		# Epoch and Batch Size - Both used to describe how the model will run, and
		# for how long. Epochs represents how many times the data will be presented to
		# the model for deep learning. The batch size simply defines how the model will
		# batch the samples into miniature training samples to help plot model performance.
		self.epochs = 10
		self.batch_size = 50

		#---------------------------- Model Hyperparameters ---------------------------#
		# Model Hyperparameters - Hyperparameters used within the models algorithms to
		# to learn about the dataset. These values were found while optimizing the model
		# over a simple stroop dataset so consider using the optiimize function within
		# the KleinNet library to find optimum values. Bias can be a bit tricky to optimize
		# and I would recommend using the KleinNet.optimum_bias() to find bias when using
		# an inbalanced dataset. Hyperparameter descriptors to be added with GUI.
		self.negative_slope = 0.1 # Formally known as alpha 
		self.epsilon = 1e-6
		self.learning_rate = 0.0001
		self.bias = 0
		self.dropout = 0.5
		self.momentum = 0.01

		# The Kernel initializer - is used to initialize the state of the model. We initialize
		# the model (e.g. weights, biases, etc.) using Xavier (glorot) uniform which randomly selects
		# values from a uniform gaussian distribution. I found this initializer to be superior to others
		# at the time of building the model however you can change the initializer by typing
		# in the tensorflow string code for the initializer here (e.g. 'glorot_uniform')
		self.kernel_initializer = 'glorot_uniform'

		# Convolution Depth - KleinNet is built to use a basic convolution layer structure
		# that is stacked based on how deep the model is indicated her. Having the depth
		# set at 2 meaning will cause the model to build 2 convolutions layers from
		# the convolution template in the KleinNet.build() function before building the top density
		self.convolution_depth = 2

		# Initial Filter Count - KleinNet convolution layer filter sizes are calculated
		# within the KleinNet.plan() function using a common machine learning rule of
		# doubling filter count per convolution layer. init_filter_count is the initial
		# value the filters starts on before doubling.
		self.init_filter_count = 1

		# Kernel Size & Stride  - These variables are used to decide what the convolution
		# kernel size will be along with how it moves across the layer to generate Features.
		# Generally the bigger the kernel stride, the small the output which... could be a good thing?
		self.kernel_size = (2, 2, 2)
		self.kernel_stride = (1, 1, 1)

		# Zero Padding - Padding is used to decide if 0's will be added onto the edge of
		# the input to make sure the convolutions don't move outside of the model and crash
		# your script. 'valid' padding means no 0's will be added to the side were 'same'
		# padding means 0's will be added to the edges of the layer input. The padding
		# variable declares, if using same padding, the size of the padding to add on the the edges
		self.zero_padding = 'valid'
		self.padding = (0, 0, 0)

		# Max Pooling - Max pooling is used to generally reduce the size of the input.
		# These layers will general a pool kernel of size pool_size and move throughout the layers input
		# based on pool_stride. The max pooling layer finds the max value within the kernel's
		# area and pools it all into a smaller space.
		self.pool_size = (2, 2, 2)
		self.pool_stride = (2, 2, 2)

		# Top Density Layer(s) - The following variables are used to define the structure
		# of the top density. The top_density variable holds the sizes of each layer were
		# the density_dropout layer defines whether there is dropout moving into that layer.
		# You might notice that the density_dropout variables hold an extra value conpared
		# to the top_density variable and this is to account for the flattening layer that
		# is automatically built within the KleinNet.build() function. The first value of
		# density_dropout[0] corresponds to dropout applied to the flatterning layer.
		self.top_density = [100, 40, 20]
		self.density_dropout = [True, False, False, False]


		# Outputs - The output variables are used to help the model process and understand what it
		# is classifying and help it display some of the output better. These variables are
		# also used within KleinNet.build() to create the output layer. The output activation
		# is used to decide what activation the model will us in it's output layer.
		self.output_activation = 'linear'
		self.outputs = [0.0, 1.0]
		self.outputs_category = ['Negative', 'Positive']

		# Optimizers - This section is used to help switch between different Optimizers
		# without having to worry about changing code too much. While talking about
		# SGD/Adam and Nestrov/AMSGrad is not within the scope of this config file I
		# would recommend looking up literature to find which would be best for you.
		self.optimizers = ['SGD', 'Adam']
		self.optimizer = 'SGD' # Set to either 'SGD' or 'Adam'
		self.use_nestrov = True # If using SDG optimizer
		self.use_amsgrad = True # If using Adam optimizer

		# Loss - This variable describes the loss calculation used within the model.
		# the standard used while initially building KleinNet was binary crossentropy
		# however you may need to change this based on the questions you are asking.
		self.loss = 'mse'

		# Folder Structure - These variables are used to desribes where data is stored
		# along with where to store the outputs of KleinNet. The results directory is
		# a general folder where specific model run folder will be created. The run
		# directory is the specific folder KleinNet will generate and output results into.
		# The data directory is the folder that holds all the data to be inputed into model.
		# Note that this data directory should include the folder parents up to the main
		# main BIDS formatted folder that holds all experiment data
		self.result_directory = 'kleinnet/'
		self.run_directory = 'run_1'

		self.tool = 'fmriprep'