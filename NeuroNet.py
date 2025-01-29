from __future__ import absolute_import, division, print_function, unicode_literals
import os, atexit, pipeline, observer, config, psutil
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob

class NeuroNet:
	
	def __init__(self, network_folder = None):

		self.config = config.build(network_folder) or config.build() # Load default configuration
		
		if self.config == False: # If configuration build failed
			print("Configuration failed, exiting...") 
			return # Exit

		self.wrangler = pipeline.wrangler(self.config)
		self.lens = observer.lens(self.config)

		self.model = None # Initialize model variable

		atexit.register(self.save) # Set up force save model before exiting

		print("\n - NeuroNet Initialized -\n - Process PID - " + str(os.getpid()) + ' -\n')
	
	def orient(self, bids_directory, bold_identifier, label_identifier, exclude_trained = False):
		self.wrangler.create_dir()

		# Attach orientation variables to object for future use
		self.config.bids_directory = bids_directory
		self.config.bold_identifier = bold_identifier
		self.config.label_identifier = label_identifier

		# Grab all available subjects with fMRIPrep data
		self.subject_pool = []
		print(f"\nOrienting and generating NeuroNet lexicon for bids directory {self.config.bids_directory}...")
		
		# Generate a lexicon of all potential subjects
		lexicon = [item for item in glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/sub-*") if os.path.isdir(item)]
		# Iterate through each subject found
		for subject in lexicon:
			subject_id = subject.split('/')[-1]

			# If subject was previously run, exclude from subject pool
			if exclude_trained == True and subject_id in self.config.previously_run: 
				continue

			# Grab all available sessions
			sessions = glob(f"{subject}/ses-*/")

			# If none found alert about missing data
			if len(sessions) == 0:
				print(f"No sessions found for {subject_id}")

			# Iterate through each session
			for session in sessions:

				# Look if the subject has usable bold file
				bold_filename = glob(f"{session}/func/{self.config.bold_identifier}")
				if len(bold_filename) == 0: # If none found, skip
					continue
				if len(bold_filename) > 1: # If too many found, skip
					bold_filename = '\n'.join(bold_filename)
					print(f"Multiple bold files found for {subject_id}...\n{bold_filename}")
					continue

				# Look if the subject has labels (regressors/classifiers)
				label_filename = glob(f"{session}/func/{self.config.label_identifier}")
				if len(label_filename) == 0: # If none found, skip
					print(f"No labels found for {subject_id}, excluding from analysis...")
					continue
				if len(label_filename) > 1: # If too many found, continue
					label_filename = '\n'.join(label_filename)
					print(f"Multiple label files found for {subject_id}...\n{label_filename}")
					continue

				# Add subject to subject pool if it passed criteria
				if subject_id not in self.subject_pool:
					self.subject_pool.append(subject_id) 

		# Print found subject pool to user
		self.config.subject_pool = self.subject_pool
		subject_pool = '\n'.join(self.subject_pool)
		print(f"\n\nSubject pool available for use...\n{subject_pool}")
		
	def load(self, subjects = [], count = 0, session = '*', activation = 'linear', shuffle = False, jackknife = None, exclude_trained = True):
		if len(subjects) > 0: count = len(subjects)

		if count >= 0:
			results = self.wrangler.wrangle(self.subject_pool, subjects, count, session, activation, shuffle, jackknife, exclude_trained)
			if results != False:
				self.x_train, self.y_train, self.x_test, self.y_test = results
			else:
				return False
		else:
			self.wrangler.wrangle(self.subject_pool, subjects, count, session, activation, shuffle, jackknife, exclude_trained)
		return True

	def plan(self):
		print("\nPlanning NeuroNet model structure")
		# initialize an empty list to store layer filter counts
		self.filter_counts = []
		
		#Iterate through each layer and calculate filter counts
		convolution_size = self.config.init_filter_count
		for depth in range(self.config.convolution_depth):
			self.filter_counts.append(convolution_size)
			convolution_size = convolution_size*2 # Double the size each layer

		self.layer_shapes = []
		self.output_layers = []
		conv_shape = [self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2]]
		conv_layer = 1
		print(f"Convolution Shape: {conv_shape}")
		for depth in range(self.config.convolution_depth):
			if depth > 0:
				conv_shape = self.calc_conv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 1
			if depth < self.config.convolution_depth - 1:
				conv_shape = self.calc_maxpool(conv_shape)

		self.new_shapes = []
		print(f'Layer shapes...\n{self.layer_shapes}')
		for layer_ind, conv_shape in enumerate(self.layer_shapes):
			new_shape = self.calc_convtrans(conv_shape)
			for layer in range(layer_ind,  0, -1):
				new_shape = self.calc_convtrans(new_shape)
				if layer != 1:
					new_shape = self.calc_upsample(new_shape)
			self.new_shapes.append(new_shape)

		for layer, plan in enumerate(zip(self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes)):
			print(f"Layer {layer + 1} ({plan[0]}) | Filter count: {plan[1]} | Layer Shape: {plan[2]} | Deconvolution Output: {plan[3]}")

	def build(self):
		# Plan out model structure
		
		if self.config.data_shape == None:
			self.wrangler.wrangle(self.subject_pool, count = -1, session = 0, shape_extraction = True)
		self.plan()

		self.checkpoint_path = f"{self.config.project_directory}{self.config.model_directory}/model/ckpt.weights.h5"

		print('\nConstructing NeuroNet model')
		self.model = tf.keras.models.Sequential() # Create first convolutional layer

		# Add in initial input layer
		self.model.add(tf.keras.Input((self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2], 1), self.config.batch_size))
		
		for layer in range(self.config.convolution_depth): # Build the layer on convolutions based on config convolution depth indicated
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer], self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.zero_padding, use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias)))
			self.model.add(tf.keras.layers.BatchNormalization())
			self.model.add(tf.keras.layers.LeakyReLU(self.config.negative_slope))
			self.model.add(SpatialAttention())
			if layer + 1 < self.config.convolution_depth:
				self.model.add(tf.keras.layers.MaxPooling3D(pool_size = self.config.pool_size, strides = self.config.pool_stride, padding = self.config.zero_padding, data_format = "channels_last"))
			if self.config.dropout:
				self.model.add(tf.keras.layers.Dropout(self.config.dropout))

		if self.config.multiscale_pooling:
			self.model.add(MultiScalePooling())

		self.model.add(tf.keras.layers.Flatten()) # Create top density layers
		
		for density, dense_dropout in zip(self.config.top_density, self.config.density_dropout):
			self.model.add(tf.keras.layers.Dense(density, use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias))) # Density layer based on population size of V1 based on Full-density multi-scale account of structure and dynamics of macaque visual cortex by Albada et al.
			self.model.add(tf.keras.layers.LeakyReLU(self.config.negative_slope))
			if dense_dropout == True:
				self.model.add(tf.keras.layers.Dropout(self.config.dropout))
		self.model.add(tf.keras.layers.Dense(1, activation=self.config.output_activation)) #Create output layer

		self.model.build()
		self.model.summary()

		if self.config.learning_rate == 'cos':
			self.config.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
				initial_learning_rate=self.config.learning_rate,
				decay_steps=10000  # Total steps to reach the minimum learning rate
			)

		if self.config.optimizer == 'Adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate, epsilon = self.config.epsilon, amsgrad = self.config.use_amsgrad)
		elif self.config.optimizer == 'SGD':
			optimizer = tf.keras.optimizers.SGD(learning_rate = self.config.learning_rate, momentum = self.config.momentum, nesterov = self.config.use_nestrov)
		else:
			optimizer = self.config.optimizer

		if self.config.output_activation == 'linear': # Compile model for regression task
			self.config.loss = 'mse'
			self.config.history_types = ['loss']
			self.model.compile(optimizer = optimizer, loss = self.config.loss, run_eagerly = True) # Compile model
		else: # Else compile model for classification
			self.config.loss = 'binary_crossentropy'
			self.config.history_types = ['accuracy', 'loss']
			self.model.compile(optimizer = optimizer, loss = self.config.loss, metrics = ['accuracy', 'loss'], run_eagerly=True) # Compile mode

		print(f'\nNeuroNet model compiled using {self.config.optimizer}')

		# Check if a model already exists and load	
		if self.load_model():
			print('NeuroNet weights and history loaded...')
		else: # Else save new weights to checkpoint path
			if os.path.exists(self.checkpoint_path) == False or self.config.rebuild == True:
				self.model.save_weights(self.checkpoint_path)
			self.config.model_history = {}
			for history_type in self.config.history_types:
				self.config.model_history[history_type] = [] 
				self.config.model_history[f"val_{history_type}"] = [] 
			print(f"Model weights reinitialized")

		# Create a callback to the saved weights for saving model while training
		self.callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path + '',
											save_weights_only=True,
											verbose=1)]

	def calc_conv(self, shape):
		return [(input_length - filter_length + (2*pad))//stride + 1 for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calc_maxpool(self, shape):
		return [(input_length - pool_length + (2*pad))//stride + 1 for input_length, pool_length, stride, pad in zip(shape, self.config.pool_size, self.config.pool_stride, self.config.padding)]

	def calc_convtrans(self, shape):
		if self.config.zero_padding == 'valid':
			return [(input_length - 1)*stride + filter_length for input_length, filter_length, stride in zip(shape, self.config.kernel_size, self.config.kernel_stride)]
		else:
			return [(input_length - 1)*stride + filter_length - 2*pad for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calc_upsample(self, shape):
		return [input_length * self.config.pool_stride[0] for input_length in shape]

	def train(self):
		# Display info about the training set
		print(f"\nx-train: {self.x_train.shape}\ny-train: {self.y_train.shape}\n\nx-test: {self.x_test.shape}\ny-test: {self.y_test.shape}")
		
		# Fit the model to the training set
		self.history = self.model.fit(self.x_train, self.y_train, epochs = self.config.epochs, batch_size = self.config.batch_size, validation_data = (self.x_test, self.y_test), callbacks = self.callbacks)
		print(f"Train history: {self.history}")
		
		# Iterate through all history types and add too run history object
		for history_type in self.config.history_types: # Save training history
			self.config.model_history[history_type] += self.history.history[history_type]
			self.config.model_history[f'val_{history_type}'] += self.history.history[f'val_{history_type}']

		# Add subjects trained on to previously run batch and save
		self.config.previously_run += self.wrangler.current_batch 

	def test(self):
		# Test model by evaluating on test set
		self.history = self.model.evaluate(self.x_test,  self.y_test, verbose=2)
		print(f"Test History: {self.history}")
		for history_type in self.config.history_types: # Save test history
			self.config.model_history[f"val_{history_type}"] += self.history.history[f"val_{history_type}"]

	def predict(self, x_range = None, subject_id = ""):
		output = self.model.predict(self.x_test)
		
		if x_range == None:
			x_range = range(50)

		if subject_id != "":
			fileend = f"_{subject_id}.png"
		else:
			fileend = ".png"

		plt.plot(x_range, [output[value] for value in x_range], color = 'orange')
		plt.plot(x_range, [self.y_test[value] for value in x_range], color = 'blue')
		plt.legend()
		plt.xlabel("Sample Volume")
		plt.ylabel("Value")
		plt.title(f"Predictions vs. Actual {self.config.output_descriptor} for fMRI Images")
		plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/plots/prediction_vs_real_output{fileend}")
		plt.close()
		
		plt.scatter(self.y_test, output, color='blue', label='Outcomes')
		plt.plot(self.config.outputs, self.config.outputs, color='gray', linestyle='-', label="Unity Line")
		plt.xlabel(f'Actual {self.config.output_descriptor}')
		plt.ylabel(f'Predicted {self.config.output_descriptor}')
		plt.title(f'Scatter Plot of {self.config.output_descriptor} Predicted vs. Actual Outcomes')
		plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/plots/prediction_vs_real_scatterplot{fileend}")
		plt.close()

	def jack_knife(self, Range = None):
			for self.jackknife in self.subject_pool:
				print(f"Running Jack-Knife on Subject {str(self.jackknife)}")
				self.wrangle(self.subject_pool, self.jackknife)
				self.build()
				self.train()
				self.lens.plot_accuracy()
				self.lens.ROC()

	def save(self):
		if self.model != None:
			self.model.save_weights(self.checkpoint_path) # Save model
			self.config.save_config()
	
	def load_model(self):
		if os.path.exists(self.checkpoint_path):
			if len(os.listdir('/'.join(self.checkpoint_path.split('/')[:-1]))) > 0:
				#try:
				self.model.load_weights(self.checkpoint_path)
				self.config.load_config()
				print('NeuroNet loaded successfully')
				return True
				#except:
				#	print('NeuroNet weights and history failed to load...')
				#	return False
			else:
				print('NeuroNet not found...')
				return False
	
	def display_memory(self):
		# Get the memory information
		memory = psutil.virtual_memory()
		swap = psutil.swap_memory()

		# Display RAM usage
		print("---- System Memory (RAM) ----")
		print(f"Total RAM: {memory.total / (1024 ** 3):.2f} GB")
		print(f"Available RAM: {memory.available / (1024 ** 3):.2f} GB")
		print(f"Used RAM: {memory.used / (1024 ** 3):.2f} GB")
		print(f"RAM Usage: {memory.percent}%")

		# Display swap memory usage
		print("\n---- Swap Memory ----")
		print(f"Total Swap: {swap.total / (1024 ** 3):.2f} GB")
		print(f"Used Swap: {swap.used / (1024 ** 3):.2f} GB")
		print(f"Free Swap: {swap.free / (1024 ** 3):.2f} GB")
		print(f"Swap Usage: {swap.percent}%")

class SpatialAttention(tf.keras.layers.Layer):
	def __init__(self, kernel_size=7, **kwargs):
		super(SpatialAttention, self).__init__(**kwargs)
		self.kernel_size = kernel_size
		
	def build(self, input_shape):
		self.concat = tf.keras.layers.Concatenate(axis=-1)
		self.multiply = tf.keras.layers.Multiply()
		
		self.conv = tf.keras.layers.Conv3D(
			filters=1, 
			kernel_size=(1, self.kernel_size, self.kernel_size), 
			strides=1, 
			padding="same", 
			activation="sigmoid"
		)

	def call(self, inputs):
		# Calculate average-pooling and max-pooling
		avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
		max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

		# Concatenate pooled tensors
		concat = self.concat([avg_pool, max_pool]) 

		# Compute spatial attention map
		attention = self.conv(concat)

		# Scale by attention map
		output = self.multiply([inputs, attention])
		return output

	def get_config(self):
		config = super(SpatialAttention, self).get_config()
		config.update({"kernel_size": self.kernel_size})
		return config


class MultiScalePooling(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(MultiScalePooling, self).__init__(**kwargs)
		# Define the pooling layers with different kernel sizes
		self.pool1 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding='same')
		self.pool2 = tf.keras.layers.MaxPooling3D(pool_size = (4, 4, 4), strides = (4, 4, 4), padding='same')
		self.pool3 = tf.keras.layers.AveragePooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding='same')
		self.small_upsample = tf.keras.layers.UpSampling3D(size = (2, 2, 2))
		self.large_upsample = tf.keras.layers.UpSampling3D(size = (4, 4, 4))

	def call(self, inputs):
		# Apply the pooling operations
		pooled1 = self.pool1(inputs)
		pooled2 = self.pool2(inputs)
		pooled3 = self.pool3(inputs)

		# Reshape so all pools are the same size
		pooled2_resized = self.large_upsample(pooled2) 
		pooled3_resized = self.small_upsample(pooled3)

		# Concatenate the outputs along the channel axis
		return tf.keras.layers.Concatenate(axis=-1)([pooled1, pooled2_resized, pooled3_resized])
	
	def compute_output_shape(self, input_shape):
		# Compute output shape for the layer
		pooled_shape = (
			input_shape[0],  # Batch size
			input_shape[1] // 2,  # Depth reduced by pool1
			input_shape[2] // 2,  # Height reduced by pool1
			input_shape[3] // 2,  # Width reduced by pool1
			input_shape[4] * 3   # Concatenated channels
			)
		return pooled_shape