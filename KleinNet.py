from __future__ import absolute_import, division, print_function, unicode_literals
import os, shutil, time, time, random, config, csv, tqdm
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
		self.config = configuration()
		print("\n - KleinNet Initialized -\n - Process PID - " + str(os.getpid()) + ' -\n')
		try:
			self.cwd = os.getcwd()
			os.chdir(f"{self.config.run_directory}/Layer_1/")
			os.chdir(self.cwd)
		except:
			self.create_dir()

	def orient(self, bids_dir, bold_identifier, label_identifier):
		# Attach orientation variables to object for future use
		self.bids_dir = bids_dir
		self.bold_identifier = bold_identifier
		self.label_identifier = label_identifier

		print(f"\nOrienting and generating KleinNet lexicons for bids directory {bids_dir}...")
		# Grab all available subjects with fMRIPrep data
		self.subject_pool = []
		self.subject_folders = [item for item in glob(f"{bids_dir}/derivatives/{self.config.tool}/sub-*") if os.path.isdir(item)]
		for subject in self.subject_folders:
			subject_id = subject.split('/')[-1]
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
					print(f"Multiple bold files found for {subject_id}...\n{'\n'.join(bold_filename)}")
					continue

				# Look if the subject has labels (regressors/classifiers)
				label_filename = glob(f"{session}/func/{label_identifier}")
				if len(label_filename) == 0:
					print(f"No labels found for {subject_id}, excluding from analysis...")
					continue
				if len(label_filename) > 1:
					print(f"Multiple label files found for {subject_id}...\n{'\n'.join(label_filename)}")
					continue

				self.subject_pool.append(subject_id)

		print(f"Subject pool available for use...\n{'\n'.join(self.subject_pool)}")

	def wrangle(self, subjects = None, jackknife = None):
		# Run through each subject and load data
		for subject in subjects:
			if subject != jackknife:
				image, label = self.load_subject(subject)
				try:
					images = np.append(images, image, axis = 0)
					labels = np.append(labels, label)
				except:
					images = image
					labels = label
		if config.shuffle == True:
			images, labels = self.shuffle(images, labels)
		if jackknife == None:
			self.x_train = images[:(round((images.shape[0]/3)*2)),:,:,:]
			self.y_train = labels[:(round((len(labels)/3)*2))]
			self.x_test = images[(round((images.shape[0]/3)*2)):,:,:,:]
			self.y_test = labels[(round((len(labels)/3)*2)):]
		else:
			self.x_train = images
			self.y_train = labels
			self.x_test, self.y_test = self.load_subject(jackknife)

	def load_subject(self, subject, session = '*'):
			image_filename = glob(f"{self.bids_dir}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.bold_identifier}")[0]
			image_file = nib.load(image_filename) # Load images

			header = image_file.header # Grab images header

			# Grab image shape and affine from header
			image_shape = header.get_data_shape() 

			# Reshape image to have time dimension as first dimension and add channel dimension
			image = image_file.get_fdata().reshape(image_shape[3], image_shape[0], image_shape[1], image_shape[2], 1)
			
			# Grab fMRI affine transformation matrix
			affine = image_file.affine 

			# Load labels
			label_filename = glob(f"{self.bids_dir}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.label_identifier}")[0]
			labels = []
			with open(label_filename, 'r') as label_file:
				if label_filename[-4:] == '.txt':
					labels = label_file.readlines()
					labels = ''.join(labels).split('\n')
				if label_filename[-4:] == '.csv':
					labels = []
					csv_reader = csv.reader(label_filename)
					for row in csv_reader:
						labels.append(row)
				if label_filename[-4:] == '.tsv':
					tsv_reader = csv.reader(label_filename, '\t')
					for row in tsv_reader:
						labels.append(row)
				labels = np.array(labels)
			
			return image, labels


	def shuffle(self, images, labels):
		indices = np.arange(images.shape[0])
		np.random.shuffle(indices)
		images = images[indices, :, :, :, :]
		labels = labels[indices]
		return images, labels

	def plan(self):
		print("\nPlanning KleinNet model structure")
		self.filter_counts = []
		convolution_size = config.init_filter_count
		for depth in range(config.convolution_depth*2):
			self.filter_counts.append(convolution_size)
			convolution_size = convolution_size*2

		self.layer_shapes = []
		self.output_layers = []
		conv_shape = [config.x_size, config.y_size, config.z_size]
		conv_layer = 1
		for depth in range(config.convolution_depth):
			conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 3
			conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 4
			if depth < config.convolution_depth - 1:
				conv_shape = self.calcMaxPool(conv_shape)

		self.new_shapes = []
		for layer_ind, conv_shape in enumerate(self.layer_shapes):
			new_shape = self.calcConvTrans(conv_shape)
			for layer in range(layer_ind,  0, -1):
				new_shape = self.calcConvTrans(new_shape)
				if layer % 2 == 1 & layer != 1:
					new_shape = self.calcUpSample(new_shape)
			self.new_shapes.append(new_shape)

		for layer, plan in enumerate(zip(self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes)):
			print("Layer ", layer + 1, " (", plan[0], ")| Filter count:", plan[1], "| Layer Shape: ", plan[2], "| Deconvolution Output: ", plan[3])

	def calcConv(self, shape):
		return [(input_length - filter_length + (2*pad))//stride + 1 for input_length, filter_length, stride, pad in zip(shape, config.kernel_size, config.kernel_stride, config.padding)]

	def calcMaxPool(self, shape):
		return [(input_length - pool_length + (2*pad))//stride + 1 for input_length, pool_length, stride, pad in zip(shape, config.pool_size, config.pool_stride, config.padding)]

	def calcConvTrans(self, shape):
		if config.zero_padding == 'valid':
			return [round((input_length - 1)*stride + filter_length) for input_length, filter_length, stride in zip(shape, config.kernel_size, config.kernel_stride)]
		else:
			return [round(input_length*stride) for input_length, filter_length, stride in zip(shape, config.kernel_size, config.kernel_stride)]

	def calcUpSample(self, shape):
		return [round((input_length - 1)*(filter_length/stride)*2) for input_length, filter_length, stride in zip(shape, config.pool_size, config.pool_stride)]

	def build(self):
		try:
			self.filter_counts
		except:
			self.plan()
		print('\nConstructing KleinNet model')
		self.model = tf.keras.models.Sequential() # Create first convolutional layer
		for layer in range(1, config.convolution_depth + 1): # Build the layer on convolutions based on config convolution depth indicated
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 2], config.kernel_size, strides = config.kernel_stride, padding = config.zero_padding, input_shape = (config.x_size, config.y_size, config.z_size, 1), use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias)))
			self.model.add(LeakyReLU(alpha = config.alpha))
			self.model.add(tf.keras.layers.BatchNormalization())
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 1], config.kernel_size, strides = config.kernel_stride, padding = config.zero_padding, use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias)))
			self.model.add(LeakyReLU(alpha = config.alpha))
			self.model.add(tf.keras.layers.BatchNormalization())
			if layer < config.convolution_depth:
				self.model.add(tf.keras.layers.MaxPooling3D(pool_size = config.pool_size, strides = config.pool_stride, padding = config.zero_padding, data_format = "channels_last"))
		if config.density_dropout[0] == True: # Add dropout between convolution and density layer
			self.model.add(tf.keras.layers.Dropout(config.dropout))
		self.model.add(tf.keras.layers.Flatten()) # Create heavy top density layers
		for density, dense_dropout in zip(config.top_density, config.density_dropout[1:]):
			self.model.add(tf.keras.layers.Dense(density, use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias))) # Density layer based on population size of V1 based on Full-density multi-scale account of structure and dynamics of macaque visual cortex by Albada et al.
			self.model.add(LeakyReLU(alpha = config.alpha))
			if dense_dropout == True:
				self.model.add(tf.keras.layers.Dropout(config.dropout))
		self.model.add(tf.keras.layers.Dense(1, activation=self.config.output_activation)) #Create output layer

		self.model.build()
		self.model.summary()

		if config.optimizer == 'Adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate, epsilon = config.epsilon, amsgrad = config.use_amsgrad)
		if config.optimizer == 'SGD':
			optimizer = tf.keras.optimizers.SGD(learning_rate = config.learning_rate, momentum = config.momentum, nesterov = config.use_nestrov)
		self.model.compile(optimizer = optimizer, loss = config.loss, metrics = ['accuracy']) # Compile model and run
		print('\nKleinNet model compiled using', config.optimizer)

	def train(self):
		self.history = self.model.fit(self.x_train, self.y_train, epochs = config.epochs, batch_size = config.batch_size, validation_data = (self.x_test, self.y_test))

	def test(self):
		self.loss, self.accuracy = self.model.evaluate(self.x_test,  self.y_test, verbose=2)

	def save(self):
		tf.save_model.save(self.model, 'Model_Description') # Save model

	def plot_accuracy(self, i = 1):
		print("\nEvaluating KleinNet model accuracy & loss...")
		for history_type in ['Accuracy', 'Loss']:		# Evaluate the model accuracy and loss
			plt.plot(self.history.history[history_type.lower()], label=history_type)
			plt.plot(self.history.history['val_' + history_type.lower()], label = 'Validation ' + history_type)
			plt.xlabel('Epoch')
			plt.ylabel(history_type)
			plt.legend(loc='upper right')
			plt.ylim([0, 1])
			title = "~learnig rate: " + str(config.learning_rate) + " ~alpha: " + str(config.alpha) + ' ~bias: ' + str(config.bias) + ' ~optimizer: ' + config.optimizer
			if config.optimizer == 'SGD':
				title = title + ' ~epsilon: ' + str(config.epsilon)
			else:
				title = title + ' ~momentum: ' + str(config.momentum)
			plt.title(title)
			plt.savefig(config.result_directory + config.run_directory + "/Model_Description/Model_" + str(i + 1) + "_" + history_type + ".png")
			plt.close()

	def observe(self, interest):
		print("\nObserving " + config.outputs_category[interest].lower() + " outcome structure")
		try:
			self.images
		except:
			self.wrangle(self.subject_pool[0])
		self.sample_label = -1
		while self.sample_label != interest: # Grab next sample that is the other category
			self.sample_label = self.labels[random.randint(self.images.shape[0])] # Grab sample label
		self.sample = self.images[rand_ind, :, :, :, :] # Grab sample volume
		#self.anatomie = self.anatomies[rand_ind]
		self.header = self.headers[rand_ind]
		self.category = config.outputs[sample_label]

		print("\nExtracting " + category + " answer features from KleinNet convolutional layers...")
		self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes
		layer_outputs = [layer.output for layer in self.model.layers[:]]

		for self.layer in range(1, (config.convolution_depth*2 + 1)): # Build deconvolutional models for each layer
			self.activation_model = tf.keras.models.Model(inputs = self.model.input, outputs = [layer_outputs[self.output_layers[self.layer - 1]], self.model.output])
			self.deconv_model = tf.keras.models.Sequential() # Create first convolutional layer
			self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, config.kernel_size, strides = config.kernel_stride, input_shape = (self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], self.layer_shapes[self.layer - 1][2], 1), kernel_initializer = tf.keras.initializers.Ones()))
			for deconv_layer in range(self.layer - 1, 0, -1): # Build the depths of the deconvolution model
				if deconv_layer % 2 == 1 & deconv_layer != 1:
					self.deconv_model.add(tf.keras.layers.UpSampling3D(size = config.pool_size, data_format = 'channels_last'))
				self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, config.kernel_size, strides = config.kernel_stride, kernel_initializer = tf.keras.initializers.Ones()))
			print('Summarizing layer ', self.layer, ' deconvolution model')
			self.deconv_model.build()
			self.deconv_model.summary()
			self.feature_maps, predictions = activation_model.predict(sample) # Grab feature map using single volume
			self.feature_maps = self.feature_maps[0, :, :, : ,:].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2], self.current_shape[3])

			for self.map_index in range(self.feature_maps.shape[3]): # Save feature maps in glass brain visualization pictures
				feature_map = (self.feature_maps[:, :, :, map_index].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2])) # Grab Feature map
				deconv_feature_map = self.deconv_model.predict(self.feature_maps[:, :, :, map_index].reshape(1, self.current_shape[0], self.current_shape[1], self.current_shape[2], 1)).reshape(self.new_shape[0], self.new_shape[1], self.new_shape[2])
				self.plot_all(heatmap, 'DeConv_Feature_Maps', map_index)

			print("\n\nExtracting KleinNet model class activation maps for layer " + self.layer)
			with tf.GradientTape() as gtape: # Create CAM
				conv_output, predictions = self.activation_model(sample)
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
		title = layer + " " + data_type + " Map " + str(map_index) + " for  " + self.category + " Answer"
		output_folder = config.result_directory + config.run_directory + self.catergory + '/Layer_' + self.layer + '/' + data_type + '/' + plot_type + '/'
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
			print("Running Jack-Knife on Subject " + str(self.jackknife))
			self.wrangle(self.subject_pool, self.jackknife)
			self.build()
			self.train()
			self.plot_accuracy()
			self.ROC()

	def ROC(self):
		self.probabilities = self.model.predict(self.x_test).ravel()
		np.save(self.config.result_directory + self.config.run_directory + '/Jack_Knife/Probabilities/Sub-' + str(self.jackknife) + '_Volumes_Prob.np', self.probabilities)
		fpr, tpr, threshold = roc_curve(self.y_test, self.probabilities)
		predictions = np.argmax(self.probabilities, axis=-1)
		AUC = auc(fpr, tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr, tpr, label = 'RF (area = {:.3f})'.format(AUC))
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Subject ' + str(self.jackknife) + ' ROC Curve')
		plt.legend(loc = 'best')
		plt.savefig(self.config.result_directory + self.config.run_directory + "/Jack_Knife/Sub_" + str(self.jackknife) + "_ROC_Curve.png")
		plt.close()

	def create_dir(self, cleandir = True):
		first_dir = self.config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = ['Layer_' + str(layer) for layer in range(1, self.config.convolution_depth*2 + 1)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		try:
			os.chdir(self.config.result_directory + self.config.run_directory + '/')
			os.chdir('../..')
			print('\nRun directory ' + self.config.result_directory + self.config.run_directory + ' currently exists, a clean run directory is needed for KleinNet to output results correctly, would you like to remove and replace the current run directory? (yes or no)')
			response = 'yes'#input()
			if response == 'yes':
				shutil.rmtree(self.config.result_directory + self.config.run_directory)
				time.sleep(1)
			else:
				return
		except:
			print('\nGenerating run directory')
		os.mkdir(self.config.result_directory + self.config.run_directory + '/')
		os.mkdir(self.config.result_directory + self.config.run_directory + "/Model_Description")
		os.mkdir(self.config.result_directory + self.config.run_directory + '/SVM')
		os.mkdir(self.config.result_directory + self.config.run_directory + '/Jack_Knife')
		os.mkdir(self.config.result_directory + self.config.run_directory + '/Jack_Knife/Probabilities')
		for first in first_dir:
			os.mkdir(self.config.result_directory + self.config.run_directory + "/" + first)
			for second in second_dir:
				os.mkdir(self.config.result_directory + self.config.run_directory + "/" + first + "/" + second)
				for third in third_dir:
					os.mkdir(self.config.result_directory + self.config.run_directory + "/" + first + "/" + second + "/" + third)
					for fourth in fourth_dir:
						os.mkdir(self.config.result_directory + self.config.run_directory + "/" + first + "/" + second + "/" + third + "/" + fourth)
		print('\nResult directories generated for ' + self.config.run_directory + '\n')


class configuration:

	def __init__(self):
		self.build() # Build configuration

	def build(self):
		#-------------------------------- Shuffle Data ------------------------------##
		# Shuffle will zip all images and labels and shuffle the data before assigning
		# them to training and testing sets. Defaults to true, however you might want
		# to change this to false if you think your question has a time dimension to it
		self.shuffle = True

		#------------------------------- Run parameters ------------------------------#
		# Epoch and Batch Size - Both used to describe how the model will run, and
		# for how long. Epochs represents how many times the data will be presented to
		# the model for deep learning. The batch size simply defines how the model will
		# batch the samples into miniature training samples to help plot model performance.
		self.epochs = 100
		self.batch_size = 20

		#---------------------------- Model Hyperparameters ---------------------------#
		# Model Hyperparameters - Hyperparameters used within the models algorithms to
		# to learn about the dataset. These values were found while optimizing the model
		# over a simple stroop dataset so consider using the optiimize function within
		# the KleinNet library to find optimum values. Bias can be a bit tricky to optimize
		# and I would recommend using the KleinNet.optimum_bias() to find bias when using
		# an inbalanced dataset. Hyperparameter descriptors to be added with GUI.
		self.alpha = 0.1
		self.epsilon = 1e-6
		self.learning_rate = 0.0001
		self.bias = 0
		self.dropout = 0.3
		self.momentum = 0

		#-------------------------------- Model Set-Up -------------------------------#

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
		self.top_density = [40, 11, 10]
		self.density_dropout = [True, False, False, False]


		# Outputs - The output variables are used to help the model process and understand what it
		# is classifying and help it display some of the output better. These variables are
		# also used within KleinNet.build() to create the output layer. The output activation
		# is used to decide what activation the model will us in it's output layer.
		self.output_activation = 'sigmoid'
		self.outputs = [0, 1]
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
		self.loss = 'binary_crossentropy'

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