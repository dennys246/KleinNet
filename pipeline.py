from __future__ import absolute_import, division, print_function, unicode_literals
import os, csv, shutil, time, random
import numpy as np
import nibabel as nib
import nibabel as nib
from glob import glob
from scipy.signal import fftconvolve
from scipy.stats import gamma

class wrangler:
	def __init__(self, config):
		self.config = config
		return
	
	def wrangle(self, subject_pool, subjects = [], count = 0, session = '*', activation = 'linear', shuffle = False, jackknife = None, exclude_trained = False, shape_extraction = False):
		# wrangle subjects into training and testing datasets
		# 
		# Define the subject count found
		subject_count = 0
		if count == 0 and len(subjects) < 3: # Check that enough subjects were passed in
			print("Not enough subjects passed into wrangle, must pass in at least 3 subjects to evenly split between test and training")
			return 
		if count == 0: # If count not set
			count = len(subjects) # Set count to number of subjects passed
		if count == -1:
			count = 1
		
		# Run through each subject and load data
		self.current_batch = subjects
		
		# Define variables to hold subject data
		images = np.array([])
		labels = np.array([])

		test_indices = []
		train_indices = []
		train_test_mod = 0

		# While we haven't reached out subject count goal (minimum of 3)
		while subject_count < count:

			# If we've run out of subjects but need more
			if subjects == []: 
				subject = self.config.subject_pool.pop(random.randint(0, len(self.config.subject_pool) - 1))
			else: # Grab the first subject in our passes subject list
				subject = subjects.pop(0)
			print(subject)

			# Check if subject is in excluded list
			if subject in self.config.excluded_subjects: 
				print(f'Excluding subject {subject} from run...')
				continue

			# If subject previously run and we're not retraining
			if exclude_trained == True and subject in self.config.previously_run: # check is subject previously run
				print(f"Subject {subject} previously run, skipping subject...")
				continue

			# If we're not 
			if subject != jackknife:
				image, label = self.load_subject(subject, session, activation, shuffle)
				if len(label) == 0: # If no images available
					print(f"Skipping {subject}, no labels...")
					continue # Skip subjects
				try: # appending the images to the image object
					images = np.append(images, image, axis = 0)
					labels = np.append(labels, label)
				except: # initialize a new image object
					images = image
					labels = label

				# Figure out if the subject is apart of training for testing
				if train_test_mod % 3 > 0: # add indices to training
					train_indices += [ind for ind in range(len(train_indices), len(train_indices) + len(image) - 1 )]
				else: # add indices to testing
					test_indices += [ind for ind in range(len(test_indices), len(test_indices) + len(image) - 1)]
				
				self.current_batch.append(subject)
				train_test_mod += 1 # increment training
			subject_count += 1
		
		# If grabbing a single subject...
		if count == 1 and shape_extraction == True:
			subjects.insert(0, subject)
			return True

		# Check if images loaded had any volumes in then
		if images.shape[0] == 0:
			return False
		
		if jackknife == None: # Split images and labels into training and test sets
			if train_indices != []:
				print(f'Max test indice {max(test_indices)}\nMax train indices {max(train_indices)}\nLength of images {images.shape}\nLength of labels: {len(labels)}\nTraining Indices: {len(train_indices)}\nTesting Indices: {len(test_indices)}')
				x_train = images[train_indices,:,:,:]
				y_train = labels[train_indices]
				x_test = images[test_indices,:,:,:]
				y_test = labels[test_indices]
			else:
				print(f'Max test indice {max(test_indices)}\nLength of labels: {len(labels)}\nTesting Indices: {len(test_indices)}')
				x_train = []
				y_train = []
				x_test = images[test_indices,:,:,:]
				y_test = labels[test_indices]
		else: # handle jackknife case of grabbing just one subject for test and using all other samples for training
			x_train = images
			y_train = labels
			x_test, self.y_test = self.load_subject(jackknife, session, activation, shuffle)
		return x_train, y_train, x_test, y_test
	
	def load_subject(self, subject, session, activation = 'linear', shuffle = False, load_affine = False, load_header = False):
		
		print(f"Loading subject {subject}")

        # Create a call for handling an empty empty exit
		def empty_exit(load_affine, load_header):
			if load_affine == False and load_header == False:
				return [], []
			if load_header == False and load_affine == True:
				return [], [], []
			if load_header == True and load_affine == False:
				return [], [], []
			if load_header == True and load_affine == True:
				return [], [], [], []
	
		# look for all relavent image file
		image_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.bold_identifier}")
		if len(image_filenames) == 0: #  if no image files found, return empty handed
			print(f'No images found for {subject} ses-{session}')
			return empty_exit(load_affine, load_header)
		else: # Grab and load image
			image_filename = image_filenames[0]
			image_file = nib.load(image_filename) # Load images
	
		# Grab images header/meta data
		header = image_file.header 
		self.config.header = header
	
		# Grab image shape and affine from header
		image_shape = header.get_data_shape() 

        # Grab data
		image = image_file.get_fdata()

		# Reshape image to have time dimension as first dimension and add channel dimension
		image = image.reshape(image_shape[3], image_shape[0], image_shape[1], image_shape[2], 1)
		if self.config.data_shape == None:
			self.config.data_shape = image.shape[1:-1]
			print(f"Data shape: {self.config.data_shape}")

        # Normalize data
		image = self.normalize(image)

        # Grab fMRI affine transformation matrix
		affine = image_file.affine 

        # Grab relavent label filenames
		label_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.label_identifier}")
		if len(label_filenames) != 1: # If we grabbed more/less than one label file, exit
			print(f"Multiple/no label files found for subject {subject}: {label_filenames}")
			empty_exit(load_affine, load_header)
		else: # Grab filename
			label_filename = label_filenames[0]
			
		print(f"Loading {label_filename}...")
		# initialize empty labels and read in labels
		labels = []
		with open(label_filename, 'r') as label_file:
            # Handle text file case
			if label_filename[-4:] == '.txt': 
				labels = label_file.readlines()
				labels = [float(label) for label in ''.join(labels).split('\n') if label != '']
            
            # Handle csv file case
			if label_filename[-4:] == '.csv':
				labels = []
				csv_reader = csv.reader(label_filename)
				for row in csv_reader:
					labels.append(float(row))

			# Handle tsv file case
			if label_filename[-4:] == '.tsv':
				tsv_reader = csv.reader(label_filename, '\t')
				for row in tsv_reader:
					labels.append(float(row))

            # Load labels into a numpy array
			labels = np.array(labels)
			
		print(f'Subject {subject} image shape: {labels.shape}')
        
        # If no labels ended up being loaded, exit
		if labels.shape[0] == 0 or labels.shape[0] != image.shape[0]:
			print(f"Labels are empty or labels length does not match image length...\n Image shape - {image.shape}\n Label shape - {labels.shape})")
			return empty_exit(load_affine, load_header)
		
		if activation != 'linear': # If not a regression problem
			labels = [int(label) for label in labels] # Convert labels to integers for classifing
        
        #  Shuffle images and labels if configured
		if self.config.shuffle == True or shuffle == True:
			image, labels = self.shuffle(image, labels)

        # Return requested subject data
		if load_affine == False and load_header == False:
			return image, labels
		if load_header == False and load_affine == True:
			return image, labels, affine
		if load_header == True and load_affine == False:
			return image, labels, header
		if load_header == True and load_affine == True:
			return image, labels, header, affine
		
	def shuffle(self, images, labels):
		# Get sample indices
		indices = np.arange(images.shape[0])
		
        # Shuffle indices
		np.random.shuffle(indices)

		# Rearrange images and labels using shuffled indices
		images = images[indices, :, :, :, :]
		
		labels = labels[indices]
		
		return images, labels
	
	def normalize(self, array):
		# Find array min
		array_min = np.min(array)

		# Find array max
		array_max = np.max(array)

		# Pass back normalized array
		return 2 * (array - array_min) / (array_max - array_min) - 1


	def mean_filter(self, image, filter_size):
		return
	
	def median_filter(self, image, filter_size):
		return

	def gaussian_filter(self, image, filter_size, sigma = 1):
		return
	
	def bilateral_filter(self, image, filter_size):
		return
	
	def laplacian_filter(self, image, filter_size, sign = 'positive'):
		return

	def dilation_filter(self, image, filter_size):
		return

	def erosion_filter(self, image, filter_size):
		return


	def create_dir(self):
		print(self.config.outputs_category)
		first_dir = self.config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = [f'Layer_{str(layer)}' for layer in range(1, self.config.convolution_depth)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		if self.config.rebuild == True:
			if os.path.exists(f'{self.config.project_directory}{self.config.model_directory}/') == True:
				print(f'\nRun directory {self.config.project_directory}{self.config.model_directory} already exists, clearing directory...')
				shutil.rmtree(f'{self.config.project_directory}{self.config.model_directory}')
				time.sleep(1)
		else: # If not resetting model
			if os.path.isdir(f'{self.config.project_directory}{self.config.model_directory}/') == True: # If model exists
				print(f"Run directory already exists for {self.config.model_directory}, if you intended to train a fresh model consider altering run directory or deleting the model saved in {self.config.model_directory} (i.e. setting config.reset_model = True)")
				return

		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/model')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/plots')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife/probabilities')
		for first in first_dir:
			os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}')
			for second in second_dir:
				os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}')
				for third in third_dir:
					os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}')
					for fourth in fourth_dir:
						os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}/{fourth}')
		print(f'\nResult directories generated for {self.config.model_directory}\n')
