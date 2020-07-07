from __future__ import absolute_import, division, print_function, unicode_literals
import reader, os, time, csv, random, nilearn, config, json, scipy, shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, plotting
from nilearn.masking import compute_epi_mask
from random import randint, randrange
from scipy.interpolate import InterpolatedUnivariateSpline as Interp

class DataWrangler:

	def __init__(self):
		print("\nData Wrangler Initialized\nProcess PID - " + str(os.getpid()) + '\n')
		os.chdir('../..')
		print('Current working directory change to ' + os.getcwd() + '\n')

	def run(self):
		self.orient()
		self.wrangle()

	def orient(self):
		self.subject_IDs = [config.ID_prefix + ("0"*(config.ID_len - len(str(sub)))) + str(sub) + config.ID_suffix for sub in range(1, config.subject_count + 1)]
		self.labels_filenames = [subject_ID + "_task-stroop_events.tsv" for subject_ID in self.subject_IDs] # .tsv
		self.volumes_filenames = [subject_ID + "_task-stroop_bold.nii" for subject_ID in self.subject_IDs] # .nii
		self.volumes_folders = [config.data_directory  + subject_ID + '/func/' for subject_ID in self.subject_IDs]
		self.anat_filenames = [subject_ID + "_T1w.nii" for subject_ID in self.subject_IDs] # .tsv
		self.anat_folders = [config.data_directory + subject_ID + '/anat/' for subject_ID in self.subject_IDs]
		print("\nFile naming schemas and paths generated for labels, volumes and anatomies\n")

	def wrangle(self):
		self.progressbar(0, len(self.labels_filenames), prefix = 'Wrangling Nifti Files', suffix = 'Complete', length = 40)
		for index, label_filename, volumes_filename, volumes_folder, self.subject_ID in zip(range(len(self.labels_filenames)), self.labels_filenames, self.volumes_filenames, self.volumes_folders, self.subject_IDs):

			image_file = nib.load(volumes_folder + volumes_filename) # Load fMRI image
			self.header = image_file.header # Grab fMRI image header
			self.affine = image_file.affine # Grab fMRI affine transformation matrix
			self.volumes = image_file.get_fdata().reshape(config.volumes_per_scan, config.x_size, config.y_size, config.z_size)
			self.labels = np.zeros(config.volumes_per_scan) # Load labels to create a map
			with open(volumes_folder + label_filename) as tsvfile:
				tsvreader = csv.reader(tsvfile, delimiter="\t")
				next(tsvreader)
				for line in tsvreader:
					if line[2] == 'Y': # If they answered correctly
						self.labels[round(float(line[0])/config.TR)] = 1 # Use onset to find the correct volume for the label
					elif line[2] == 'N':# If they answered incorrectly
						self.labels[round(float(line[0])/config.TR)] = 2

			removal_index = 0
			for label in self.labels: # Find valuable volumes of model
				if label == 0: # remove slightly-resting state data
					self.labels = np.delete(self.labels, removal_index)
					self.volumes = np.delete(self.volumes, removal_index, axis = 0)
				else: # convert 1's and 2's to 0's and 1's respetively
					self.labels[removal_index] = label - 1
					removal_index += 1
					self.affines = np.asarray(self.affine)
					self.affines.reshape((1, 4, 4))
					self.headers = np.asarray(self.header)
			self.volumes = self.volumes.reshape(removal_index, config.x_size, config.y_size, config.z_size, 1) # Add channel layer
			try:
				os.mkdir(config.data_directory + self.subject_ID + '/' + config.numpy_output_dir)
			except:
				shutil.rmtree(config.data_directory + self.subject_ID + '/' + config.numpy_output_dir)
				time.sleep(1)
				os.mkdir(config.data_directory + self.subject_ID + '/' + config.numpy_output_dir)
			self.trim_n_wig()
			data = [self.volumes, self.labels, self.affines, self.headers]
			filenames = [config.volumes_filename_prefix + self.subject_ID + config.volumes_filename_suffix, config.labels_filename_prefix + self.subject_ID + config.labels_filename_suffix, config.affines_filename_prefix + self.subject_ID + config.affines_filename_suffix, config.header_filename_prefix + self.subject_ID + config.header_filename_suffix]
			for datum, filename in zip(data, filenames):
				np.save(config.data_directory + self.subject_ID + '/' + config.numpy_output_dir + '/' + filename, datum, allow_pickle = True)
			self.progressbar(index, len(self.labels_filenames), prefix = 'Wrangling Nifti Files', suffix = 'Complete', length = 60) # Update progress bar
		print('Scan data uploaded into your data wrangler successfully')



	def strip_skull(self):
		extactor = Extractor() # generate a deepbrain extractor
		self.prob = extractor.run(self.volumes) # Generate a volume of probabilities of the value to be tissue
		mask = prob > 0.5 # Generates a mask of all voxels that a likely brain tissue
		for vox_z in range(config.z_size):
			for vox_x in range(config.x_size):
				right_prob, left_prob, left_vox_y = 0
				right_vox_y = config.y_size - 1
				while self.prob[vox_x, right_vox_y, vox_z] < 0.5 & self.prob[vox_x, left_vox_y, vox_z] < 0.5: # While right and left pointers are most likely not skull
					self.volumes[:, vox_x, left_vox_y, vox_z] = 0
					self.volumes[:, vox_x, right_vox_y, vox_z] = 0
					if right_vox_y == left_vox_y + 1 | right_vox_y == left_vox_y: # If the left and right y voxels are next to each other
						break # Go to next x voxel strip
					else: # If y voxels are still far away
						right_vox_y -= 1 # Incriment indexes
						left_vox_y += 1
		print('Skull stripping complete')

	def STC(self): # Slice timing corrections
		timing_for_single_slice = config.TR / config.n_z_slice
		for vox_z in range(config.z_size): # Changed from 29 to 28, might break
			for vox_x in range(config.x_size):
				for vox_y in range(config.y_size):
					time_course_slice_0 = self.volumes[vox_x, vox_y, 0, :] # Grab time course of some slices
					time_course_slice_current = self.volumes[vox_x, vox_y, vox_z, :]

					vol_nos = np.arange(self.volumes.shape[-1]) # Time of acquisition of the vox for slice 0 are at beginning of TR
					vol_onset_times = vol_nos * config.TR

					times_slices_0 = vol_onset_times
					times_slices_current = vol_onset_times + config.TR / (48 / (vox_z + 1)) # Time of acquisition of slice 1 are half TR later

					lin_interper = Interp(times_slices_current, time_course_slice_current, k = 1)
					interped_vals = lin_interper(times_slices_0)
					self.volumes[vox_x, vox_y, vox_z, :] = interped_vals

	def oversample(self, oversample_label, oversample_doubling):
		for x in range(oversample_doubling - 1):
			for i, label in enumerate(self.labels):
				if label == oversample_label:
					self.volumes = np.append(self.volumes[:, :, :, :, :], self.volumes[i, :, :, :, :].reshape(1, config.x_size, config.y_size, config.z_size, 1), axis = 0)
					self.labels = np.append(self.labels, self.labels[i])
					self.headers = np.append(self.headers, self.headers[i])
		self.count()

	def trim_n_wig(self):
		correct_counts = []
		incorrect_counts = []
		self.count()
		while self.correct > self.incorrect:
			rand_vol = random.randint(0, (self.volumes.shape[0] - 1))# Grab a random volume
			if self.labels[rand_vol] == 0: # If label is correct
				self.volumes= np.delete(self.volumes, rand_vol, axis = 0)
				self.labels = np.delete(self.labels, rand_vol)
				self.correct -= 1
			else:
				self.volumes = np.append(self.volumes[:, :, :, :, :], self.volumes[rand_vol, :, :, :, :].reshape(1, config.x_size, config.y_size, config.z_size, 1), axis = 0)
				self.labels = np.append(self.labels, self.labels[rand_vol])
				self.incorrect += 1
			correct_counts.append(self.correct)
			incorrect_counts.append(self.incorrect)
		x_axis = range(len(correct_counts))
		plot = plt.figure()
		ax = plot.add_subplot(211)
		line1 = ax.plot(correct_counts, 'bo-', label = 'Correct count')
		line2 = ax.plot(incorrect_counts, 'ro-', label = 'Incorrect count')
		plt.title("Trim 'n Wig")
		plt.xlabel("Trim 'n Wig Index")
		plt.ylabel("Count")
		plot.savefig(config.data_directory + self.subject_ID + '/' + config.numpy_output_dir + '/' + 'TNW_results.png')
		plt.close()

	def normalize(self):
		self.volumes = (self.volumes - self.minval) / (self.maxval - self.minval)

	def shuffle(self):
		deck = list(zip(self.volumes, self.labels, self.affines, self.headers)) # Zip and create list of data
		for i in range(10):
			random.shuffle(deck) # Shuffle data a lot
		self.volumes, self.labels, self.affines, self.headers = zip(*deck) # Unzip data

	def count(self):
		self.correct = 0
		self.incorrect = 0
		for label in self.labels:
			if label == 0:
				self.correct += 1
			else:
				self.incorrect += 1
		config.correct = self.correct
		config.incorrect = self.incorrect


	def progressbar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 40, fill = '█', printEnd = "\r"):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
		# Print New Line on Complete
		if iteration == total:
		    print()

	# Pre-process

	#	1. MotionCorrection?
	#	2. Slice-TimingCorrection
	#	3. B0DistortionCorrection
	#	4. SpatialNormalization
	#	5. SpatialSmoothing
