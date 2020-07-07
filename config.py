
## =========================== KleinNet Config Overview ==================== ##
# The config file apart of this python script is used to help streamline the  #
# the process of apply the code to new datasets. Most of the variables are    #
# already set and will work for most datasets (aside from the data descriptors#
# like x/y/z size) however they will most likely be suboptimal. Please look   #
# through this document before starting to alter any of the code to ensure    #
# you run the model as easily as possible.                                    #

#-------------------------------- Shuffle Data ------------------------------##
# Shuffle will zip all images and labels and shuffle the data before assigning
# them to training and testing sets. Defaults to true, however you might want
# to change this to false if you think your question has a time dimension to it
shuffle = True

#------------------------------- Run parameters ------------------------------#
# Epoch and Batch Size - Both used to describe how the model will run, and
# for how long. Epochs represents how many times the data will be presented to
# the model for deep learning. The batch size simply defines how the model will
# batch the samples into miniature training samples to help plot model performance.
epochs = 100
batch_size = 20

#---------------------------- Model Hyperparameters ---------------------------#
# Model Hyperparameters - Hyperparameters used within the models algorithms to
# to learn about the dataset. These values were found while optimizing the model
# over a simple stroop dataset so consider using the optiimize function within
# the KleinNet library to find optimum values. Bias can be a bit tricky to optimize
# and I would recommend using the KleinNet.optimum_bias() to find bias when using
# an inbalanced dataset. Hyperparameter descriptors to be added with GUI.
alpha = 0.1
epsilon = 1e-6
learning_rate = 0.0001
bias = 0
dropout = 0.3
momentum = 0

#-------------------------------- Model Set-Up -------------------------------#

# The Kernel initializer - is used to initialize the state of the model. We initialize
# the model (e.g. weights, biases, etc.) using Xavier (glorot) uniform which randomly selects
# values from a uniform gaussian distribution. I found this initializer to be superior to others
# at the time of building the model however you can change the initializer by typing
# in the tensorflow string code for the initializer here (e.g. 'glorot_uniform')
kernel_initializer = 'glorot_uniform'

# Convolution Depth - KleinNet is built to use a basic convolution layer structure
# that is stacked based on how deep the model is indicated her. Having the depth
# set at 2 meaning will cause the model to build 2 convolutions layers from
# the convolution template in the KleinNet.build() function before building the top density
convolution_depth = 2

# Initial Filter Count - KleinNet convolution layer filter sizes are calculated
# within the KleinNet.plan() function using a common machine learning rule of
# doubling filter count per convolution layer. init_filter_count is the initial
# value the filters starts on before doubling.
init_filter_count = 1

# Kernel Size & Stride  - These variables are used to decide what the convolution
# kernel size will be along with how it moves across the layer to generate Features.
# Generally the bigger the kernel stride, the small the output which... could be a good thing?
kernel_size = (2, 2, 2)
kernel_stride = (1, 1, 1)

# Zero Padding - Padding is used to decide if 0's will be added onto the edge of
# the input to make sure the convolutions don't move outside of the model and crash
# your script. 'valid' padding means no 0's will be added to the side were 'same'
# padding means 0's will be added to the edges of the layer input. The padding
# variable declares, if using same padding, the size of the padding to add on the the edges
zero_padding = 'valid'
padding = (0, 0, 0)

# Max Pooling - Max pooling is used to generally reduce the size of the input.
# These layers will general a pool kernel of size pool_size and move throughout the layers input
# based on pool_stride. The max pooling layer finds the max value within the kernel's
# area and pools it all into a smaller space.
pool_size = (2, 2, 2)
pool_stride = (2, 2, 2)

# Top Density Layer(s) - The following variables are used to define the structure
# of the top density. The top_density variable holds the sizes of each layer were
# the density_dropout layer defines whether there is dropout moving into that layer.
# You might notice that the density_dropout variables hold an extra value conpared
# to the top_density variable and this is to account for the flattening layer that
# is automatically built within the KleinNet.build() function. The first value of
# density_dropout[0] corresponds to dropout applied to the flatterning layer.
top_density = [40, 11, 10]
density_dropout = [True, False, False, False]


# Outputs - The output variables are used to help the model process and understand what it
# is classifying and help it display some of the output better. These variables are
# also used within KleinNet.build() to create the output layer. The output activation
# is used to decide what activation the model will us in it's output layer.
output_activation = 'sigmoid'
outputs = [0, 1]
outputs_category = ['Correct', 'Incorrect']

# Optimizers - This section is used to help switch between different Optimizers
# without having to worry about changing code too much. While talking about
# SGD/Adam and Nestrov/AMSGrad is not within the scope of this config file I
# would recommend looking up literature to find which would be best for you.
optimizers = ['SGD', 'Adam']
optimizer = 'SGD' # Set to either 'SGD' or 'Adam'
use_nestrov = True # If using SDG optimizer
use_amsgrad = True # If using Adam optimizer

# Loss - This variable describes the loss calculation used within the model.
# the standard used while initially building KleinNet was binary crossentropy
# however you may need to change this based on the questions you are asking.
loss = 'binary_crossentropy'


#------------------------- Experiment & Data Description ----------------------#
# Subject Descriptors - This section is used to describe the subject pool this data
# was aquired from. The core information is how many subject there are and how
# their subject ID is contructed in output files. This information is used in
# both KleinNet and the DataWrangler to assess what files to grab along with some
# deeper analysis (Jack knife evaluations)
subject_count = 28
ID_prefix = 'sub-'
ID_suffix = ''
ID_len = 3

# Folder Structure - These variables are used to desribes where data is stored
# along with where to store the outputs of KleinNet. The results directory is
# a general folder where specific model run folder will be created. The run
# directory is the specific folder KleinNet will generate and output results into.
# The data directory is the folder that holds all the data to be inputed into model.
# Note that this data directory should include the folder parents up to the main
# main BIDS formatted folder that holds all experiment data
project_directory = 'Stroop_MRI/'
result_directory = 'Scripts/'
run_directory = 'Run_8'
data_directory = 'BIDS/ds000164_R1.0.1-Data/'

# Numpy Output Directory & Filenames - The DataWrangler generate a folder for each subject that
# contains their numpy volume & event extracts. The filename prefixes and suffixes
# are used both in the generation of the extracts and their later wrangling for
# model training so as long as these variables stay consistent between using the
# DataWrangler and KleinNet  no problem should pop up.
numpy_output_dir = 'numpy_extracts'
volumes_filename_prefix = ''
volumes_filename_suffix = '_volumes'
affines_filename_prefix = ''
affines_filename_suffix = '_affines'
header_filename_prefix = ''
header_filename_suffix = '_headers'
labels_filename_prefix = ''
labels_filename_suffix = '_labels'
prob_filename_prefix = ''
prob_filename_suffix = '_prob'

# Scan Details - These variables hold information about the volumes aquired and
# will be used in every step of this code. The TR is primarily used for slice timing
# acquasition corrections. The n_z_slice declares the first slice aquired. X, y
# and z are all the sizes of the different dimensions of the volumes aquires.
# Lastly volumes per scan describes how many volumes are aquired per scan and
# helps the script load in data.
TR = 1.47
n_z_slice = 29
x_size = 64
y_size  = 64
z_size = 29
volumes_per_scan = 370

# Wumbo - This variables simple declares whether random variables are being used.
# Sometimes you might find turning this on when your model is doing too well to
# confirm the model is actual learning something from the data.
wumbo = False
