
epochs = 1
batch_size = 20

alpha = 0.1
epsilon = 1e-6
learning_rate = 0.001
bias = 0
dropout = 0.3
momentum = 0

kernel_initializer = 'glorot_uniform'

convolution_depth = 2
init_filter_count = 16
kernel_size = (2, 2, 2)
kernel_stride = (1, 1, 1)

zero_padding = 'valid'
padding = (0, 0, 0)

pool_size = (2, 2, 2)
pool_stride = (2, 2, 2)

top_density = [200, 57, 55]
density_dropout = [True, False, False, False]

outputs = [0, 1]
outputs_category = ['Correct', 'Incorrect']

optimizers = ['SGD', 'Adam']
optimizer = 'SGD' # Set to either 'SGD' or 'Adam'
use_amsgrad = True # If using Adam optimizer
use_nestrov = True # If using SDG optimizer

project_directory = 'Stroop_MRI/'
result_directory = 'Scripts/'
run_directory = 'Run_6'
data_directory = 'BIDS/ds000164_R1.0.1-Data/'

subject_count = 28
ID_prefix = 'sub-'
ID_suffix = ''
ID_len = 3

numpy_output_dir = 'numpy_extracts'
volumes_filename_prefix = ''
volumes_filename_suffix = '_volumes'
affines_filename_prefix = ''
affines_filename_suffix = '_affines'
header_filename_prefix = ''
header_filename_suffix = '_headers'
labels_filename_prefix = ''
labels_filename_suffix = '_labels'
prob_filenames_prefix = ''
prob_filename_suffix = '_prob'

configured = False

TR = 1.47
n_z_slice = 29
x_size = 64
y_size  = 64
z_size = 29

volumes_per_scan = 370
volumes = 5129
correct = 0
incorrect = 0

wumbo = False # Used to set if using random labels to make sure model isn't learning nothing

description = ''
