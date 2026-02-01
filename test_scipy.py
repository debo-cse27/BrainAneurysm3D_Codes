from scipy.io import savemat
from scipy.io import loadmat
import numpy as np
import time
# a = np.arange(20)
# mdic = {"a": a, "label": "experiment"}




# mat_data = loadmat('../Results/Aneurysm3D/11_03_2023_14_02_46.mat')
mat_data = loadmat('../Data/Aneurysm3D.mat')





# savemat("matlab_matrix.mat", mdic)


# import scipy.io

# Load the .mat file
# mat_data = scipy.io.loadmat('your_file.mat')

# Open a text file for writing
with open('data.txt', 'w') as text_file:
    # Iterate through the keys in the mat_data dictionary
    for key in mat_data:
        # Convert the data to a string and write it to the text file
        text_file.write(f'Key: {key}\n')
        text_file.write(f'Data: {mat_data[key]}\n\n')
