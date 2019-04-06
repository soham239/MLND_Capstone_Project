import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import h5py
import os


# Function to loop through images folders and rename the images from 0 to N-1 for ease of access later
def rename_images_by_order():
    for dirname in os.listdir("."):
        if os.path.isdir(dirname):
            for i, filename in enumerate(os.listdir(dirname)):
                os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".png")
            
            

# Function to generate training patches from images dataset

# Inputs
# folder_path, number_of_images, patch_size, stride, output_filename
# Output
# creation of the training dataset in the folder

def generate_training_dataset(folder_path, number_of_images, patch_size, stride, output_filename):
    
    train_images=[]
    total_sub_images=0
    for i in range(number_of_images):

        if i%10==0:
            print("{} images converted".format(i))
        
        img=cv2.imread(folder_path +str(i) +".png")  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # scaling the inputs
        
        img=img/255
        
        
        sub_image_count=0
        j=0
        while (j+patch_size <= img.shape[0]):
            k=0
            while (k+patch_size <= img.shape[1]):
                train_images.append(img[j:j+patch_size, k:k+patch_size, :])
                sub_image_count+=1
                k+=stride
            j+=stride
            
        total_sub_images+=sub_image_count
        
    train_images=np.array(train_images)
        
        
#         if i==0:
#             h5f = h5py.File(output_filename + '.h5', 'w')
#             h5f.create_dataset('data', data=train_images, chunks=True, maxshape=(None,patch_size,patch_size,3))
#             h5f.close()
#         else:
#             with h5py.File(output_filename + '.h5', 'a') as hf:                               
#                 hf['data'].resize((hf["data"].shape[0] + train_images.shape[0]), axis = 0)
#                 hf["data"][-train_images.shape[0]:] = np.array(train_images)
      
    print("Total {} images created.".format(total_sub_images))
    return train_images
    
    
    
# Input must be hdf5 file   
def read_training_dataset(file_name):

    h5f = h5py.File(file_name,'r')
    train_images = h5f['data'][:]
    h5f.close()
    
    return train_images



