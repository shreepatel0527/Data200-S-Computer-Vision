from data_utils import *
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from feature_utils import get_sobel_features, get_gabor_features, generate_gabor_kernel, get_local_binary_pattern
from functions import *
import random
from sklearn.preprocessing import OneHotEncoder

# load in the dataset 

disaster_list = ["hurricane-matthew", "socal-fire", "midwest-flooding"]

data = {}
split = "train"

with open('config.json') as config_file:
    config = json.load(config_file)
    # data_dir = "/home/jovyan/shared/course/data100-shared-readwrite/fa24_grad_project_data/satellite-image-data"
    data_dir = "satellite-image-data"

for disaster in disaster_list:
    print(f"Loading {split} images and labels for {disaster} dataset...")
    images = get_images(data_dir, disaster, split=split)
    labels = get_labels(data_dir, disaster, split=split)
    data[disaster] = {"images": images, "labels": labels}

# choose hurricane matthew from our labels list 
disaster = disaster_list[0]

#separate the labels and the images for our EDA
hurricane_matthew_labels = data[disaster]["labels"]
hurricane_matthew_images = data[disaster]["images"]

# choose socal-fires from our labels list 
disaster = disaster_list[1]

#separate the labels and the images for our EDA
socal_fires_labels = data[disaster]["labels"]
socal_fires_images = data[disaster]["images"]

# choose midwest-floods from our labels list 
disaster = disaster_list[2]

#separate the labels and the images for our EDA
midwest_floods_labels = data[disaster]["labels"]
midwest_floods_images = data[disaster]["images"]

#Sobel processing both image sets 
sobel_fires_edges = img_to_sobel(socal_fires_images)
sobel_floods_edges = img_to_sobel(midwest_floods_images)

#Processing both image sets to LBP 
socal_fires_lbp = img_to_LBP(socal_fires_images)
midwest_floods_lbp = img_to_LBP(midwest_floods_images)

#Converting Midwest and SoCal image sets to RGB lists
red_socal, green_socal, blue_socal = image_to_RGB(socal_fires_images)
red_midwest, green_midwest, blue_midwest = image_to_RGB(midwest_floods_images)

red_socal_log = np.log(np.array(red_socal))
red_midwest_log = np.log(np.array(red_midwest)) 
green_socal_log = np.log(np.array(green_socal)) 
green_midwest_log = np.log(np.array(green_midwest)) 
blue_socal_log = np.log(np.array(blue_socal)) 
blue_midwest_log = np.log(np.array(red_socal)) 

gabor_socal = img_to_gabor(socal_fires_images)
gabor_midwest = img_to_gabor(midwest_floods_images)

#Ensure replicable data
random.seed(42)

print(len(socal_fires_images))

# Sample 7004 images from socal_fires_images to match the size of midwest_floods_images
random_indices = random.sample(range(len(socal_fires_images)), 7004)

print(len(random_indices))

#turn all lists into NumPy arrays
sobel_fires_edges = np.array(sobel_fires_edges)
socal_fires_lbp = np.array(socal_fires_lbp)
red_socal_log = np.array(red_socal_log)
green_socal_log = np.array(green_socal_log)
gabor_socal = np.array(gabor_socal)

# Adjust socal fires rows to match the size of midwest floods
sobel_fires_edges = [sobel_fires_edges[i] for i in random_indices]
socal_fires_lbp = socal_fires_lbp[random_indices]
red_socal_log = red_socal_log[random_indices]
green_socal_log = green_socal_log[random_indices]
gabor_socal = [gabor_socal[i] for i in random_indices]

features = {
    'Sobel_Edges': sobel_fires_edges + sobel_floods_edges,
    'LBP': list(socal_fires_lbp) + list(midwest_floods_lbp),
    'Red_Log': list(red_socal_log) + list(red_midwest_log),
    'Green_Log': list(green_socal_log) + list(green_midwest_log),
    'Gabor': gabor_socal + gabor_midwest,
    'Label': [1] * 7004 + [0] * 7004  # Adjusted label list for balanced dataset
}

# Generate features df for logistic model
logistic_features = pd.DataFrame(features)

print(logistic_features.head())
print(f"DataFrame shape: {logistic_features.shape}")

logistic_features.to_csv("socal_fires_midwest_floods.csv", header = True, index = False)

# smallest number of images for a given label is 1544

# our random seed has been generated above, no need to repeat

# to generate the entire hurricane dataset, set the label count for the longer label categories

# also avoids changing most of the code 

label_0_count = len(list(np.where(hurricane_matthew_labels == 0)[0]))
label_1_count = len(list(np.where(hurricane_matthew_labels == 1)[0]))
label_3_count = len(list(np.where(hurricane_matthew_labels == 3)[0]))

random_0_indeces = random.sample(list(np.where(hurricane_matthew_labels == 0)[0]), label_0_count)
random_1_indeces = random.sample(list(np.where(hurricane_matthew_labels == 1)[0]), label_1_count)
random_3_indeces = random.sample(list(np.where(hurricane_matthew_labels == 3)[0]), label_3_count)
random_2_indeces = random.sample(list(np.where(hurricane_matthew_labels == 2)[0]), 1544)

sum_indeces = random_0_indeces + random_1_indeces + random_3_indeces + random_2_indeces

#sanity check: len of all the indeces I want to use should be 1544 * 4
print(f"Total number of unique indices: {len(set(sum_indeces))}")

# let us extract the images from these specific indeces and generate our features 
filtered_images = [hurricane_matthew_images[index] for index in sum_indeces]
filtered_labels = [hurricane_matthew_labels[i] for i in sum_indeces]

hurricane_sobel_edges = np.array(img_to_sobel(filtered_images))
hurricane_lbp = np.array(img_to_LBP(filtered_images))
red_hurricane, green_hurricane, blue_hurricane = image_to_RGB(filtered_images)
hurricane_red_log = np.array(np.log(np.array(red_hurricane)))
hurricane_green_log = np.array(np.log(np.array(green_hurricane)))
hurricane_blue_log = np.array(np.log(np.array(blue_hurricane)))
hurricane_gabor = np.array(img_to_gabor(filtered_images))

# build our DataFrame

features= {
    "Sobel_Edges": hurricane_sobel_edges,
    "LBP": hurricane_lbp,
    "Red_Log": hurricane_red_log,
    "Green_Log": hurricane_green_log,
    # "Blue_Log": hurricane_blue_log,
    "Gabor": hurricane_gabor,
    "Label": filtered_labels
}
    
hurricane_features = pd.DataFrame(features)
print(hurricane_features.head())
print(f"DataFrame shape: {hurricane_features.shape}")
hurricane_features.to_csv("hurricane_unbalanced_labels.csv", index = False, header = True)

# have one hot encoding 

enc = OneHotEncoder()
merger = hurricane_features.copy()
new_data = enc.fit_transform(hurricane_features["Label"].values.reshape(-1, 1)).toarray()
output = pd.DataFrame(new_data, columns=enc.get_feature_names_out(["Label"]))
final = merger.join(output)
final.head()

final.to_csv("hurricane_unbal_ohe.csv", index = False, header = True)

print(f"Length of full hurricane dataset: {len(hurricane_matthew_labels)}")