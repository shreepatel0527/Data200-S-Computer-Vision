# Leveraging Computer Vision for Real-Time Natural Diasaster Analysis 

This repository contains satellite images, helper files, and test files to build a models that can classify natural disaster images based on the type of natural disaster and the level of damage. 

### Local SetUp

To set up your local environment, run `make environment` from your command line. This will load the necessary libraries and modules with Python 3.10. 

Run `unzip satellite-image-data.zip` form the command line to extract the images on your local machine before running the notebooks. The code needs to access the images to load them in directly to the CNN. 

**For portability, let us avoid pushing the images folder onto the repo but keep a local copy!**

### Files Contained

The `data_utils.py` contains functions to load the dataset and visualize it. The `feature_utils.py` contains a handful of functions to define useful features from the loaded images. The `functions.py` contains functions that we have written for our feature engineering. The `feature_engineer.py` contains the functions we used to clean the dataset. At the end of the data cleaning, csv files of the dataframes are generated to pss into the modeling notebooks smoothly. The `config.json` was provided by the Data 200 staff, and it used to load in the data. 

### Modeling Notebooks 

The modeling work is split up into two notebooks; one for each task. `task_A_model.ipynb` contains all code for prediction function definitions, plotting, and CNN for the fires and floods dataset.

The prediction classes and functions are the first cell in each subsection for easy access. 