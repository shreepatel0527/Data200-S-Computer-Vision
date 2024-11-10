from feature_utils import get_sobel_features, get_gabor_features, generate_gabor_kernel, get_local_binary_pattern
import numpy as np 

def img_to_imgsize(images_list):
    '''
    A function that will convert the list of images to a list of image sizes (number of pixels)

    Parameters
    ---------
    images_list : list
    A list of np arrays of np arrays, each np array contains an np array of thruples, which is an RGB pixel.

    Returns
    -------
    lst : NumPy array
    An array of pixel counts for each image.  
    '''
    
    lst = []
    for image in images_list:
        height = image.shape[0]
        width = image.shape[1]
        lst.append(width * height)
    return np.array(lst)

def img_to_sobel(image_list):
    '''
    This function processes a list of images using the Sobel operation, normalized by the size of the image. 

    Parameters
    ---------

    image_list: list
        List of images to be processed

    Returns
    ------

    edges_list: list
        List of Sobel edge intensity after processing

    '''
    
    edges_list = []
    
    for image in image_list:
        size = img_to_imgsize(image)[0]
        edges = np.sum(get_sobel_features(image)) / size
        edges_list.append(edges)
    return edges_list

def img_to_LBP(image_list):
    '''
    This function takes a list of images and processes the images using local binary pattern.

    Parameters
    ---------
    image_list: list
        List of images to be processed

    Returns
    ------
    lbp_values: list
        List of normalized LBP values (summed for each image) after LBP processing
    '''
    
    lbp_values = []
    
    for image in image_list:
        size = img_to_imgsize([image])[0]
        lbp = get_local_binary_pattern(image, radius=3)
        lbp_sum = np.sum(lbp)
        normalized_lbp = lbp_sum / size
        lbp_values.append(normalized_lbp)

    return lbp_values

def image_to_RGB(image_list):
    '''
    Converts an image to average R, G, B, value, normalized by the number of pixels

    Parameters
    ----------
    
    image_list: list
        List of images

    Returns
    ------

    red_total: list
        List with the average R value for each image, divided by the number of pixels
    
    green_total: list
        List with the average G value for each image, divided by the number of pixels
        
    blue_total: list
        List with the average B value for each image, divided by the number of pixels
    '''


    red_total = []
    green_total = []
    blue_total = []

    for image in image_list:
        size = img_to_imgsize(image)
        red = (np.sum(image[:, :, 0]) / size)[0]
        red_total.append(red)
        green = (np.sum(image[:, :, 1]) / size)[0]
        green_total.append(green)
        blue = (np.sum(image[:, :, 2]) / size)[0]
        blue_total.append(blue)

    return red_total, green_total, blue_total

def img_to_gabor(image_list, theta=0, sigma=1.0, frequency=0.1):
    '''
    This function processes a list of images using the Gabor filter, normalized by the size of the image.

    Parameters
    ---------
    image_list: list
        List of images to be processed.
    theta: float, optional
        The orientation of the Gabor filter. Default is 0.
    sigma: float, optional
        The standard deviation of the Gaussian function used in the Gabor filter. Default is 1.0.
    frequency: float, optional
        The frequency of the sinusoidal wave. Default is 0.1.

    Returns
    ------
    gabor_list: list
        List of Gabor filter responses (intensity) after processing.
    '''
    
    gabor_list = []
    
    for image in image_list:
        size = img_to_imgsize(image)[0]
        kernel = generate_gabor_kernel(theta, sigma, frequency)
        gabor_response = np.sum(get_gabor_features(image, kernel)) / size
        gabor_list.append(gabor_response)
    
    return gabor_list