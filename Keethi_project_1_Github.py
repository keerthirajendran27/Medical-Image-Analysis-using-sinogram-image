#*****************************************************************************#
# Project Name: Image Reconstruction using Sinogram.
# Author: Keerthi Rajendran.
# Resources Used: 
# > Python Programming Language.
# > Anaconda Spyder 5 Programming Environment.
# Libraries Used: Scipy, Numpy, Pil, Matplotlib.
#*****************************************************************************#
#******************* Python Libraries Used************************************#
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import numpy as np
from skimage.transform import rotate
from PIL import Image
import skimage.io as io
#***************************Open the image of Sinogram from the location******************************#

Image.open("sinogram.png") #Open an image from a file named "sinogram.png"

#*****************************Creating an array of RGB Colours****************************************#

sinogram = io.imread("sinogram.png") #Reads the image from the path and stores it as sinogram
plt.imshow(sinogram) #Plots the image that is read above
Red_chan = sinogram[:, :, 0] #Obtain the red channel from the sinogram
Green_chan = sinogram[:, :, 1] #Obtain the green channel from the sinogram
Blue_chan = sinogram[:, :, 2] #Obtain the blue channel from the sinogram

#************************Adding Lables and Plotting the Images****************************************#

fig = plt.figure() #Create a figure

ax1 = fig.add_subplot(131) #Create a subplot on the figure 1 row, 3 columns & subplot 1
ax1.title.set_color('red') #Add title color to the plot
ax1.set_title('Red_Channel') #Add title to the plot
ax1.imshow(Red_chan)  #show the plot on subplot 1

ax2 = fig.add_subplot(132)  #Create a subplot on the figure 1 row, 3 columns & subplot 2
ax2.title.set_color('green') #Add title color to the plot
ax2.set_title('Green_Channel') #Add title to the plot
ax2.imshow(Green_chan)  #show the plot on subplot 2

ax3 = fig.add_subplot(133) #Create a subplot on the figure 1 row, 3 columns & subplot 3
ax3.title.set_color('blue') #Add title color to the plot
ax3.set_title('Blue_Channel') #Add title to the plot
ax3.imshow(Blue_chan) #show the plot on subplot 3

plt.show() #Display the plot


RGB_Combined_Channels = [Red_chan,Green_chan,Blue_chan]   #Combining each channels in to one single RGB Channel

#***************************1-D Fourier Transform applied to each channel*****************************#
def Construct_Sinogram_FFTS(sinogram): 
   # ”Build 1-d FFTs of an array of projections, each projection 1 row of the array." 
    return fft.rfft(sinogram, axis=1)

#**************************Using Radon Transform for Getting Back Projection**************************#

def Back_Projection_Process(operator): #This function calculates the back projection process using a given operator
    laminogram=np.zeros((operator.shape[1],operator.shape[1])) #The laminogram is initiated as an array of zeros
    Angle_Theta = 180.0 / operator.shape[0] #Calculate the angle θ
    for i in range(operator.shape[0]): #Create an array of zeros with the same size as the operator
        rotated_matrix = np.tile(operator[i],(operator.shape[1],1)) #Create a matrix with the same values as the element of the operator
        rotated_matrix = rotate(rotated_matrix, Angle_Theta*i) #Rotate the matrix with the angle θ
        laminogram += rotated_matrix #Add the rotated matrix to the array of zeros
    return laminogram #Return the result

#*******************************Applying Ramp Filter to the Each Row using FT*************************#

def ramp_filter(ffts):
    """ Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows). """
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5)) #It creates a ramp array
    return ffts * ramp #multiply every element in ffts with ramp

#****************Converting the single channel image of RGB to Each Grey_Scale Image******************#

def Grey_Scale_Image_View(rgb):
    """ Reconstructs each channel of the image with ramp-filtering. """
    FD_PROJECTIONS = Construct_Sinogram_FFTS(rgb) #Calling the function Construct_Sinogram_FFTS 
    FFD_PROJECTIONS = ramp_filter(FD_PROJECTIONS) #Calling the function ramp_filter with input argument FFD_PROJECTIONS
    return FFD_PROJECTIONS #Returns the FFD_PROJECTIONS
    
Reconstructed_FFT_Image = [] #Starts an Empty List
for rgb in RGB_Combined_Channels: #Creating a loop for combined channels 
        Reconstructed_FFT_Image.append(Grey_Scale_Image_View(rgb)) #Calling the function   
Grey_Scale_Image = np.dstack(tuple(Reconstructed_FFT_Image)) #Creates an image array
plt.title('Grey-Scale Image', fontsize=15,color = 'Grey',fontweight='bold') #Creating a plot title
plt.imshow(Grey_Scale_Image) #Shows the plot 
plt.show() #Displays the plot

#***************************Applying inverse FFT to the Spatial Domain********************************#

def invert_fft_transform(operator):
    #" Build 1-d IRFFTs of an array of projections, each projection 1 row of the array. "
    return fft.irfft(operator, axis=1)

#***********************************Combined images in RGB View***************************************#

def RGB_Image_View(rgb):
    """ Combining the single channels and adding RGB using spatial transform"""
    Filtered_Inv_FFD = invert_fft_transform(Grey_Scale_Image_View(rgb)) #Calling the function with invert_fft_transform 
    return Filtered_Inv_FFD #Returns the Filtered_Inv_FFD 

Reconstructed_Spatial_Image = [] #Starts an Empty List
for rgb in RGB_Combined_Channels: #Creating a loop for combined channels  
        Reconstructed_Spatial_Image.append(RGB_Image_View(rgb)) #Calling the function
RGB_Image = np.dstack(tuple(Reconstructed_Spatial_Image)) #Creates an image array
plt.title('RGB Combined Image', fontsize=15, color='black', fontweight='bold') #Creating a plot title
plt.imshow(RGB_Image) #Shows the plot
plt.show() #Displays the plot

#********Cropping Previously Generated Image by removing the unwanted Background**********************#

def Image_Cropping(image, crop_width, crop_height): #Defines the Image_Cropping function with 3 parameters
    """ Cropping the image and removing the unwanted background """
    img_width = image.shape[0] #Generates the width of the image
    img_height = image.shape[1] #Generates the height of the image
    start_x = img_width//2-(crop_width//2) #Calculates the starting x coordinate of the image 
    start_y = img_height//2-(crop_height//2) #Calculates the starting y coordinate of the image
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width] #Returns the cropped image from the calculated coordinates

#*********************Applying Ramp Filter to the Above Generated Image*******************************#

def Ramp_Filtered_Image(rgb): #function to modify the image with ramp-filtering
    """ Applies ramp-filtering for the Previously Cropped Image. """
    Apply_Back_Proj = Back_Projection_Process(RGB_Image_View(rgb)) #applying back projection process to the RGB image view
    Filtered_Channel_Crop = Image_Cropping(Apply_Back_Proj, 450, 450) #cropping the filtered channel to the specified size
    return (255 * (Filtered_Channel_Crop - np.min(Filtered_Channel_Crop)) 
             / np.ptp(Filtered_Channel_Crop)).astype('uint8') #return the updated image in uint8 (8-Bit) format

Ramp_Applied_Image = [] #Starts an Empty List
for rgb in RGB_Combined_Channels: #Creating a loop for combined channels 
        Ramp_Applied_Image.append(Ramp_Filtered_Image(rgb)) #Calling the function
       
Filtered_Image = np.dstack(tuple(Ramp_Applied_Image)) #Creates an image array
plt.title('Ramp filter Applied Image',fontsize=12, color='k', fontweight='bold' ) #Creating a plot title
plt.imshow(Filtered_Image) #Shows the plot
plt.show() #Displays the plot

#*************************Creating a function for applying Hamming Fliter*****************************#

def hamming_filter(FFTS):  
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    Ramp_Filtered = np.floor(np.arange(0.5, FFTS.shape[1]//2 + 0.1, 0.5)) #Creating Ramp Filter Values Again 
    c = 0.54 
    a = np.pi*Ramp_Filtered/(len(Ramp_Filtered)/2) #Calculating the angle
    hamming_filter_function = c + (1-c)*np.cos(a) #Calculate the Hamming filter
    filtered_hamming = hamming_filter_function * Ramp_Filtered #Multiplying Hamming Filter with Ramp
    return FFTS * filtered_hamming #Returns the Filtered 2-D array

#****************************Creating a function for applying Hanning Fliter**************************#

def hanning_filter(FFTS):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    Ramp_Filtered = np.floor(np.arange(0.5, FFTS.shape[1]//2 + 0.1, 0.5)) #Creating Ramp Filter Values Again 
    c = 0.5
    a = np.pi*Ramp_Filtered/(len(Ramp_Filtered)/2) #Calculating the angle 
    hanning_filter_function = c + (1-c)*np.cos(a) #Calculate the Hanning filter
    filtered_hamming = hanning_filter_function * Ramp_Filtered #Multiplying Hanning Filter with Ramp
    return FFTS*filtered_hamming #Returns the Filtered 2-D array

#*****************************Applying Hamming Filter to the Previously Ramp-Filtered Image***********#

def Hamming_Windowed_Image(rgb): #function to modify the previously applied ramp-Filtered image with Hamming Windowing
    """Hamming Windowed ramp-filtering is applied to the Ramp-Filtered Image."""
    FD_projections = Construct_Sinogram_FFTS(rgb) #calling the function Construct_Sinogram_FFTS
    Hamming_F_Transformation = hamming_filter(FD_projections) #calling the function hamming_filter
    Hamming_S_Transformation = invert_fft_transform(Hamming_F_Transformation) #calling the function invert_fft_transform
    Apply_Hamming = Back_Projection_Process(Hamming_S_Transformation) #applying back projection process to the Hamming_S_Transformation
    Hamming_Crop = Image_Cropping(Apply_Hamming, 450, 450) #cropping the filtered channel to the specified size
    return (255 * ( Hamming_Crop - np.min( Hamming_Crop))
             / np.ptp( Hamming_Crop)).astype('uint8') #return the updated image in uint8 (8-Bit) format

Generated_Hamming_Windowing = [] #Starts an Empty List
for rgb in RGB_Combined_Channels: #Creating a loop for combined channels
        Generated_Hamming_Windowing.append(Hamming_Windowed_Image(rgb)) #Calling the function
       
Hamming_Filtered_Image = np.dstack(tuple(Generated_Hamming_Windowing)) #Creates an image array
plt.title(' Hamming Windowed Ramp-filtered Image',fontsize=10, color='m', fontweight='bold') #Creating a plot title
plt.imshow(Hamming_Filtered_Image) #Shows the plot
plt.show() #Displays the plot

#*****************************Applying Hanning Filter to the Previously Ramp-Filtered Image**********#


def Hanning_Windowed_Image(rgb): #function to modify the previously applied ramp-Filtered image with Hanning Windowing
    """ Reconstructs each channel of the image with Hanning Windowed ramp-filtering. """
    FD_projections = Construct_Sinogram_FFTS(rgb) #calling the function Construct_Sinogram_FFTS
    Hanning_F_Transformation = hanning_filter(FD_projections) #calling the function hanning_filter
    Hanning_S_Transformation = invert_fft_transform(Hanning_F_Transformation) #calling the function invert_fft_transform
    Apply_Hanning = Back_Projection_Process(Hanning_S_Transformation) #applying back projection process to the Hanning_S_Transformation
    Hanning_Crop = Image_Cropping(Apply_Hanning, 450, 450) #cropping the filtered channel to the specified size
    return (255 * (Hanning_Crop - np.min(Hanning_Crop))
             / np.ptp(Hanning_Crop)).astype('uint8') #return the updated image in uint8 (8-Bit) format

Generated_Hanning_Windowing = []  #Starts an Empty List
for rgb in RGB_Combined_Channels: #Creating a loop for combined channels 
        Generated_Hanning_Windowing.append(Hanning_Windowed_Image(rgb)) #Calling the function
       
Hanning_Filtered_Image = np.dstack(tuple(Generated_Hanning_Windowing)) #Creates an image array
plt.title('Hanning Windowed Ramp-filtered Image',fontsize=10, color='c', fontweight='bold') #Creating a plot title
plt.imshow(Hanning_Filtered_Image) #Shows the plot
plt.show() #Displays the plot

#******************************************* END OF THE PROGRAM *************************************#
