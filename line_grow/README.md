# Line_growing

This is a algorithm developed for detecting weld joints using image processing techniques without any prior information about the weld joint shape or weld conditions such as welding environment, welding surface texture etc. It uses a line growing algorithm to detect the weld joint. The basic line growing algorithm is explained in detail [here](https://www.sciencedirect.com/science/article/abs/pii/S0736584513000896)

## Development Environment
- __Ubuntu 16.04.2__
- __python 2.7__
- __OpenCv 3.3.1__

##  Python dependencies

- numpy 
- cv2 
- image_geometry
- argparse
- time
- skimage
- datetime

__Step 1: Install the dependencies__

Install the mentioned python dependencies.


__Step 2: Save the images__

Save the image of the weld joint

__Step 3: Run the python program__

Run the python programm butt_weld.py.

It will promt to enter the name of the image. 

The final results will be saved to the output folder.

Set show_image flag in every function as '1' to display the images after each step. 

<p align="center">    
<img src="output/trimm_2020-08-29_10.57.25.523629.png" align="center" width="70%" height="70%">
</p>
