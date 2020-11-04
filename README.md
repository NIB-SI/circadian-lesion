# circadian-lesion

Necessary software: 
- Fiji https://imagej.net/Fiji + FFT bandpass filter https://imagej.nih.gov/ij/plugins/fft-filter.html
- OpenCV-3.1.0 and OpenCV_contrib-3.1.0: https://github.com/NIB-SI/circadian-lesion/blob/main/Installation_OpenCV3.1.0

# Algorithms									
									
## Preprocessing with Fiji

bandpass adjustment of uneven illumination (all pictures are much brighter in the middle part)									
- Fiji macro for FFT bandpass filter: filter_large=40, filter_small=3, suppress=None, tolerance=5, autoscale, saturate	
- example:

<img src="https://github.com/NIB-SI/circadian-lesion/blob/main/example/D572%20-%2020170413_204628.bmp" width="400"> <img src="https://github.com/NIB-SI/circadian-lesion/blob/main/example/D572%20-%2020170413_204628.jpg" width="400"> 

## Running .cpp files

- create test-folder at your HOME directory and copy respective .cpp file 
- rename .cpp file e. g. Rywal 7-18 dpi each 6 hours (point zero 2017_09_01 at 07.00.00).cpp to Rywal7-18.cpp (remove spaces)
- create folders results/filtered_lesions within the test-folder
- copy input images to test-folder
- create CMakeLists.txt
- run cmake .
- run make
- ./Rywal7-18 'input_folder'

- example: https://github.com/NIB-SI/circadian-lesion/tree/main/example_leaves

## Output

Terminal:
1. row file number

following rows: timepoint | Time after inoculation [min] | lesion area
red marked timepoints indicate night images

paramters of last image (possibility to insert these paramters to a .cpp file with other settings e. g. Part1/Part2/Part3 to receive optimal results, it's also possible to insert artificial lesion parameters here)
-lesion indices
-lesion areas
-mass points
-radii
-centers

## Documentation of algorithms/methods

**int main()**	

extraction of experiment information 								
- folder name									
- file names									
- extraction of timepoints + calculation of time after inoculation									

	**vector<int> measure_brightness()**:
	measurement of brightness									
	- to discriminate day and night images									

	**Mat normalization() and Mat adjusted_brightness()**:	
	adjustment of brightness									
	- to create even distribution of intensities									

	**Mat findLesions1() and Mat findLesions2()**:
	find potential lesions									
	- convert RGB images to HSV									
	- extract range of hue values (different range between day and night images)						
	- apply morphological operations (dilation, erosion) to remove noise									


find contours									
- apply contours (vector of points) to potential lesions									
- exclude contours near to the image border									
- set contour area size cut-off									
- extract geometrical features for all contours: mass center, minimum enclosing circle, radius									
									
processing of contours (to extract real lesions)									
- compare all candidates with each other via area of intersecting circles (minimum enclosing circles) --> possibility to insert manual determined lesions (necessary parameters: mass point (x, y), lesions area)
- if candidate is inside of another, add area to bigger one									
- calculate all distances between mass points to all given lesions from the previous picture									
- find most suitable candidate via smallest distance: all lesions from the previous picture should have a follower									
evaluation of new candidates:									
- if distance is too near to proofed lesions: add to proofed lesion									
- if area size is too small: just ignore									
divide same candidates:									
- for two or more lesion from the previous picture only one candidate was found (lesions grow together)									
- adjust area size with the help of previous picture areas									
calculate growth of lesion areas at night pictures									
- set area size of lesions from first night picture to areas of last day image 									
- add the difference between areas of night images to calculated one --> save parameters of last evaluated lesion (mass point (x, y), lesions area):							→ manual adjustment possible to continue with better results
draw proofed lesions and areas									
									
save results									
- images with lesion numbers and areas									
- timepoints, time after inoculation, lesion numbers and areas to text file									



