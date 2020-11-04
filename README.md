# circadian-lesion

Necessary software: 
- Fiji https://imagej.net/Fiji + FFT bandpass filter https://imagej.nih.gov/ij/plugins/fft-filter.html
- OpenCV3.1.0 and OpenCV-contrib3.1.0: https://github.com/NIB-SI/circadian-lesion/blob/main/Installation_OpenCV3.1.0

# Algorithms									
									
## Preprocessing with Fiji

bandpass adjustment of uneven illumination (all pictures are much brighter in the middle part)									
- Fiji macro for FFT bandpass filter: filter_large=40, filter_small=3, suppress=None, tolerance=5, autoscale, saturate	
- example:

<img src="https://github.com/NIB-SI/circadian-lesion/blob/main/example/D572%20-%2020170413_204628.bmp" width="400"> <img src="https://github.com/NIB-SI/circadian-lesion/blob/main/example/D572%20-%2020170413_204628.jpg" width="400"> 

## Running .cpp files

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



