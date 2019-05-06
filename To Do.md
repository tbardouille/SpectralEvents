To Do:

In spectralevents_find.py:
	- make a new function for .m lines 117-209
		- identify and organize events from spectralEvents
	- find_localmax_method_1 
		- THIS IS DONE, BUT I'M NOTE SURE IF IT'S RIGHT
		- REDO SO THAT THE FWHM STUFF IS ITS OWN FUNCTION
			CALLED FOR FREQUENCY AND TIME SEPARATELY 
	- find_localmax_method_2
	- find_localmax_method_3

Code to replace imregionalmax:
 lm = scipy.ndimage.filters.maximum_filter( img, ... )
 msk = (img == lm)
 See https://stackoverflow.com/questions/27598103/what-is-the-difference-between-imregionalmax-of-matlab-and-scipy-ndimage-filte