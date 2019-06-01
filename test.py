import os, sys
import time
import mne
import numpy as np
import scipy.io as io
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import spectralevents_functions as tse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Top-level run script for finding spectral events in test data 
# provided by original SpectralEvents GitHub repo.

####################################################################
# Main Code
####################################################################

# Set dataset and analysis parameters
numSubj = 10

eventBand = [15,29]      # Frequency range of spectral events
fVec = np.arange(1,60+1)            # Vector of fequency values over which to calculate TFR
Fs = 600                            # Sampling rate of time-series
findMethod = 1                      # Event-finding method (1 allows for maximal overlap while 2 limits overlap in each respective suprathreshold region)
width = 7

thrFOM = 6; #Factors of Median threshold (see Shin et al. eLife 2017 for details concerning this value)

footprintFreq = 4
footprintTime = 80
threshold = 0.00
neighbourhood_size = (footprintFreq,footprintTime)

vis = True

################################
# Processing starts here
subjectIDs = np.arange(numSubj)+1

# Load data sessions/subjects from the same experimental setup so that 
# spectral event features are differentially characterized only between the 
# desired trial classification labels: in this case, detection vs. 
# non-detection
#
# Note: each .mat file contains: 
#   'prestim_raw_yes_no' - 200 trials x 600 time samples matrix of time series data
#   'YorN' - 200 trials x 1 matrix of 1s or 0s to indicate trial label
x = []
for s in subjectIDs:
    testFile = os.path.join('test_data', "".join(['prestim_humandetection_600hzMEG_subject',
        str(s), '.mat']))
    a = io.loadmat(testFile)
    x.append( a )

numTrials, numSamples = a['prestim_raw_yes_no'].shape

# Validate fVec input
Fn = Fs/2                   # Nyquist frequency
dt = 1/Fs                   # Sampling time interval
Fmin = 1/(numSamples*dt)    # Minimum resolvable frequency

if fVec[0] < Fmin: 
    sys.exit('Frequency vector includes values outside the resolvable/alias-free range.')
elif fVec[-1] > Fn: 
    sys.exit('Frequency vector includes values outside the resolvable/alias-free range.')
elif np.abs(fVec[1]-fVec[0]) < Fmin:
    sys.exit('Frequency vector includes values outside the resolvable/alias-free range.')

# Run spectral event analysis per dataset
TFR = []
specEvents = []
ctr = 1
for thisX in x:

    # Convert data to TFR 
    thisData = thisX['prestim_raw_yes_no']
    thisClassLabels = thisX['YorN']
    thisTFR, tVec, fVec = tse.spectralevents_ts2tfr( thisData.T, fVec, Fs, width )
    TFR.append( thisTFR )

    # Find local maxima in TFR
    thisSpecEvents = tse.spectralevents_find (findMethod, thrFOM, tVec, fVec, thisTFR, thisClassLabels, 
        neighbourhood_size, threshold, Fs)
    thisSpecEvents = pd.DataFrame( thisSpecEvents )
    specEvents.append( thisSpecEvents )

    # Plot results
    if vis:
        figs, classes = tse.spectralevents_vis( thisSpecEvents, thisClassLabels, thisData, thisTFR, 
            tVec, fVec, eventBand, ctr )
        
        # Save each figure made
        for figNum in np.arange(len(figs)):

            figName = os.path.join('test_data', 'results', 'python', 
                "".join(['prestim_humandetection_600hzMEG_subject', str(ctr), '_class_', 
                str(classes[figNum]), '.png']))
            figs[figNum].savefig(figName)

    plt.close()

    ctr = ctr + 1
    




