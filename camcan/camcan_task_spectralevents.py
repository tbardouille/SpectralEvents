#!/usr/bin/env python

# Import libraries
import os
import sys
import time
import mne
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import multiprocessing as mp
import warnings

logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

def spectralevents_ts2tfr (S,fVec,Fs,width):
    # spectralevents_ts2tfr(S,fVec,Fs,width);
    #
    # Calculates the TFR (in spectral power) of a time-series waveform by 
    # convolving in the time-domain with a Morlet wavelet.                            
    #
    # Input
    # -----
    # S    : signals = time x Trials      
    # fVec    : frequencies over which to calculate TF energy        
    # Fs   : sampling frequency
    # width: number of cycles in wavelet (> 5 advisable)  
    #
    # Output
    # ------
    # t    : time
    # f    : frequency
    # B    : phase-locking factor = frequency x time
    #     
    # Adapted from Ole Jensen's traces2TFR in the 4Dtools toolbox.
    #
    # See also SPECTRALEVENTS, SPECTRALEVENTS_FIND, SPECTRALEVENTS_VIS.

    S = S.T
    numTrials = S.shape[0]
    numSamples = S.shape[1]
    numFrequencies = len(fVec)

    tVec = np.arange(numSamples)/Fs

    TFR = []
    # Trial Loop
    for i in np.arange(numTrials):
        B = np.zeros((numFrequencies, numSamples))
        # Frequency loop
        for j in np.arange(numFrequencies):
            B[j,:] = energyvec(fVec[j], signal.detrend(S[i,:]), Fs, width)
        TFR.append(B)

    TFR = np.asarray(TFR)

    return TFR, tVec, fVec

def energyvec(f,s,Fs,width):
    # Return a vector containing the energy as a
    # function of time for frequency f. The energy
    # is calculated using Morlet's wavelets. 
    # s : signal
    # Fs: sampling frequency
    # width : width of Morlet wavelet (>= 5 suggested).
    
    dt = 1/Fs
    sf = f/width
    st = 1/(2 * np.pi * sf)

    t= np.arange(-3.5*st, 3.5*st, dt)
    m = morlet(f, t, width)

    y = np.convolve(s, m)
    y = 2 * (dt * np.abs(y))**2
    lowerLimit = int(np.ceil(len(m)/2))
    upperLimit = int(len(y)-np.floor(len(m)/2)+1)
    y = y[lowerLimit:upperLimit]

    return y

def morlet(f,t,width):
    # Morlet's wavelet for frequency f and time t. 
    # The wavelet will be normalized so the total energy is 1.
    # width defines the ``width'' of the wavelet. 
    # A value >= 5 is suggested.
    #
    # Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)

    sf = f/width
    st = 1/(2 * np.pi * sf)
    A = 1/(st * np.sqrt(2 * np.pi))
    y = A * np.exp(-t**2 / (2 * st**2)) * np.exp(1j * 2 * np.pi * f * t)

    return y

def spectralevents_find (findMethod, thrFOM, tVec, fVec, TFR, classLabels, neighbourhood_size, threshold, Fs):
    # SPECTRALEVENTS_FIND Algorithm for finding and calculating spectral 
    #   events on a trial-by-trial basis of of a single subject/session. Uses 
    #   one of three methods before further analyzing and organizing event 
    #   features:
    #
    #   1) (Primary event detection method in Shin et al. eLife 2017): Find 
    #      spectral events by first retrieving all local maxima in 
    #      un-normalized TFR using imregionalmax, then selecting suprathreshold
    #      peaks within the frequency band of interest. This method allows for 
    #      multiple, overlapping events to occur in a given suprathreshold 
    #      region and does not guarantee the presence of within-band, 
    #      suprathreshold activity in any given trial will render an event.
    #   2) Find spectral events by first thresholding
    #      entire normalize TFR (over all frequencies), then finding local 
    #      maxima. Discard those of lesser magnitude in each suprathreshold 
    #      region, respectively, s.t. only the greatest local maximum in each 
    #      region survives (when more than one local maxima in a region have 
    #      the same greatest value, their respective event timing, freq. 
    #      location, and boundaries at full-width half-max are calculated 
    #      separately and averaged). This method does not allow for overlapping
    #      events to occur in a given suprathreshold region and does not 
    #      guarantee the presence of within-band, suprathreshold activity in 
    #      any given trial will render an event.
    #   3) Find spectral events by first thresholding 
    #      normalized TFR in frequency band of interest, then finding local 
    #      maxima. Discard those of lesser magnitude in each suprathreshold region,
    #      respectively, s.t. only the greatest local maximum in each region
    #      survives (when more than one local maxima in a region have the same 
    #      greatest value, their respective event timing, freq. location, and 
    #      boundaries at full-width half-max are calculated separately and 
    #      averaged). This method does not allow for overlapping events to occur in
    #      a given suprathreshold region and ensures the presence of 
    #      within-band, suprathreshold activity in any given trial will render 
    #      an event.
    #
    # specEv_struct = spectralevents_find(findMethod,eventBand,thrFOM,tVec,fVec,TFR,classLabels)
    # 
    # Inputs:
    #   findMethod - integer value specifying which event-finding method 
    #       function to run. Note that the method specifies how much overlap 
    #       exists between events.
    #   eventBand - range of frequencies ([Fmin_event Fmax_event]; Hz) over 
    #       which above-threshold spectral power events are classified.
    #   thrFOM - factors of median threshold; positive real number used to
    #       threshold local maxima and classify events (see Shin et al. eLife 
    #       2017 for discussion concerning this value).
    #   tVec - time vector (s) over which the time-frequency response (TFR) is 
    #       calcuated.
    #   fVec - frequency vector (Hz) over which the time-frequency response 
    #       (TFR) is calcuated.
    #   TFR - time-frequency response (TFR) (trial-frequency-time) for a
    #       single subject/session.
    #   classLabels - numeric or logical 1-row array of trial classification 
    #       labels; associates each trial of the given subject/session to an 
    #       experimental condition/outcome/state (e.g., hit or miss, detect or 
    #       non-detect, attend-to or attend away).
    #
    # Outputs:
    #   specEv_struct - event feature structure with three main sub-structures:
    #       TrialSummary (trial-level features), Events (individual event 
    #       characteristics), and IEI (inter-event intervals from all trials 
    #       and those associated with only a given class label).
    #
    # See also SPECTRALEVENTS, SPECTRALEVENTS_FIND, SPECTRALEVENTS_TS2TFR, SPECTRALEVENTS_VIS.

    # Initialize general data parameters
    # Number of elements in discrete frequency spectrum
    flength = TFR.shape[1]
    # Number of point in time
    tlength = TFR.shape[2]
    # Number of trials
    numTrials = TFR.shape[0]
    classes = np.unique(classLabels)

    # Median power at each frequency across all trials
    TFRpermute = np.transpose(TFR, [1, 2, 0]) # freq x time x trial
    TFRreshape = np.reshape(TFRpermute, (flength, tlength*numTrials))
    medianPower = np.median(TFRreshape, axis=1)

    # Spectral event threshold for each frequency value
    eventThresholdByFrequency = thrFOM*medianPower

    # Validate consistency of parameter dimensions
    if flength != len(fVec):
        sys.exit('Mismatch in frequency dimensions!')
    if tlength != len(tVec): 
        sys.exit('Mismatch in time dimensions!')
    if numTrials != len(classLabels): 
        sys.exit('Mismatch in number of trial!')

    # Find spectral events using appropriate method
    #    Implementing one for now
    if findMethod == 1:
        spectralEvents = find_localmax_method_1(TFR, fVec, tVec, eventThresholdByFrequency, classLabels, medianPower, neighbourhood_size, threshold, Fs)
    elif findMethod == 2:
        spectralEvents = find_localmax_method_2 # HACK!!!!
    elif findMethod == 3:
        spectralEvents = find_localmax_method_3 # HACK!!!!

    return spectralEvents

def fwhm_lower_upper_bound1(vec, peakInd, peakValue):
    # Function to find the lower and upper indices within which the vector is less than the FWHM
    #   with some rather complicated boundary rules (Shin, eLife, 2017)
    halfMax = peakValue/2

    # Extract data before the peak only (data should be rising at the end of the new array)
    vec1 = vec[0:peakInd]
    # Find indices less than half the max
    vec1_underThreshold = np.where(vec1<halfMax)[0]
    if len(vec1_underThreshold)==0:
        # There are no indices less than half the max, so we have to estimate the lower edge
        estimateLowerEdge = True
    else:
        # There are indices less than half the max, take the last one under halfMax as the lower edge
        estimateLowerEdge = False
        lowerEdgeIndex = vec1_underThreshold[-1]

    # Extract data following the peak only (data should be falling at the start of the new array)
    vec2 = vec[peakInd:]
    # Find indices less than half the max
    vec2_underThreshold = np.where(vec2<halfMax)[0]
    if len(vec2_underThreshold)==0:
        # There are no indices less than half the max, so we have to estimate the upper edge
        estimateUpperEdge = True
    else:
        # There are indices less than half the max, take the first one under halfMax as the upper edge
        estimateUpperEdge = False
        upperEdgeIndex = vec2_underThreshold[0] + len(vec1)

    if not estimateLowerEdge:
        if not estimateUpperEdge:
            # FWHM fits in the range, so pick off the edges of the FWHM
            lowerInd = lowerEdgeIndex
            upperInd = upperEdgeIndex
            FWHM = upperInd - lowerInd
        if estimateUpperEdge:
            # FWHM fits in on the low end, but hits the edge on the high end
            lowerInd = lowerEdgeIndex
            upperInd = len(vec)-1
            FWHM = 2 * (peakInd - lowerInd + 1)
    else:
        if not estimateUpperEdge:
            # FWHM hits the edge on the low end, but fits on the high end
            lowerInd = 0
            upperInd = upperEdgeIndex
            FWHM = 2 * (upperInd - peakInd + 1)
        if estimateUpperEdge:
            # FWHM hits the edge on the low end and the high end
            lowerInd = 0
            upperInd = len(vec)-1
            FWHM = 2*len(vec)

    '''
    plt.plot(vec)
    plt.axhline(y=peakValue)
    plt.axvline(x=lowerInd, color='#aa0000')
    plt.axvline(x=upperInd, color='#00aa00')
    plt.scatter(peakInd,peakValue)
    plt.show()
    '''

    return lowerInd, upperInd, FWHM

def find_localmax_method_1(TFR, fVec, tVec, eventThresholdByFrequency, classLabels, medianPower, neighbourhood_size,
                           threshold, Fs):
    # 1st event-finding method (primary event detection method in Shin et
    # al. eLife 2017): Find spectral events by first retrieving all local
    # maxima in un-normalized TFR using imregionalmax, then selecting
    # suprathreshold peaks within the frequency band of interest. This
    # method allows for multiple, overlapping events to occur in a given
    # suprathreshold region and does not guarantee the presence of
    # within-band, suprathreshold activity in any given trial will render
    # an event.

    # spectralEvents: 12 column matrix for storing local max event metrics:
    #        trial index,            hit/miss,         maxima frequency,
    #        lowerbound frequency,     upperbound frequency,
    #        frequency span,         maxima timing,     event onset timing,
    #        event offset timing,     event duration, maxima power,
    #        maxima/median power
    # Number of elements in discrete frequency spectrum
    flength = TFR.shape[1]
    # Number of point in time
    tlength = TFR.shape[2]
    # Number of trials
    numTrials = TFR.shape[0]

    spectralEvents = []

    # Retrieve all local maxima in TFR using python equivalent of imregionalmax
    for ti in range(numTrials):

        # Get TFR data for this trial [frequency x time]
        thisTFR = TFR[ti, :, :]

        # Find local maxima in the TFR data
        data = thisTFR
        data_max = filters.maximum_filter(data, neighbourhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighbourhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))

        numPeaks = len(xy)

        peakF = []
        peakT = []
        peakPower = []
        for thisXY in xy:
            peakF.append(int(thisXY[0]))
            peakT.append(int(thisXY[1]))
            peakPower.append(thisTFR[peakF[-1], peakT[-1]])

        # Find local maxima lowerbound, upperbound, and full width at half max
        #    for both frequency and time
        Ffwhm = []
        Tfwhm = []
        for lmi in range(numPeaks):
            thisPeakF = peakF[lmi]
            thisPeakT = peakT[lmi]
            thisPeakPower = peakPower[lmi]

            # Indices of TFR frequencies < half max power at the time of a given local peak
            TFRFrequencies = thisTFR[:, thisPeakT]
            lowerInd, upperInd, FWHM = fwhm_lower_upper_bound1(TFRFrequencies,
                                                               thisPeakF, thisPeakPower)
            lowerEdgeFreq = fVec[lowerInd]
            upperEdgeFreq = fVec[upperInd]
            FWHMFreq = FWHM

            # Indices of TFR times < half max power at the frequency of a given local peak
            TFRTimes = thisTFR[thisPeakF, :]
            lowerInd, upperInd, FWHM = fwhm_lower_upper_bound1(TFRTimes,
                                                               thisPeakT, thisPeakPower)
            lowerEdgeTime = tVec[lowerInd]
            upperEdgeTime = tVec[upperInd]
            FWHMTime = FWHM / Fs

            # Put peak characteristics to a dictionary
            #        trial index,            hit/miss,         maxima frequency,
            #        lowerbound frequency,     upperbound frequency,
            #        frequency span,         maxima timing,     event onset timing,
            #        event offset timing,     event duration, maxima power,
            #        maxima/median power
            peakParameters = {
                'Trial': ti,
                'Hit/Miss': classLabels[ti],
                'Peak Frequency': fVec[thisPeakF],
                'Lower Frequency Bound': lowerEdgeFreq,
                'Upper Frequency Bound': upperEdgeFreq,
                'Frequency Span': FWHMFreq,
                'Peak Time': tVec[thisPeakT],
                'Event Onset Time': lowerEdgeTime,
                'Event Offset Time': upperEdgeTime,
                'Event Duration': FWHMTime,
                'Peak Power': thisPeakPower,
                'Normalized Peak Power': thisPeakPower / medianPower[thisPeakF],
                'Outlier Event': thisPeakPower > eventThresholdByFrequency[thisPeakF]
            }

            # Build a list of dictionaries
            spectralEvents.append(peakParameters)

    return spectralEvents


def get_spectral_events(subjectID):
    """Top-level run script for finding spectral events in MEG data."""

    # Event-finding method (1 allows for maximal overlap while 2 limits overlap in each respective suprathreshold region)
    #   Only method 1 is implemented so far
    findMethod = 1  # Method 4 is a potential fix of method 1 (which gives negative durations and freq spans)
    # Factors of Median threshold (see Shin et al. eLife 2017 for details concerning this value)
    thrFOM = 6 

    fmin = 1        # Hertz (integer)
    fmax = 60       # Hertz (integer)
    fstep = 1       # Hertz (integer)
    width = 10      # integer number of samples
    channelName = 'MEG0711'

    footprintFreq = 4
    footprintTime = 80
    threshold = 0.00
    neighbourhood_size = (footprintFreq,footprintTime)

    # Setup paths and names for file
    dataDir = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef'
    outDir = os.path.join('/home/timb/camcan/spectralEvents', subjectID)
    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
    spectralEventsCSV = "".join([channelName, '_spectral_events_-1.0to1.0s.csv'])
    csvFile = os.path.join(outDir, spectralEventsCSV)
    logFile = os.path.join(outDir, "".join([channelName, '_', str(fmin), '-', str(fmax), 'Hz', '_', str(fstep),
                                            'Hzstep_get_spectral_events.log']))

    plotOK = False

    ################################
    # Processing starts here
    logger.info(subjectID)
    # Make the filename with path
    epochFif = os.path.join(dataDir, subjectID, epochFifFilename)

    if os.path.exists(epochFif):

        # Make output file folder
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # Setup log file for standarda output and error
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename=logFile,
            filemode='w'
        )
        '''
        stdout_logger = logging.getLogger('STDOUT')
        sl = StreamToLogger(stdout_logger, logging.INFO)
        sys.stdout = sl
        stderr_logger = logging.getLogger('STDERR')
        sl = StreamToLogger(stderr_logger, logging.ERROR)
        sys.stderr = sl
        '''

        # Read the epochs
        epochs = mne.read_epochs(epochFif)
        Fs = epochs.info['sfreq']

        # Extract the data
        epochs.pick_channels([channelName])
        epochData = np.squeeze(epochs.get_data())
        tmin = epochs.times[0]

        # Vector of frequencies in TFR [Hz]
        fVec = np.arange(fmin, fmax+1, fstep)

        # Make a TFR for each epoch [trials x frequency x time]
        logger.info('Calculating TFRs')
        TFR, tVec, fVec = spectralevents_ts2tfr(epochData.T,fVec,Fs,width)

        # Vector of times in TFR [s]
        tVec = tVec + tmin

        # Set all class labels to the same value (button press)
        numTrials = TFR.shape[0]
        classLabels = [1 for x in range(numTrials)]

        # Find spectral events based on TFR
        logger.info('Finding Spectral Events')
        spectralEvents = spectralevents_find (findMethod, thrFOM, tVec,
            fVec, TFR, classLabels, neighbourhood_size, threshold, Fs)
        logger.info('Found ' + str(len(spectralEvents)) + ' spectral events.')

        # Write data to a CSV file
        logger.info('Writing event characteristics to file')
        df = pd.DataFrame(spectralEvents)
        df['Subject ID'] = subjectID
        df.to_csv(csvFile)


if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
    subjectData = pd.read_csv(camcanCSV)

    # Take only subjects with more than 55 epochs
    subjectData = subjectData[subjectData['numEpochs'] > 55]
    subjectIDs = subjectData['SubjectID'].tolist()

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/3))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(get_spectral_events, subjectIDs)

    # Or run one subject for testing purposes
    #get_spectral_events(subjectIDs[0])






