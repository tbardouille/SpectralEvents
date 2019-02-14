import numpy as np
import scipy.ndimage.filters as filters

#   -----------------------------------------------------------------------
#   SpectralEvents::spectralevents_ts2tfr
#   Copyright (C) 2018  Ryan Thorpe
#	Adapted by Tim Bardouille - 2019 (tim.bardouille@dal.ca)
#
#   This file is part of the SpectralEvents toolbox.
# 
#   SpectralEvents is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   SpectralEvents is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#   -----------------------------------------------------------------------

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

	B = np.zeros(numFrequencies, numSamples)

	for i in np.arange(numTrials):
		for j in np.arange(numFrequencies):

			B[j,:] = energyvec(fVec[j], detrend(S[i,:]), Fs, width) + B[j,:]

	TFR = B/numTrials

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
	y = 2 * (dt * np.abs(y))^2
	y = y[np.ceil(len(m)/2):len(y)-np.floor(len(m)/2)]

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
	y = A * np.exp(-t^2 / (2 * st^2)) * np.exp(1j * 2 * np.pi * f * t)

	return y

def spectralevents_find (findMethod, eventBand, thrFOM, tVec, fVec, TFR, classLabels):
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
	#   TFR - time-frequency response (TFR) (frequency-by-time-trial) for a
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
	# Logical vector representing indices of freq vector within eventBand
	a = fVec >= eventBand[0]
	b = fVec <= eventband[1]
	eventBand_inds = a*b 
	# Number of elements in discrete frequency spectrum
	flength = TFR.shape[0]
	# Number of point in time
	tlength = TFR.shape[1]
	# Number of trials
	numTrials = TFR.shape[2]
	classes = np.unique(classLabels)

	# Median power at each frequency across all trials
	TFRreshape = np.reshape(TFR, (flength, tlength*numTrials))
	medianPower = np.median(TFRreshape, axis=1)

	# Spectral event threshold for each frequency value
	thr = thrFOM*medianPower

	# Validate consistency of parameter dimensions
	if flength != len(fVec) 
  		error('Mismatch in input parameter dimensions!')
	end
	if tlength != len(tVec) 
  		error('Mismatch in input parameter dimensions!')
	end
	if numTrials != len(classLabels) 
  		error('Mismatch in input parameter dimensions!')
	end

	# Find spectral events using appropriate method
	#	Implementing one for now
	if findMethod == 1:
		spectralEvents = find_localmax_method_1
	elif findMethod == 2:
		spectralEvents = find_localmax_method_1 # HACK!!!!
	elif findMethod == 3:
		spectralEvents = find_localmax_method_3 # HACK!!!!

	return spectralEvents

def find_localmax_method_1(TFR, fVec, tVec, classLabels, medianPower):
	# 1st event-finding method (primary event detection method in Shin et 
    # al. eLife 2017): Find spectral events by first retrieving all local 
    # maxima in un-normalized TFR using imregionalmax, then selecting 
    # suprathreshold peaks within the frequency band of interest. This 
    # method allows for multiple, overlapping events to occur in a given 
    # suprathreshold region and does not guarantee the presence of 
    # within-band, suprathreshold activity in any given trial will render 
    # an event.

    # spectralEvents: 12 column matrix for storing local max event metrics: 
    #		trial index,			hit/miss, 		maxima frequency, 
    #		lowerbound frequency, 	upperbound frequency, 
    #		frequency span, 		maxima timing, 	event onset timing, 
    #		event offset timing, 	event duration, maxima power, 
    #		maxima/median power
	# Number of elements in discrete frequency spectrum
	flength = TFR.shape[0]
	# Number of point in time
	tlength = TFR.shape[1]
	# Number of trials
	numTrials = TFR.shape[2]

    spectralEvents = [];

    # Finds_localmax: stores peak frequecy at each local max (columns) for each
    # trial (rows)
    Finds_localmax = [];

    # Retrieve all local maxima in TFR using imregionalmax
    for t1 in range(numTrials):

    	thisTFR = TFR[:,:,ti]
    	# Find local maxima in the TFR data
    	TFRPeaks = filters.maximum_filter(dat, size=(3,3))
    	# Indices of max local power
    	ind = np.where(TFRPeaks==1)
    	peakF, peakT = np.unravel_index(ind, (flength, tlength), order='C')
    	# Power values at local maxima
    	peakPower = thisTFR(peakF, peakT)
    	numPeaks = len(peakPower)

    	#Find local maxima lowerbound, upperbound, and full width at half max
        #	for both frequency and time
        Ffwhm = []
        Tfwhm = []
        for lmi in range(numPeaks):

        	thisPeakF = peakF[lmi]
        	thisPeakT = peakT[lmi]
        	thisPeakPower = peakPower[lmi]
        	# Indices of TFR frequencies < half max power at the time of a given local peak
            fwhmFrequencies = np.squeeze(thisTFR[:,thisPeakT) < thisPeakPower/2
            lmF_underthr = np.where(fwhmFrequencies)[0]
            # Indices of frequencies below the FWHM
            belowFWHM = np.where(lmF_underthr < thisPeakF)[0]
            # Indices of frequencies above the FWHM
            aboveFWHM = np.where(lmF_underthr > thisPeakF)[0]
            # Does the FWHM include the lower bound?
            noLowerEdge = len(belowFWHM) == 0
            # Does the FWHM include the upper cound?
            noUpperEdge = len(aboveFWHM) == 0
            if not noLowerEdge:
            	if not noUpperEdge:
            		# FWHM fits in the range, so pick off the edges of the FWHM
            		lowerEdgeFreq = fVec[lmF_underthr[belowFWHM[-1]+1]]
            		upperEdgeFreq = fVec[lmF_underthr[aboveFWHM[0]-1]]
            		FWHMFreq = upperEdgeFreq - lowerEdgeFreq + np.min(np.diff(fVec))
            	if noUpperEdge:
            		# FWHM fits in on the low end, but hits the edge on the high end
            		lowerEdgeFreq = fVec[lmF_underthr[belowFWHM[-1]]]
            		upperEdgeFreq = fVec[-1]
            		FWHMFreq = 2 * (fVec[peakF[lmi]] - lowerEdgeFreq + np.min(np.diff(fVec)))
  			else:
  				if not noUpperEdge:
            		# FWHM hits the edge on the low end, but fits on the high end
            		lowerEdgeFreq = fVec[0]
            		upperEdgeFreq = fVec[lmF_underthr[aboveFWHM[0]-1]]
            		FWHMFreq = 2 * (upperEdgeFreq - fVec[peakF[lmi]] + np.min(np.diff(fVec)))
            	if noUpperEdge:
            		# FWHM hits the edge on the low end and the high end
            		lowerEdgeFreq = fVec[0]
            		upperEdgeFreq = fVec[-1]
            		FWHMFreq = 2 * (upperEdgeFreq - lowerEdgeFreq + np.min(np.diff(fVec)))
  			          		
        	# Indices of TFR times < half max power at the time of a given local peak
            fwhmTimes = np.squeeze(thisTFR[thisPeakF,:) < thisPeakPower/2
            lmT_underthr = np.where(fwhmTimes)[0]
            # Indices of frequencies below the FWHM
            belowFWHM = np.where(lmT_underthr < thisPeakT)[0]
            # Indices of frequencies above the FWHM
            aboveFWHM = np.where(lmT_underthr > thisPeakT)[0]
            # Does the FWHM include the lower bound?
            noLowerEdge = len(belowFWHM) == 0
            # Does the FWHM include the upper cound?
            noUpperEdge = len(aboveFWHM) == 0
            if not noLowerEdge:
            	if not noUpperEdge:
            		# FWHM fits in the range, so pick off the edges of the FWHM
            		lowerEdgeTime = tVec[lmT_underthr[belowFWHM[-1]+1]]
            		upperEdgeTime = tVec[lmT_underthr[aboveFWHM[0]-1]]
            		FWHMTime = upperEdgeTime - lowerEdgeTime + np.min(np.diff(tVec))
            	if noUpperEdge:
            		# FWHM fits in on the low end, but hits the edge on the high end
            		lowerEdgeTime = tVec[lmT_underthr[belowFWHM[-1]]]
            		upperEdgeTime = tVec[-1]
            		FWHMTime = 2 * (tVec[peakT[lmi]] - lowerEdgeTime + np.min(np.diff(tVec)))
  			else:
  				if not noUpperEdge:
            		# FWHM hits the edge on the low end, but fits on the high end
            		lowerEdgeTime = tVec[0]
            		upperEdgeTime = tVec[lmT_underthr[aboveFWHM[0]-1]]
            		FWHMTime = 2 * (upperEdgeTime - tVec[peakT[lmi]] + np.min(np.diff(tVec)))
            	if noUpperEdge:
            		# FWHM hits the edge on the low end and the high end
            		lowerEdgeTime = tVec[0]
            		upperEdgeTime = tVec[-1]
            		FWHMTime = 2 * (upperEdgeTime - lowerEdgeTime + np.min(np.diff(tVec)))

		    #		trial index,			hit/miss, 		maxima frequency, 
		    #		lowerbound frequency, 	upperbound frequency, 
		    #		frequency span, 		maxima timing, 	event onset timing, 
		    #		event offset timing, 	event duration, maxima power, 
		    #		maxima/median power
            peakParameters = {
            	'Trial': t1,
            	'Hit/Miss': classLabels[ti],
            	'Peak Frequency': fVec[peakF],
            	'Lower Frequency Bound': lowerEdgeFreq,
            	'Upper Frequency Bound': upperEdgeFreq,
            	'Frequency Span': FWHMFreq,
            	'Peak Time': tVec[peakT],
            	'Event Onset Time': lowerEdgeTime,
            	'Event Offset Time': upperEdgeTime,
            	'Event Duration': FWHMTime,
            	'Peak Power': thisTFR[peakF, peakT],
            	'Normalized Peak Power': thisTFR[peakF, peakT]/medianPower[peakF]
            	}
            spectralEvents.append(peakParameters)


