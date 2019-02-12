import numpy as np

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

	t=-3.5*st:dt:3.5*st; #??? NOT SURE WHAT THIS DOES
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


