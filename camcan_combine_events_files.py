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


def read_spectral_events(subjectID):
    """Top-level run script for finding spectral events in MEG data."""

    channelName = 'MEG0711'

    # Setup paths and names for file
    dataDir = os.path.join('/home/timb/camcan/spectralEvents', subjectID)
    spectralEventsCSV = "".join([channelName, '_spectral_events_-1.0to1.0s.csv'])
    csvFile = os.path.join(dataDir, spectralEventsCSV)

    # Read all spectral events from CSV file to dataframe
    df = pd.read_csv(csvFile)

    # Keep only outlier events
    df = df[df['Outlier Event']]

    return df

if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
    subjectData = pd.read_csv(camcanCSV)

    # File to write
    channelName = 'MEG0711'
    allSubjectsCSV = os.path.join(dataDir, 'spectralEvents', "".join([channelName, '_spectral_events_-1.0to1.0s.csv']))

    # Take only subjects with more than 55 epochs
    subjectData = subjectData[subjectData['numEpochs'] > 55]
    subjectIDs = subjectData['SubjectID'].tolist()

    # Read all CSVs with spectral events (only outlier events are included)
    allDfs = []
    for subjectID in subjectIDs:
        allDfs.append(read_spectral_events(subjectID))

    # Concatenate all events to one list
    allDf = pd.concat(allDfs)
    allDf.to_csv(allSubjectsCSV)

    # Crop the events a little bit to get rid of edge effects
    dfSub = allDf[allDf['Peak Frequency'] < 60]
    dfSub = dfSub[dfSub['Peak Time'] >= -0.75]
    dfSub = dfSub[dfSub['Peak Time'] <= 0.75]

    # Plot as 2D histogram
    sns.jointplot(data=dfSub, x='Peak Time', y='Peak Frequency', kind='hex')
    plt.show()

