#!/usr/bin/env python

# Import libraries
import os
import pandas as pd

# Paths and variables
homeDir = os.path.expanduser("~")
dataDir = homeDir + '/camcan/'
camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
channelName = 'MEG0711'

preStimStcFileName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS-lh.stc'
PMBRStcFileName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_PMBREvents_DICS-lh.stc'

spectralEventsLogCSV = 'spectralEventAnalysis.csv'

# Take only subjects with more than 55 epochs
subjectData = pd.read_csv(camcanCSV)
subjectData = subjectData[subjectData['numEpochs'] > 55]
subjectIDs = subjectData['SubjectID'].tolist()

# Only keep columns relevant for the spectral events analysis
subjectData = subjectData[['SubjectID', 'Age_x', 'Gender_x', 'Hand_x',
    'numEpochs', 'transExists', 'srcExists', 'bemExists',
    'PMBRPeakFrequency', 'PMBRPeakAmplitude', 'PMBRPeakTime',
    'ERDPeakFrequency', 'ERDPeakAmplitude', 'ERDPeakTime',
    'MRGBPeakFrequency', 'MRGBPeakAmplitude', 'MRGBPeakTime']]

spectralEventsCalculated = []
numEvents = []
numSpectralEvents = []
numPreStimSpectralEvents = []
preStimStcExists = []
numPMBRSpectralEvents = []
PMBRStcExists = []
for subjectID in subjectIDs:

    # Setup paths and names for file
    spectralEventsCSV = "".join([channelName, '_spectral_events_-1.0to1.0s.csv'])
    csvFile = os.path.join(dataDir, 'spectralEvents', subjectID, spectralEventsCSV)

    if os.path.exists(csvFile):

        spectralEventsCalculated.append(True)
        df = pd.read_csv(csvFile)
        numEvents.append(len(df))

        # Events that meet Shin criteria only
        df1 = df[df['Outlier Event']]
        numSpectralEvents.append(len(df1))

        # PreStim
        fmin, fmax, startTime, endTime = 15, 30, -1.1, 0
        df2 = df1.drop(df1[df1['Lower Frequency Bound'] < fmin].index)
        df3 = df2.drop(df2[df2['Upper Frequency Bound'] > fmax].index)
        df4 = df3.drop(df3[df3['Event Offset Time'] > endTime].index)
        newDf = df4.drop(df4[df4['Event Onset Time'] < startTime].index)
        numPreStimSpectralEvents.append(len(newDf))

        stcFile = os.path.join(dataDir, 'spectralEvents', subjectID, preStimStcFileName)
        preStimStcExists.append(os.path.exists(stcFile))

        # PMBR
        fmin, fmax, startTime, endTime = 15, 30, 0.4, 1.3
        df2 = df1.drop(df1[df1['Lower Frequency Bound'] < fmin].index)
        df3 = df2.drop(df2[df2['Upper Frequency Bound'] > fmax].index)
        df4 = df3.drop(df3[df3['Event Offset Time'] > endTime].index)
        newDf = df4.drop(df4[df4['Event Onset Time'] < startTime].index)
        numPMBRSpectralEvents.append(len(newDf))

        stcFile = os.path.join(dataDir, 'spectralEvents', subjectID, PMBRStcFileName)
        PMBRStcExists.append(os.path.exists(stcFile))

    else:
        spectralEventsCalculated.append(False)
        numEvents.append(0)
        numSpectralEvents.append(0)
        numPreStimSpectralEvents.append(0)
        preStimStcExists.append(False)
        numPMBRSpectralEvents.append(0)
        PMBRStcExists.append(False)

subjectData['Events Calculated'] = spectralEventsCalculated
subjectData['Num Maxima'] = numEvents
subjectData['Num Spectral Events'] = numSpectralEvents
subjectData['Num PreStim Events'] = numPreStimSpectralEvents
subjectData['Num PMBR Events'] = numPMBRSpectralEvents
subjectData['PreStim Stc Exists'] = preStimStcExists
subjectData['PMBR Stc Exists'] = PMBRStcExists

subjectData.to_csv( os.path.join(dataDir, 'spectralEvents', spectralEventsLogCSV) )
