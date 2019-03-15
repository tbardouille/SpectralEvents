#!/usr/bin/env python

# Import libraries
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import tfr_morlet
import multiprocessing as mp


def make_BF_map(subjectID):
    """Top-level run script for making spectral events BF map from MEG data."""

    #################################
    # Variables

    channelName = 'MEG0711'

    # BF Analysis Paramaters
    fmin = 12
    fmax = 30
    startTime = -1.3
    endTime = 0.
    eventDuration = 0.4 # See distplot below for justification

    # TFR analysis parameters
    TFRfmin = 5
    TFRfmax = 60
    TFRfstep = 5

    # DICS Settings
    tmins = [0.0, -0.4] # Start of each window (active, baseline)
    tstep = 0.4
    fmin = 15
    fmax = 30
    numFreqBins = 10  # linear spacing
    DICS_regularizaion = 0.5
    data_decimation = 1

    plotOK = False

    ###############
    # Setup paths and names for file
    dataDir = '/home/timb/camcan/'
    MEGDir = os.path.join(dataDir, 'proc_data/TaskSensorAnalysis_transdef')
    outDir = os.path.join(dataDir, 'spectralEvents', subjectID)
    subjectsDir = os.path.join(dataDir, 'subjects/')

    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFif = os.path.join(MEGDir, subjectID, epochFifFilename)

    spectralEventsCSV = "".join([channelName, '_spectral_events_-1.0to1.0s.csv'])
    csvFile = os.path.join(outDir, spectralEventsCSV)

    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'

    # Files to make
    stcFile = os.path.join(outDir,
                           'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS')
    stcMorphFile = os.path.join(outDir,
                                'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS_fsaverage')

    #####################################
    # Pull events from CSV

    # Read all transient events for subject
    df = pd.read_csv(csvFile)
    # Events that meet Shin criteria only
    df1 = df[df['Outlier Event']]
    # Freq range of interest
    df2 = df1.drop(df1[df1['Lower Frequency Bound'] < fmin].index)
    df3 = df2.drop(df2[df2['Upper Frequency Bound'] > fmax].index)
    df4 = df3.drop(df3[df3['Event Offset Time'] > endTime].index)
    newDf = df4.drop(df4[df3['Event Onset Time'] < startTime].index)

    if plotOK:
        # Raster plot of event onset and offset times
        ax = sns.scatterplot(x='Event Onset Time', y='Trial', data=newDf)
        sns.scatterplot(x='Event Offset Time', y='Trial', data=newDf, ax=ax)
        plt.show()

        # Distribution of event durations
        sns.distplot(newDf['Event Duration'])
        plt.show()
        # Based on the distribution, an interval of 0-400 ms will include the full event duration in most cases

    ##############################################
    # Now do the DICS beamformer map calcaulation

    # Read epochs
    originalEpochs = mne.read_epochs(epochFif)

    # Re-calculate epochs to have one per spectral event
    numEvents = len(newDf)
    epochList = []
    for e in np.arange(numEvents):
        thisDf = newDf.iloc[e]
        onsetTime = thisDf['Event Onset Time']
        epoch = originalEpochs[thisDf['Trial']]
        epochCrop = epoch.crop(onsetTime-eventDuration, onsetTime+eventDuration)
        epochCrop = epochCrop.apply_baseline(baseline=(None,None))
        # Fix epochCrops times array to be the same every time = (-.4, .4)
        epochCrop.shift_time(tmins[1], relative=False)
        epochList.append(epochCrop)

    epochs = mne.concatenate_epochs(epochList)


    '''
    # Let's look at the TFR across sensors
    magPicks = mne.pick_types(epochs.info, meg='mag', eeg=False, eog=False, stim=False, exclude='bads')
    epochs_no_evoked = epochs.copy().subtract_evoked()
    freqs = np.arange(TFRfmin, TFRfmax, TFRfstep)
    n_cycles = freqs / 2.0
    power, _ = tfr_morlet(epochs_no_evoked, freqs=freqs, n_cycles=n_cycles, picks=magPicks,
                          use_fft=False, return_itc=True, decim=1, n_jobs=1)
    #power.save(tfrFile, overwrite=True)
    if plotOK:
        power.plot_joint(baseline=(-0.4, 0), mode='mean',
                         timefreqs = [(.2, 20)])
    '''

    # Read source space
    src = mne.read_source_spaces(srcFif)
    # Make forward solution
    forward = mne.make_forward_solution(epochs.info,
                                        trans=transFif, src=src, bem=bemFif,
                                        meg=True, eeg=False)

    # DICS Source Power example
    # https://martinos.org/mne/stable/auto_examples/inverse/plot_dics_source_power.html#sphx-glr-auto-examples-inverse-plot-dics-source-power-py

    # Compute DICS spatial filter and estimate source power.
    stcs = []
    for tmin in tmins:
        csd = csd_morlet(epochs, tmin=tmin, tmax=tmin + tstep, decim=data_decimation,
                         frequencies=np.linspace(fmin, fmax, numFreqBins))
        filters = make_dics(epochs.info, forward, csd, reg=DICS_regularizaion)
        stc, freqs = apply_dics_csd(csd, filters)
        stcs.append(stc)

    # Take difference between active and baseline, and mean across frequencies
    ERS = np.log2(stcs[0].data / stcs[1].data)
    a = stcs[0]
    ERSstc = mne.SourceEstimate(ERS, vertices=a.vertices, tmin=a.tmin, tstep=a.tstep, subject=a.subject)
    ERSband = ERSstc.mean()
    ERSband.save(stcFile)

    ERSmorph = ERSband.morph(subject_to='fsaverage', subject_from='sub-' + subjectID, subjects_dir=subjectsDir)
    ERSmorph.save(stcMorphFile)

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
    #pool.map(make_BF_map, subjectIDs)

    # Or run one subject for testing purposes
    make_BF_map(subjectIDs[1])

