#!/usr/bin/env python

# Import libraries
import os
import pandas as pd
import mne
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Paths and variables
homeDir = os.path.expanduser("~")
dataDir = homeDir + '/camcan/'
camcanCSV = dataDir + 'spectralEvents/spectralEventAnalysis.csv'

preStimStcFileName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS_fsaverage-lh.stc'
PMBRStcFileName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_PMBREvents_DICS_fsaverage-lh.stc'

# Functions
def Sv(a, b):
    numer = 2*np.abs(np.dot(a.T,b))
    denom = np.abs(np.dot(a.T,a)) + np.abs(np.dot(b.T,b))
    value = numer/denom
    return value[0][0]

# Main program 

# Take only subjects with a .stc file that localizes prestim beta events
subjectData = pd.read_csv(camcanCSV)
subjectData = subjectData[subjectData['PreStim Event Stc Exists']]
subjectIDs = subjectData['SubjectID'].tolist()

# Read stc maps
print('Reading data')
stcData = []
for subjectID in subjectIDs:

    stcFile = os.path.join(dataDir, 'spectralEvents', subjectID, preStimStcFileName)

    thisStc = mne.read_source_estimate(stcFile)
    stcData.append(thisStc.data)

allData = np.asarray(stcData)
stc_gAvg = np.squeeze(np.mean(allData, 0))


# Calculate similarity between maps and grand-average
print('Calculate simiarity indices')
similarity = []
for a in np.arange(len(stcData)):
        similarity.append( Sv(stcData[a],stc_gAvg) )

# Plot similarity values
sns.distplot(similarity, bins=50)
plt.title('Similarity between pre-stim beta event DICS BF maps and grand-average')
plt.xlabel('Sorensen-Dice Coef')
plt.ylabel('# Occurrences')
plt.show()

'''
# Calculate similarity between maps
print('Calculate simiarity indices')
similarity = []
for a in np.arange(len(stcData)):
    for b in stcData[a+1:]:
        similarity.append( Sv(stcData[a],b) )

# Plot similarity values
sns.distplot(similarity, bins=50)
plt.title('Similarity between pairs of pre-stim beta event DICS BF maps')
plt.xlabel('Sorensen-Dice Coef')
plt.ylabel('# Occurrences')
plt.show()
'''
