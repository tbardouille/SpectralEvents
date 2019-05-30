import mne
import os
import numpy as np

# Script to read in DICS beamformer maps for transient beta events in the
#   prestimulus and PMBR intervals of CamCAN data
#
# Note: data files are copied from Biden folder:
#       /home/timb/camcan/spectralEvents

################### Interesting next steps ####################
#
#   1. Single subject differences to get statistical test of resultant map
#   2. ROI time courses to establish independent transient spectral events at each ROI
#           This would show if the bursts at the PMBR ROI are temporally distinct from the 
#           burst at the preStim (ERD) ROI  

################################
# Load PMBR data

# Set folders and files
dataDir = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/SpectralEvents/source_data'
subjectDir = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/subjects/'
stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_PMBREvents_DICS_fsaverage'

# Find all subject folders that exist
subjects = os.listdir(dataDir)

# Loop over all subject folders
stcs = []
for subjectID in subjects:

    # Set file path for stc file (without Xh.stc)
    thisStcFile = os.path.join(dataDir, subjectID, stcPrefix)

    # Set file path with lh.stc to make sure the file exists
    fileCheckName = "".join([thisStcFile, '-lh.stc'])

    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs.append(stc.data)

# Turn list of stc data elements into an array (participants x vertices x 1)
stcArray = np.asarray(stcs)

# Average over participants and make an stc
stcGAvgData = np.mean(stcArray, axis=0)
stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices, 
    tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
stcPMBR = stcGAvgData


################################
# Load prestim data

stcPrefix = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS_fsaverage'
stcs = []
for subjectID in subjects:

    # Set file path for stc file (without Xh.stc)
    thisStcFile = os.path.join(dataDir, subjectID, stcPrefix)

    # Set file path with lh.stc to make sure the file exists
    fileCheckName = "".join([thisStcFile, '-lh.stc'])

    # If file exists, add the stc data to a list
    if os.path.exists(fileCheckName):
        stc = mne.read_source_estimate(thisStcFile)
        stcs.append(stc.data)

# Turn list of stc data elements into an array (participants x vertices x 1)
stcArray = np.asarray(stcs)

# Average over participants and make an stc
stcGAvgData = np.mean(stcArray, axis=0)
stcGAvg = mne.SourceEstimate(stcGAvgData, vertices=stc.vertices, 
    tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
stcPreStim = stcGAvgData
stcPMBR.shape
stcPMBR_norm = stcPMBR/np.max(stcPMBR)
stcPreStim_norm = stcPreStim/np.max(stcPreStim)

# Match maximum value in both datasets
stcNormDiff = stcPMBR_norm - stcPreStim_norm

# Take the difference (PMBR bigger is positive)
stcNew = mne.SourceEstimate(stcNormDiff, vertices=stc.vertices, 
    tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')

# Plot new data
stcNew.plot(surface='pial', hemi='both', subjects_dir=subjectDir, 
    subject='fsaverage', backend='mayavi', time_viewer=True,
    clim={'pos_lims': [0.1, 0.11, 0.2], 'neg_lims': [-0.1, -0.11, -0.2]})


