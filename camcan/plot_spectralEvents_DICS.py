import mne

subjectID = 'CC110033'
subjectDir = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/subjects/'

#min1 = 0.39
#min2 = 0.4
#max1 = 0.6

# Plot grand average source map
stcFif = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/SpectralEvents/source_data/'\
	 + subjectID + '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_preStimBetaEvents_DICS_fsaverage'

stc = mne.read_source_estimate(stcFif)
stc.plot(surface='pial', hemi='both', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True)
#	clim=dict(kind='value', lims=(min1, min2, max1)))

'''
stc.plot(surface='inflated', hemi='rh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True,
	clim=dict(kind='value', lims=(min1, min2, max1)))


# Plot grand average source map in functional ROI
stcFif = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/biden/source_data/' + subjectID + \
	'/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_ERD_DICS_gAvg_ROI'

stc = mne.read_source_estimate(stcFif)
stc.data = -stc.data
stc.plot(surface='inflated', hemi='lh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True,
	clim=dict(kind='value', lims=(min1, min2, max1)))
stc.plot(surface='inflated', hemi='rh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True,
	clim=dict(kind='value', lims=(min1, min2, max1)))


# Plot grand average source map - Top 5%
stcFif = '/Users/tbardouille/Documents/Work/Projects/CambridgeLargeData/Data/biden/source_data/' + subjectID + \
	'/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_ERD_DICS_top5percent'

stc = mne.read_source_estimate(stcFif)
stc.plot(surface='inflated', hemi='lh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True,
	clim=dict(kind='value', lims=(0.2, 0.21, 0.4)))
stc.plot(surface='inflated', hemi='rh', subjects_dir=subjectDir, 
	subject='fsaverage', backend='mayavi', time_viewer=True,
	clim=dict(kind='value', lims=(0.2, 0.21, 0.4)))
'''


