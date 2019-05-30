#Order of operations (and summaries) for scripts in this folder

`Builds a .csv to indicate which participant's have what files generated`
camcan_process_tracker.py

`Detects and characterises spectral events for CamCAN task MEG data (MEG0711 only)`
camcan_task_spectralevents.py

`Detects and characterises spectral events for CamCAN task MEG data 
at the source level using the betaERD region of interest from
NeuroImage 2019 paper`
camcan_task_betaERD_ROI_spectralevents.py

`Combine all spectral events across participants (MEG0711 only) `
camcan_combine_events_files.py			

`Plot some features of the spectral events (MEG0711 only)`
camcan_plot_events.py

`Make DICS BF maps of change in power between spectral events and
the time interval just prior (based on timing from MEG0711), 
including morphing to fsaverage`
```For post-movement beta rebound interval```
camcan_make_PMBR_BF_map.py			
```For pre-stimulus interval```
camcan_make_preStim_BF_map.py			

`Hack scripts to make the DICS BF files that didn't get made in the 
multiprocess scripts above`
camcan_make_remaining_preStim_BF_map.py
camcan_make_remaining_PMBR_BF_map.py

`Script to average DICS BF files across participants and plot the
BFs on the fsaverage brain`
average_PMBR_stc_files.py			
average_stc_files.py				

`Further plotting of the grand-average DICS BF data`
plot_spectralEvents_DICS.py

`Calculate similarity between each DICS BF map and the grand-average
using Sorenson-Dice coefficient`
camcan_preStim_similarity.py

`Calculate and plot the normalized difference between pre-stim DICS BF
and PMBR DICS BF in grand-average`
camcan_compare_PMBR_to_preStim_stc.py		
