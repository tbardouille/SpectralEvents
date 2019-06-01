function spectralevents_vis(specEv_struct, timeseries, TFRs, tVec, fVec)
% SPECTRALEVENTS_VIS Conduct basic analysis for the purpose of 
%   visualizing dataset spectral event features and generates spectrogram 
%   and probability histogram plots.
%
% spectralevents_VIS(specEv_struct,timeseries,TFRs,tVec,fVec)
%
% Inputs:
%   specEv_struct - spectralevents structure array.
%   timeseries - cell array containing time-series trials by 
%       subject/session.
%   TFRs - cell array with each cell containing the time-frequency response 
%       (freq-by-time-by-trial) for a given subject/session.
%   tVec - time vector (s) over which the time-frequency responses are 
%       shown.
%   fVec - frequency vector (Hz) over which the time-frequency responses
%       are shown.
%
% See also SPECTRALEVENTS, SPECTRALEVENTS_FIND, SPECTRALEVENTS_TS2TFR.

%   -----------------------------------------------------------------------
%   SpectralEvents::spectralevents_vis
%   Copyright (C) 2018  Ryan Thorpe
%
%   This file is part of the SpectralEvents toolbox.
% 
%   SpectralEvents is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   SpectralEvents is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <https://www.gnu.org/licenses/>.
%   -----------------------------------------------------------------------

numSubj = length(specEv_struct); %Number of subjects/sessions

% Spectrograms showing trial-by-trial events (see Figure 2 in Shin et al. eLife 2017)
for subj_i=1:numSubj
    % Extract TFR attributes for given subject/session
    TFR = TFRs{subj_i};
    TFR_norm = TFR./median(reshape(TFR, size(TFR,1), size(TFR,2)*size(TFR,3)), 2);
    classLabels = specEv_struct(subj_i).TrialSummary.TrialSummary.classLabels;
    eventBand = specEv_struct(subj_i).Events.EventBand;
    
    % Extract event attributes for a given subject/session 
    eventThr = specEv_struct(subj_i).Events.Threshold;
    trialInd = specEv_struct(subj_i).Events.Events.trialind;
    maximaTiming = specEv_struct(subj_i).Events.Events.maximatiming;
    maximaFreq = specEv_struct(subj_i).Events.Events.maximafreq;
    
    eventBand_inds = fVec(fVec>=eventBand(1) & fVec<=eventBand(2)); %Indices of freq vector within eventBand
    classes = unique(classLabels); %Array of unique class labels
    
    % Make plots for each type of class
    for cls_i=1:numel(classes)
        trial_inds = find(classLabels==classes(cls_i)); %Indices of TFR trials corresponding with the given class
        
        % Calculate average TFR for a given subject/session and determine
        % number of trials to sample
        if numel(trial_inds)>10
            numSampTrials = 10;
            %avgTFR = mean(TFR(:,:,trial_inds),3);
        elseif numel(trial_inds)>1
            numSampTrials = numel(trial_inds);
            %avgTFR = mean(TFR(:,:,trial_inds),3);
        else
            numSampTrials = numel(trial_inds);
        end
        avgTFR = squeeze(mean(TFR(:,:,trial_inds),3));
        avgTFR_norm = squeeze(mean(TFR_norm(:,:,trial_inds),3));
        
        % Find sample trials to view
        rng('default');
        randTrial_inds = [1:numSampTrials]; %randperm(numel(trial_inds),numSampTrials); %Sample trial indices
        
        % Plot average raw TFR
        figure
        subplot('Position',[0.08 0.75 0.37 0.17])
        imagesc([tVec(1) tVec(end)],[fVec(1) fVec(end)],avgTFR)
        x_tick = get(gca,'xtick');
        set(gca,'xtick',x_tick);
        set(gca,'ticklength',[0.0075 0.025])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[fVec(1),eventBand,fVec(end)])
        ylabel('Hz')
        pos = get(gca,'position');
        colormap jet
        cb = colorbar;
        cb.Position = [pos(1)+pos(3)+0.01 pos(2) 0.008 pos(4)];
        cb.Label.String = 'Spectral power';
        hold on
        line(tVec',repmat(eventBand,length(tVec),1)','Color','k','LineStyle',':')
        hold off
        title({'\fontsize{16}Raw TFR',['\fontsize{12}Dataset ',num2str(subj_i),', Trial class ',num2str(classes(cls_i))]})
        
        % Plot average normalized TFR
        subplot('Position',[0.53 0.75 0.37 0.17])
        imagesc([tVec(1) tVec(end)],[fVec(1) fVec(end)],avgTFR_norm)
        x_tick = get(gca,'xtick');
        set(gca,'xtick',x_tick);
        set(gca,'ticklength',[0.0075 0.025])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[fVec(1),eventBand,fVec(end)])
        pos = get(gca,'position');
        colormap jet
        cb = colorbar;
        cb.Position = [pos(1)+pos(3)+0.02 pos(2) 0.008 pos(4)];
        cb.Label.String = 'FOM spectral power';
        hold on
        line(tVec',repmat(eventBand,length(tVec),1)','Color','k','LineStyle',':')
        hold off
        title({'\fontsize{16}Normalized (frequency-by-frequency) TFR',['\fontsize{12}Dataset ',num2str(subj_i),', Trial class ',num2str(classes(cls_i))]})
        
        % Plot 10 randomly sampled TFR trials
        for trl_i=1:numSampTrials
            % Raw TFR trial
            rTrial_sub(trl_i) = subplot('Position',[0.08 0.75-(0.065*trl_i) 0.37 0.05]);
            %clims = [0 mean(eventThr(eventBand_inds))*1.5]; %Standardize upper limit of spectrogram scaling using the average event threshold
            %imagesc([tVec(1) tVec(end)],eventBand,TFR(eventBand_inds(1):eventBand_inds(end),:,trial_inds(randTrial_inds(trl_i))),clims)
            imagesc([tVec(1) tVec(end)],eventBand,TFR(eventBand_inds(1):eventBand_inds(end),:,trial_inds(randTrial_inds(trl_i))))
            x_tick_labels = get(gca,'xticklabels');
            x_tick = get(gca,'xtick');
            set(gca,'xtick',x_tick);
            set(gca,'ticklength',[0.0075 0.025])
            set(gca,'xticklabel',[])
            set(gca,'ytick',eventBand)
            rTrial_pos = get(gca,'position');
            colormap jet
            cb = colorbar;
            cb.Position = [rTrial_pos(1)+rTrial_pos(3)+0.01 rTrial_pos(2) 0.008 rTrial_pos(4)];
            
            % Overlay locations of event peaks and the waveform corresponding with each trial
            hold on
            plot(maximaTiming(trialInd==trial_inds(randTrial_inds(trl_i))),maximaFreq(trialInd==trial_inds(randTrial_inds(trl_i))),'w.') %Add points at event maxima
            yyaxis right
            plot(tVec,timeseries{subj_i}(:,trial_inds(randTrial_inds(trl_i))),'w')
            %set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            hold off
            
            % Normalized TFR trial
            nTrial_sub(trl_i) = subplot('Position',[0.53 0.75-(0.065*trl_i) 0.37 0.05]);
            clims = [0 specEv_struct(subj_i).Events.ThrFOM*1.5]; %Standardize upper limit of spectrogram scaling using the FOM threshold
            imagesc([tVec(1) tVec(end)],eventBand,TFR_norm(eventBand_inds(1):eventBand_inds(end),:,trial_inds(randTrial_inds(trl_i))),clims)
            x_tick_labels = get(gca,'xticklabels');
            x_tick = get(gca,'xtick');
            set(gca,'xtick',x_tick);
            set(gca,'ticklength',[0.0075 0.025])
            set(gca,'xticklabel',[])
            set(gca,'ytick',eventBand)
            nTrial_pos = get(gca,'position');
            colormap jet
            
            % Overlay locations of event peaks and the waveform corresponding with each trial
            hold on
            plot(maximaTiming(trialInd==trial_inds(randTrial_inds(trl_i))),maximaFreq(trialInd==trial_inds(randTrial_inds(trl_i))),'w.') %Add points at event maxima
            yyaxis right
            plot(tVec,timeseries{subj_i}(:,trial_inds(randTrial_inds(trl_i))),'w')
            %set(gca,'ytick',[])
            %set(gca,'yticklabel',[])
            hold off
        end
        
        subplot(nTrial_sub(ceil(numel(rTrial_sub)/2)))
        yyaxis right
        ylabel('Waveform ampl.')
        
        subplot(rTrial_sub(end))
        %cb = colorbar;
        %cb.Position = [rTrial_pos(1)+rTrial_pos(3)+0.01 rTrial_pos(2) 0.008 rTrial_pos(4)];
        set(gca,'ticklength',[0.0075 0.025])
        set(gca,'xticklabel',x_tick_labels)
        xlabel('s')
        
        subplot(nTrial_sub(end))
        cb = colorbar;
        cb.Position = [nTrial_pos(1)+nTrial_pos(3)+0.02 nTrial_pos(2) 0.008 nTrial_pos(4)];
        set(gca,'ticklength',[0.0075 0.025])
        set(gca,'xticklabel',x_tick_labels)
        xlabel('s')

        figureName = strcat('./test_results/matlab/prestim_humandetection_600hzMEG_subject', num2str(subj_i), '_class_', num2str(classes(cls_i)), '.png');
        saveas(gcf,figureName);

    end
end

% Event feature probability histograms (see Figure 5 in Shin et al. eLife 2017)
features = {'eventnumber','maximapowerFOM','duration','Fspan'}; %Fields within specEv_struct
feature_names = {'event number','event power (FOM)','event duration (ms)','event F-span (Hz)'}; %Full names describing each field
figure
for feat_i=1:numel(features)
    feature_agg = [];
    for subj_i=1:numSubj
        % Feature-specific considerations
        if isequal(features{feat_i},'eventnumber')
            feature_agg = [feature_agg; specEv_struct(subj_i).TrialSummary.TrialSummary.(features{feat_i})];
        else
            if isequal(features{feat_i},'duration')
                feature_agg = [feature_agg; specEv_struct(subj_i).Events.Events.(features{feat_i}) * 1000]; %Note: convert from s->ms
            else
                feature_agg = [feature_agg; specEv_struct(subj_i).Events.Events.(features{feat_i})];
            end
        end
    end
    
    % Don't plot if no events occurred
    if isequal(features{feat_i},'eventnumber') && nnz(feature_agg)==0
        close 
        break
    end
    
    % Calculate probability of aggregate (accross subjects/sessions) and 
    % standardize bins
    [featProb_agg,bins] = histcounts(feature_agg,'Normalization','probability'); 
    
    % Correct to show left-side dropoff of histogram if applicable
    if bins(2)-(bins(2)-bins(1))/2>0
        bins = [bins(1)-(bins(2)-bins(1)),bins];
        featProb_agg = histcounts(feature_agg,bins,'Normalization','probability');
    end
    featProb_agg(isnan(featProb_agg)) = 0; %Correct for NaN values resulting from dividing by 0 counts
    
    % Calculate and plot for each subject
    subplot(numel(features),1,feat_i)
    for subj_i=1:numSubj
        % Feature-specific considerations
        if isequal(features{feat_i},'eventnumber')
            feature = specEv_struct(subj_i).TrialSummary.TrialSummary.(features{feat_i});
            %classLabels = specEv_struct(subj_i).TrialSummary.TrialSummary.classLabels;
        else
            feature = specEv_struct(subj_i).Events.Events.(features{feat_i});
            if isequal(features{feat_i},'duration')
                feature = feature*1000; %Convert from s->ms
            end
            %classLabels = specEv_struct(subj_i).Events.Events.classLabels;
        end

        % Calculate probability for each subject
        featProb = histcounts(feature,bins,'Normalization','probability');
        featProb(isnan(featProb)) = 0; %Correct for NaN values resulting from dividing by 0 counts
        hold on
        %plot(bins(2:end)-diff(bins)/2,featProb)
        histogram('BinEdges',bins,'BinCounts',featProb,'DisplayStyle','stairs')
        hold off
    end
    
    % Finally, plot aggregate probability for each feature
    hold on
    %plot(bins(2:end)-diff(bins)/2,featProb_agg,'k-','LineWidth',2)
    histogram('BinEdges',bins,'BinCounts',featProb_agg,'EdgeColor','k','LineStyle','-','LineWidth',2,'DisplayStyle','stairs')
    hold off
    xlim([bins(2)-(bins(2)-bins(1))/2,bins(find(cumsum(featProb_agg)>=0.95,1)+1)]) %Lower limit: smallest mid-bin; upper limit: 95% cdf interval
    xlabel(feature_names{feat_i})
    ylabel('probability')
end

end
