% This is a test script for running orica()
% Start EEGLAB for data analysis. 
% EEGLAB can be downloaded at https://bitbucket.org/sccn_eeglab/eeglab
eeglab
close

%% load sample EEG dataset
EEG = pop_loadset('SIM_STAT_16ch_3min.set');
EEG.icawinv_true = EEG.etc.LFM{1};
EEG.icawinv = EEG.icawinv_true; EEG.icaweights = pinv(EEG.icawinv); EEG.icasphere = eye(EEG.nbchan); EEG = eeg_checkset(EEG);

[nChs, nPts] = size(EEG.data);
EEG.icaweights = [];    
EEG.icasphere  = [];    
EEG.icawinv    = [];    
EEG.icaact     = [];

%% simulate online processing with ORICA
[EEG.icaweights, EEG.icasphere] = orica(EEG.data,'numpass',1, ...
	'sphering','online','block_white',8,'block_ica',8,'nsub',0, ...
	'forgetfac','cooling','localstat',Inf,'ffdecayrate',0.6, ...
	'evalconverg',1,'verbose','on');

EEG.icawinv = pinv(EEG.icaweights*EEG.icasphere);

%% evaluation
% compute mutual information reduction
h0 = getent2(EEG.data);
y_true = pinv(EEG.icawinv_true) * EEG.data;
y_orica = EEG.icaweights * EEG.icasphere * EEG.data;
mir_orica = sum(h0) - sum(getent2(y_orica)) + sum(log(abs(eig(EEG.icaweights * EEG.icasphere))));
mir_true = sum(h0) - sum(getent2(y_true)) + sum(log(abs(eig(pinv(EEG.icawinv_true)))));

% compute performance index (cross-talk error)
H = EEG.icaweights * EEG.icasphere * EEG.icawinv_true; C = H.^2;
crossTalkError = (EEG.nbchan-sum(max(C,[],1)./sum(C,1))/2-sum(max(C,[],2)./sum(C,2))/2)/(EEG.nbchan-1);

fprintf('Cross talk error: %f (dB) \nMIR: %f(true)\nMIR: %f(orica)\n',db(crossTalkError),mir_true,mir_orica);

%% visualize decomposed scalp maps
[correlation, idx_truth, idx_orica] = matcorr(EEG.icawinv_true', EEG.icawinv', 0, 0);
idx = sortrows([idx_truth, idx_orica],1); [~,idx_corr] = sortrows([idx_truth, correlation],1);
EEG.icawinv = EEG.icawinv(:,idx(:,2)) * diag(sign(correlation(idx_corr)));
pop_topoplot(EEG,0,1:EEG.nbchan);

