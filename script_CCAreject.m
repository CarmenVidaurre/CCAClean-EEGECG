%% CCA-based cardiac artefact removal
%Copyright (C) 2026 Carmen Vidaurre

%This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.

%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%GNU General Public License for more details.

% Uses canoncorr (Statistics and Machine Learning Toolbox) together with
% REJECT_CCA_COMPONENTS_V2.M, following Section 2(c)-(d)
% (Eqs. 2.1-2.4) of manuscript: "Canonical Correlation Analysis and Multi-Channel...
% Cardiography Improve Artefact Cleaning in Heartbeat-Locked Analyses".
% C Vidaurre, M Azanova et al. 2026, Philosophical Transactions of the 
% Royal Society B. 
%
% CONVENTIONS:
%   EEG_data : [time x nEEGchan]  
%   ECG_data : [time x nECGchan]  
%  By C. Vidaurre. Please cite our paper if you use this code.

% --- 1. Canonical Correlation Analysis ---
% canoncorr ranks components by descending canonical correlation
% (R(1) = highest).
[Wx, Wy, R] = canoncorr(EEG_data, ECG_data);

% --- 2. EEG covariance ---
Sigma_XX = cov(EEG_data);

% --- 3. Iteratively remove the top-k correlated components ---
nComp = size(Wx, 2);            % = min(nEEGchan, nECGchan)
EEG_clean = cell(1, nComp);     % EEG_clean{k}: EEG after removing top-k comps
P_remove  = cell(1, nComp);     % projector onto the removed subspace at each k
A_reject  = cell(1, nComp);     % patterns of the removed components at each k

for k = 1:nComp
    [EEG_clean{k}, P_remove{k}, A_reject{k}] = ...
        reject_cca_components(EEG_data, Wx, Sigma_XX, 1:k);
end



% EEG_clean{k} now holds the continuous EEG with the k most
% cardiac-correlated CCA components removed (k = 1 ... nComp).


function [X_clean, P_remove, A_reject] = reject_cca_components(X, Wx, Sigma_XX, reject_idx)

% X:           [times x nchan]
% Wx:          [nchan x ncomp]
% Sigma_XX:    [nchan x nchan]
% reject_idx:  components to remove

if nargin < 4 || isempty(reject_idx)
    reject_idx = [];
end

% Dual/mixing matrix
A = Sigma_XX * Wx / (Wx' * Sigma_XX * Wx);

% Rejected subspace
W_reject = Wx(:, reject_idx);
A_reject = A(:, reject_idx);

% Oblique projector onto rejected subspace
P_remove = W_reject * A_reject';

% Artifact estimate
X_artifact = X * P_remove;

% Subtract artifact
X_clean = X - X_artifact;

end