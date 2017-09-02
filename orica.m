% orica()  - Perform Online Recursive Independent Component Analysis (ORICA) decomposition
%            of input data with optional Online Recursive Least Square Whitening.  
% Usage:
%         >> [weights,sphere] = orica(data); % train using defaults 
%    else
%         >> [weights,sphere] = orica(data,'Key1',Value1',...);
% Input:
%   data     = input data (chans-by-samples)
%
% Optional Keywords [argument]:
% 'weights'     = [W] initial weight matrix     (default -> eye())
% 'sphering'    = ['online'|'offline'] use online RLS whitening method or pre-whitening
% 'numpass'     = [N] number of passes over input data
% 'block_ica'   = [N] block size for ORICA (in samples)
% 'block_white' = [N] block size for online whitening (in samples)
% 'forgetfac'   = ['cooling'|'constant'|'adaptive'] forgetting factor profiles
%                 'cooling': monotonically decreasing, for relatively stationary data
%                 'constant': constant, for online tracking non-stationary data.
%                 'adaptive': adaptive based on Nonstatinoarity Index (in dev)
%                 See reference [2] for more information.
% 'localstat'   = [f] local stationarity (in number of samples) corresponding to 
%                 constant forgetting factor at steady state
% 'ffdecayrate' = [0<f<1] decay rate of (cooling) forgetting factor (default -> 0.6)  
% 'nsub'        = [N] number of subgaussian sources in EEG signal (default -> 0)
%                 EEG brain sources are usually supergaussian
%                 Subgaussian sources are motstly artifact or noise
% 'evalconverg' = [0|1] evaluate convergence such as Non-Stationarity Index
% 'verbose'     = ['on'|'off'] give ascii messages  (default -> 'off')
%
% Output:
%   weights  = ICA weight matrix (comps,chans)
%   sphere   = data sphering matrix (chans,chans)
%              Note that unmixing_matrix = weights*sphere
%
% Reference:
%       [1] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Real-time 
%       adaptive EEG source separation using online recursive independent
%       component analysis," IEEE Transactions on Neural Systems and 
%       Rehabilitation Engineering, 2016.
%
%       [2] S.-H. Hsu, L. Pion-Tanachini, T.-P Jung, and G. Cauwenberghs, 
%       "Tracking non-stationary EEG sources using adaptive online 
%       recursive independent component analysis," in IEEE EMBS, 2015.
% 
%       [3] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Online
%       recursive independent component analysis for real-time source
%       separation of high-density EEG," in IEEE EMBS, 2014.
%
% Author:
%       Sheng-Hsiou (Shawn) Hsu, SCCN, UCSD.
%       shh078@ucsd.edu

function [weights,sphere] = orica(data,varargin)

if nargin < 1
    help orica  
    return
end

[nChs,nPts] = size(data);

%
%%%%%%%%%%%%%%% declare default parameters %%%%%%%%%%%%%%%
%
% options
numPass = 1;
verbose = true;

% Parameters for data whitening
onlineWhitening = false; % Run online RLS whitening prior to ORICA. Suggested for online processing.
blockSizeWhite = 8; % L_{white}: Block size for online whitening block update. Suggested value: 4 to 8, depending on the noise of the data. Larger the noise, larger the block size.

% Parameters and options for online recursive ICA.
blockSizeICA = 8; % L_{ICA}: Block size for online ICA block update. Guideline: if signal is relatively stationary increasing blockSize will speed up runtime without sacrificing too much performance.
numSubgaussian = 0; % Number of subgaussian sources in EEG signal. EEG brain sources are usually supergaussian. Subgaussian sources are motstly artifact or noise.

% strategies and parameters for setting up the forgetting factors.
adaptiveFF.profile = 'cooling';
adaptiveFF.tau_const = Inf; % unit: samples
% pars for cooling ff
adaptiveFF.gamma = 0.6;
adaptiveFF.lambda_0 = 0.995; 
% pars for adaptive ff
adaptiveFF.decayRateAlpha = 0.02;
adaptiveFF.upperBoundBeta = 1e-3;
adaptiveFF.transBandWidthGamma = 1;
adaptiveFF.transBandCenter = 5;
adaptiveFF.lambdaInitial = 0.1;

% Evaluate convergence such as Non-Stationarity Index (NSI).
evalConvergence.profile = true;
evalConvergence.leakyAvgDelta = 0.01; % Leaky average value (delta) for computing non-stationarity index (NSI). NSI = norm(Rn), where Rn = (1-delta)*Rn + delta*(I-yf^T).
evalConvergence.leakyAvgDeltaVar = 1e-3; % Leaky average value (delta) for computing variance of source activation. Var = (1-delta)*Var + delta*variance.

%
%%%%%%%%%%%%%%% Collect keywords and values from argument list %%%%%%%%%%%%%%%
%
if (nargin> 1 & rem(nargin,2) == 0)
    fprintf('orica(): Even number of input arguments???')
    return
end
for i = 1:2:length(varargin) % for each Keyword
    Keyword = varargin{i};
    Value = varargin{i+1};
    if ~isstr(Keyword)
        fprintf('orica(): keywords must be strings')
        return
    end
    Keyword = lower(Keyword); % convert upper or mixed case to lower

    if strcmp(Keyword,'weights') || strcmp(Keyword,'weight') || strcmp(Keyword,'icaweights')
        if isstr(Value)
            fprintf('orica(): weights value must be a weight matrix or sphere')
            return
        else
            weights = Value;
        end
    elseif strcmp(Keyword,'whitening') || strcmp(Keyword,'white') ...
            || strcmp(Keyword,'sphering') || strcmp(Keyword,'sphere')
        Value = lower(Value);
        if ~isstr(Value)
            fprintf('orica(): whitening value must be ''offline'' or ''online''')
            return
        end
        if strcmp(Value,'offline')
            onlineWhitening = false;
        elseif strcmp(Value,'online')
            onlineWhitening = true;
        else
            fprintf('orica(): whitening value must be ''offline'' or ''online''')
            return
        end
    elseif strcmp(Keyword,'npass') || strcmp(Keyword,'numpass')
        if isstr(Value)
            fprintf('orica(): number of passes must be an integer')
            return
        end
        numPass = floor(Value);
    elseif strcmp(Keyword,'block_ica') || strcmp(Keyword,'blocksize_ica')
        if isstr(Value)
            fprintf('orica(): block size value for online ICA must be a number')
            return
        end
        blockSizeICA = floor(Value);
    elseif strcmp(Keyword,'block_white') || strcmp(Keyword,'blocksize_white')
        if isstr(Value)
            fprintf('orica(): block size value for online ICA must be a number')
            return
        end
        if ~onlineWhitening
            fprintf('orica(): use online whitening')            
        end
        blockSizeWhite = floor(Value);
    elseif strcmp(Keyword,'ff') || strcmp(Keyword,'forgetfac')
        Value = lower(Value);
        if ~isstr(Value)
            fprintf('orica(): forgetting factor must be ''cooling'', ''constant'', or ''adaptive''')
            return
        elseif strcmp(Value,'cooling') || strcmp(Value,'constant') || strcmp(Value,'adaptive')
            adaptiveFF.profile = Value;
        else
            fprintf('orica(): method not specified, choose ''cooling'', ''constant'', or ''adaptive''')
            return
        end
    elseif strcmp(Keyword,'localstat') || strcmp(Keyword,'tau_const')
        if isstr(Value)
            fprintf('orica(): local stationarity value must be a number')
            return
        end
        adaptiveFF.tau_const = Value;
    elseif strcmp(Keyword,'ffdecayrate') || strcmp(Keyword,'gamma')
        if isstr(Value)
            fprintf('orica(): decay rate (gamma) must be a number')
            return
        end
        adaptiveFF.gamma = Value;        
    elseif strcmp(Keyword,'nsub') || strcmp(Keyword,'numsubgaussian')
        if isstr(Value)
            fprintf('orica(): number of subgaussian sources must be an integer')
            return
        end
        numSubgaussian = floor(Value);
    elseif strcmp(Keyword,'evalconverg')
        if isstr(Value)
                fprintf('orica(): evaluate convergence must be ''0'' or ''1''')
                return
        else
            if Value == 0
                evalConvergence.profile = false;
            elseif Value == 1
                evalConvergence.profile = true;
            else
                fprintf('orica(): evaluate convergence must be ''0'' or ''1''')
                return
            end
        end
    elseif strcmp(Keyword,'verbose') 
        if ~isstr(Value)
            fprintf('orica(): verbose flag value must be on or off')
            return
        elseif strcmp(Value,'on'),
            verbose = 1; 
        elseif strcmp(Value,'off'),
            verbose = 0; 
        else
            fprintf('orica(): verbose flag value must be on or off')
            return
        end
    else
       fprintf('orica(): unknown flag')
       return
    end
end

%
%%%%%%%%%%%%%%% initialize state structure %%%%%%%%%%%%%%%
%
if ~exist('state','var') || isempty(state)

    if exist('weights','var')
        state.icaweights = weights;
    else
        state.icaweights = eye(nChs); % should use infinity for convergence
    end

    if onlineWhitening
        state.icasphere = eye(nChs);
    end

    state.lambda_k      = zeros(1,blockSizeICA);   % readout lambda
    state.minNonStatIdx = []; 
    state.counter       = 0; % time index counter, used to keep track of time for computing lambda

    if strcmp(adaptiveFF.profile,'cooling') || strcmp(adaptiveFF.profile,'constant')
        adaptiveFF.lambda_const  = 1-exp(-1/(adaptiveFF.tau_const)); % steady state constant lambda
    end
    
    if evalConvergence.profile
        state.Rn = [];
        state.nonStatIdx = [];
    end
    
    % sign of kurtosis for each component: true(supergaussian), false(subgaussian)
    state.kurtsign      = ones(nChs,1) > 0;      % store kurtosis sign for each channels
    if numSubgaussian ~= 0
        state.kurtsign(1:numSubgaussian) = false;
    end 
    
end


%
%%%%%%%%%%%%%%%%%%%% sphere / whiten data %%%%%%%%%%%%%%%%%%%%
%
if ~onlineWhitening     % pre-whitening
    if verbose, fprintf('Use pre-whitening method.\n'); end
    state.icasphere = 2.0*inv(sqrtm(double(cov(data')))); % find the "sphering" matrix = spher()
else                    % Online RLS Whitening
    if verbose, fprintf('Use online whitening method.\n'); end
end

% whiten / sphere the data
data = state.icasphere * data;


%
%%%%%%%%%%%%%%%%%%%% Online Recusive ICA %%%%%%%%%%%%%%%%%%%%
%

% divide data into blocks for online block update
numBlock = floor(nPts/min(blockSizeICA,blockSizeWhite));

if verbose
    printflag = 0;
    switch adaptiveFF.profile
        case 'cooling'
            fprintf('Running ORICA with cooling forgetting factor...\n');  
        case 'constant'
            fprintf('Running ORICA with constant forgetting factor...\n');  
        case 'adaptive'
            fprintf('Running ORICA with adaptive forgetting factor...\n');  
    end
    tic; 
end

for it = 1 : numPass
    for bi = 0 : numBlock-1
        
        dataRange = 1 + floor(bi*nPts/numBlock) : min(nPts, floor((bi+1)*nPts/numBlock));
        if onlineWhitening
            state = dynamicWhitening(data(:,dataRange), dataRange, state, adaptiveFF);
            data(:,dataRange) = state.icasphere * data(:,dataRange);
        end
        state = dynamicOrica(data(:, dataRange), state, dataRange, adaptiveFF, evalConvergence);
        
        if verbose
            if printflag < floor(10*((it-1)*numBlock+bi)/numPass/numBlock); 
                printflag = printflag + 1;
                fprintf(' %d%% ', 10*printflag);
            end
        end

    end 
end
if verbose, fprintf('Finished.\nEllapsed time: %f sec.\n',toc); end

% output weights and sphere matrices
% TODO: consider averaging
weights = state.icaweights;
sphere = state.icasphere;

end


function state = dynamicWhitening(blockdata, dataRange, state, adaptiveFF)

    nPts = size(blockdata,2);

    % define adaptive forgetting rate: lambda
    switch adaptiveFF.profile
        case 'cooling'
            lambda = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0);
            if lambda(1) < adaptiveFF.lambda_const
                lambda = repmat(adaptiveFF.lambda_const,1,nPts); 
            end
        case 'constant'
            lambda = repmat(adaptiveFF.lambda_const,1,nPts);
        case 'adaptive'
            lambda = repmat(state.lambda_k(end),1,nPts); % using previous adaptive lambda_k from adaptiveOrica
    end
        
    % update sphere matrix using online RLS whitening block update rule
    v = state.icasphere * blockdata; % pre-whitened data 
    lambda_avg = 1 - lambda(ceil(end/2));    % median lambda
    QWhite = lambda_avg/(1-lambda_avg) + trace(v' * v) / nPts;
    state.icasphere = 1/lambda_avg * (state.icasphere - v * v' / nPts / QWhite * state.icasphere);

end


function state = dynamicOrica(blockdata, state, dataRange, adaptiveFF, evalConvergence, nlfunc)

if nargin < 6
    nlfunc = []; 
end

% initialize
[nChs, nPts] = size(blockdata);
f            = zeros(nChs, nPts);

% compute source activation using previous weight matrix
y = state.icaweights * blockdata;

% choose nonlinear functions for super- vs. sub-gaussian 
if isempty(nlfunc)
    f(state.kurtsign,:)  = -2 * tanh(y(state.kurtsign,:));                        % Supergaussian
    f(~state.kurtsign,:) = 2 * tanh(y(~state.kurtsign,:));                        % Subgaussian
else
    f = nlfunc(y);
end

% compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
if evalConvergence.profile
    modelFitness = eye(nChs)+y*f'/nPts;
    variance = blockdata.*blockdata;
    if isempty(state.Rn)
        state.Rn = modelFitness;
    else
        state.Rn = (1-evalConvergence.leakyAvgDelta)*state.Rn + evalConvergence.leakyAvgDelta*modelFitness; % !!! this does not account for block update!
    end
    state.nonStatIdx = norm(state.Rn,'fro');
end

% compute the forgetting rate
switch adaptiveFF.profile
    case 'cooling'
        state.lambda_k = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0);
        if state.lambda_k(1) < adaptiveFF.lambda_const
            state.lambda_k = repmat(adaptiveFF.lambda_const,1,nPts); 
        end
        state.counter = state.counter + nPts;
    case 'constant'
        state.lambda_k = repmat(adaptiveFF.lambda_const,1,nPts);
    case 'adaptive'
        if isempty(state.minNonStatIdx)
            state.minNonStatIdx = state.nonStatIdx; 
        end
        state.minNonStatIdx = max(min(state.minNonStatIdx, state.nonStatIdx),1);
        ratioOfNormRn = state.nonStatIdx/state.minNonStatIdx;
        state.lambda_k = genAdaptiveFF(dataRange,state.lambda_k,adaptiveFF.decayRateAlpha,adaptiveFF.upperBoundBeta,adaptiveFF.transBandWidthGamma,adaptiveFF.transBandCenter,ratioOfNormRn);
end

% update weight matrix using online recursive ICA block update rule
lambda_prod = prod(1./(1-state.lambda_k));
Q = 1 + state.lambda_k .* (dot(f,y,1)-1);
state.icaweights = lambda_prod * (state.icaweights - y * diag(state.lambda_k./Q) * f' * state.icaweights);

% orthogonalize weight matrix 
[V,D] = eig(state.icaweights * state.icaweights');
state.icaweights = V/sqrt(D)*V' * state.icaweights; 

end


function lambda = genCoolingFF(t,gamma,lambda_0)
    % lambda = lambda_0 / sample^gamma
    lambda = lambda_0 ./ (t .^ gamma);
end


function lambda = genAdaptiveFF(dataRange,lambda,decayRateAlpha,upperBoundBeta,transBandWidthGamma,transBandCenter,ratioOfNormRn)
% lambda = lambda - DecayRate*lambda + UpperBound*Gain*lambda^2
% Gain(z) ~ tanh((z/z_{min} - TransBandCenter) / TransBandWidth)
    gainForErrors = upperBoundBeta*0.5*(1+tanh((ratioOfNormRn-transBandCenter)/transBandWidthGamma));
    f = @(n) (1+gainForErrors).^n * lambda(end) - decayRateAlpha*((1+gainForErrors).^(2*n-1)-(1+gainForErrors).^(n-1))/gainForErrors*lambda(end)^2;
    lambda = f(1:length(dataRange));    
end
