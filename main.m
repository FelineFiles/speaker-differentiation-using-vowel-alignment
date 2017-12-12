%clear all
clc;

tic

%% Define the parameter

NumFeatures = 138;        % Number of features 

FILE_DIR = './';

EXTRACT_SPEECH_SEGMENT = false;;
EXTRACT_PITCH = false;;
NUM_SEGMENTS = 3;

%--------------------------
% Get label
%--------------------------

label = load('dataLabel.mat');

dataLabel = cat(1,label.FEMALE,label.MALE);

crossInd = load('crossValIdx.mat');
crossValIdx = cat(1,crossInd.FEMALE,crossInd.MALE);
crossValIdx = crossValIdx==1;
NumPairs = size(dataLabel,1);
NumFolds = size(crossValIdx,2);


% %%
% %--------------------------
% % Prepare for cross-validation
% %--------------------------

fileNames = unique(dataLabel(:,1:2));
nFile = size(fileNames,1);

%% Get the indices of the speech segments we are analyzing
if EXTRACT_SPEECH_SEGMENT
    segment_loc = NaN(nFile, NUM_SEGMENTS, 2);
    for file_num = 1:nFile
        fprintf(['Extracting vowel location from file ' fileNames{file_num} '\n']);
        [snd,Fs] = audioread([FILE_DIR 'WavData/' fileNames{file_num} ]);
        [total_time, segment_loc(file_num,:,:)] ...
            = getWordBoundaryIndices(snd, Fs);
        fprintf(['%i of %i [from: %f to %f ] %f ' fileNames{file_num} '\n\n'],...
            file_num, nFile, segment_loc(file_num,1,1), segment_loc(file_num,1,2),total_time );
    end;
    segment_loc = round(segment_loc*Fs);
    save([FILE_DIR 'Features/segment_location.mat'], 'segment_loc', '-v7.3');
end;

%% Extract Pitch
if EXTRACT_PITCH
    load([FILE_DIR 'Features/segment_location.mat']);
    
    pitches = NaN(nFile,NUM_SEGMENTS);
    pitch = NaN(NUM_SEGMENTS, 1);
    for file_num = 1:nFile
        fprintf(['Extracting pitch from file ' fileNames{file_num} ' %i of %i \n'],...
            file_num, nFile);
        [snd,Fs] = audioread([FILE_DIR 'WavData/' fileNames{file_num} ]); 
        pitch_track = fast_mbsc_fixedWinlen_tracking(snd, Fs);
        snd_len = numel(snd); 
        pitch_track_len = length(pitch_track);
        for seg_num=1:NUM_SEGMENTS;
            n1 = round(segment_loc(file_num,seg_num,1)*pitch_track_len/snd_len);
            n2 = round(segment_loc(file_num,seg_num,2)*pitch_track_len/snd_len);
            seg_pitch = pitch_track(n1:n2);
            pitches(file_num,seg_num) = nanmean(seg_pitch(seg_pitch>0));
        end;
    end;
    save([FILE_DIR 'Features/pitch.mat'], 'pitches', '-v7.3');
end;

%% Extract the rest of the features
feature_matrix = NaN(nFile, NumFeatures-NUM_SEGMENTS);
load([FILE_DIR 'Features/pitch.mat']);
load([FILE_DIR 'Features/segment_location.mat']);

for file_num = 1:nFile
    fprintf(['Extracting features from file ' fileNames{file_num} ' %i of %i \n'],...
        file_num, nFile);
    [snd,Fs] = audioread([FILE_DIR 'WavData/' fileNames{file_num} ]);
    [F, A, MFCC] = extract_feature(snd, shiftdim(segment_loc(file_num,:,:),1), Fs, 12 );    
    feature_matrix(file_num,:) = [F(:); A(:); MFCC(:)]';    
end

%% Merge the pitch with the rest of the features
design_matrix = [pitches, feature_matrix];

%% Normalize the features
design_matrix = bsxfun(@minus, design_matrix, mean(design_matrix, 1));
design_matrix = bsxfun(@rdivide, design_matrix, max(abs(design_matrix), [], 1));

%% PCA
 [coeff, scored, ~, ~, explained] = pca(design_matrix);
 plot(explained);
% design_matrix = scored(:, :);
% NumFeatures = 92;

%%
save([FILE_DIR 'Features/design_matrix.mat'], 'design_matrix', '-v7.3');

%% Save the features to a file
for file_num = 1:nFile
    features = design_matrix(file_num, :);
    save([FILE_DIR 'Features/' fileNames{file_num}(1:end-3) 'mat'], 'features', '-v7.3');
end

%% Load the features

x = NaN*ones(NumPairs, NumFeatures);  % features
y = NaN*ones(NumPairs, 1);  % variable to predict
z = NaN*ones(NumPairs, 1);  % class label

for n=1:NumPairs
    waitbar(n/NumPairs)
%     
    sd1 = dataLabel{n,1};
    sd2 = dataLabel{n,2};

    %--------------------------
    % Load feature
    %--------------------------
     
    feat1 = load([FILE_DIR 'Features/' sd1(1:end-3) 'mat']);
    feat2 = load([FILE_DIR 'Features/' sd2(1:end-3) 'mat']);     
    
    %--------------------------
    % Save into variables
    %--------------------------
    x(n, :) = abs(feat1.features - feat2.features); % Use the difference between mean pitch
    z(n) = dataLabel{n,3}; % intra-speaker indicator
end

%% Classify

rmsErr = NaN*ones(NumFolds,1);
errRate= NaN*ones(NumFolds,1);
for n=1:NumFolds
% features 

    x_train = x(~crossValIdx(:,n),:);
    x_test  = x(crossValIdx(:,n),:);
    
%     
% intra-speaker indication label 
    
    z_train = z(~crossValIdx(:,n));
    z_test  = z(crossValIdx(:,n));

    %--------------------------
    % Random Forest Classifier
    %--------------------------
    NUM_TREES = 500;
    
%     t_template = templateTree('Surrogate','All', 'MaxNumSplits', 1);
%     forest = fitensemble(x_train, z_train,'AdaBoostM1',NUM_TREES,t_template);
    t_template = templateTree('Surrogate','All');%, 'MaxNumSplits', 2);
    forest = fitensemble(x_train, z_train,'bag',NUM_TREES,t_template, 'Type', 'Class');
    
    z_test_hat = predict(forest,x_test);
    
    err = z_test_hat ~= z_test;
    errRate(n) = sum(err)/length(z_test);
end
fprintf('averaged classification error = %.2f %% \n', 100* mean(errRate));
toc

