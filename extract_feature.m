function [F, A, MFCC] = extract_feature(s, segment_loc, F_s, nc )
%EXTRACT_FEATURE This function extracts the features from speech signal y 
% at location segment_loc.
% Inputs:
%   s: a 1 x signal_length speech signal.
%   segment_loc: a num_phoneme x 2 matrix where the first column stores the
%   beginning and the second column stores the end of each segment.
%   F_s: the sampling frequency in Hertz.
%   nc: number of MFCC coefficients
% Outputs:
%   F: A num_segment x 3 matrix giving the first three formants
%     of each segment.
%   A: A num_segment x 1 vector giving the power of the amplitudes of three 
%     formants of each segment
%   MFCC:  A num_segment x 3*(nc+1) matrix giving the mel cepstrum, the energy,
%     the delta, the delta's energy, the delta-delta, and the delta-delta's
%     energy of each segment

% Get number of segments
num_segment = size(segment_loc, 1);

% Preallocate memory
F = nan(num_segment, 3);
A = nan(num_segment, 3);
MFCC = nan(num_segment, 3*(nc+1)); 

% Loop through each segment and extract the features
for i = 1:num_segment
  
  % Extract the formants 
  ar = lpcauto(s(segment_loc(i,1):segment_loc(i,2)));
  [num_formants, formant_freq, formant_amp, formant_bandwith] ...
      = lpcar2fm(ar);
  F(i,:) = formant_freq(1:3);
  A(i,:) = formant_amp(1:3);
  % Extract the MFCC
  MFCC(i,:) = mean(melcepst(s(segment_loc(i,1):segment_loc(i,2)), F_s, 'EdD', nc),1);
end

