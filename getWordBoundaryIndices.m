function [ total_time, vowel_index ] = getWordBoundaryIndices( sig, Fs)
% Returns the indices of 'help' and 'pass'

%   Every vowel in the sentence either looks 'tea's MFCC or 'pass's MFCC
%   Hardcoded for now so we don't spend too much time on feature extraction
tea_mfcc = [    6.0845    6.1965    3.0993    5.4554    2.3657    0.6890   -1.1023   -0.8749   -0.9736   -1.9607   -1.6709   -1.6895 ...
               -1.7910   -0.6788    2.6928    4.7218    3.3300    2.4167    1.4579    0.4561   -2.4202   -5.6288   -5.9712   -5.9314 ...
               -6.1242   -6.1804]; 
          
pass_mfcc = [  4.9344    2.8573    4.5427    3.3231    4.3723    5.6097    6.2055    7.1073    6.1174    5.7320    5.1126    3.4870 ...
                2.1852    2.0722    3.0982    2.1969   -0.1054   -1.2017   -1.5887   -3.0180   -5.6864   -6.5261   -6.3301   -6.1909 ...
            	-6.6969   -6.4138];

 
sig_fil = zeros(length(sig), 1);
%% LPF

order = 3;
for n = order:length(sig)
    sig_fil(n) = sum( sig(n-order+1:n) )/order;
end

sig_ds = sig_fil;

total_time =  length(sig_ds)/Fs;


%% MFCC and High/Low Pass Characteristic


% hpass and lpass are mostly unused for now
[T, F, lpass, hpass, mfcc_arr] = fft_formants(  sig_ds, Fs  );
mfcc_len = length(mfcc_arr);
pass_band_len = length(lpass);

tea_dv = generateDiff(mfcc_arr, tea_mfcc);

pass_dv = generateDiff(mfcc_arr, pass_mfcc);

%% Silence Characteristic

silence = 0.25; % assuming no sig in the first t seconds.
silence_n = round(silence/total_time*mfcc_len);
sil_mfcc = sum(mfcc_arr(1:silence_n, :))/(silence_n);

sil_dv = generateDiff(mfcc_arr, sil_mfcc);

%% formants
% figure;
% freq_arr2 = formant(sig_ds, Fs, 800);
% freq_arr2_len = length(freq_arr2);
% t9 = [1:freq_arr2_len]/freq_arr2_len*total_time;
% plot(t9, freq_arr2(:,1:3));

%% Generate Word Boundaries

[ success, middle_indices ] = freq_analysis(tea_dv, pass_dv, sil_dv,lpass,hpass, total_time);

mi = middle_indices/mfcc_len * total_time;
vowel_index = mi;
 
%% Plotting
% t4 = [1:mfcc_len]/mfcc_len * total_time;
% t5 = [1:pass_band_len]/pass_band_len * total_time;

% figure;
% hold on;
% plot(t5, lpass);
% plot(t5, 11*hpass);
% plot(mi(1,1), 0, 'r*');
% plot(mi(1,2), 0, 'r*');
% plot(mi(2,1), 0, 'r*');
% plot(mi(2,2), 0, 'r*');

end
