function [ T, F, lpass, hpass, mfcc_arr2 ] = fft_formants(  sig_ds, Fs  )
% T and F are the time and frequency scale of lpass, hpass
%   mfcc_arr2 is the short-time-MFCC
nffT = 1024;
L = 800;


index = 1;
m = round(   (length(sig_ds)/L-3)*3);

freq_arr = zeros(m, nffT/2+1);
win_num = 1;

p = 26;
[x,mc]= melbankm(p,nffT,Fs, 0.005, 0.4);%formerly 0.1814
          
mfcc_arr2 = zeros(1,p);
while index+L < length(sig_ds)
    I0 = index;
    Iend = index + L;
    
    
    sig_buff = sig_ds(I0:Iend-1);
    sig_win = sig_buff.*hamming(L);
    preemph = [1 0.97];
    sig_win = filter(1,preemph,sig_win);
    
    Y = rfft(sig_win,nffT);
    
    z=log(x*abs(Y).^2); 
    mfcc_arr2(win_num, :) = z;
    
    P2 = abs(Y/nffT);
    P1 = P2;
    P1(2:end-1) = 2*P1(2:end-1);
    freq_arr(win_num, :) = P1;
    
    
    win_num = win_num+1;
    index = round(index + L/3);
end


F = Fs*(0:(L/2))/L;

T = (1:win_num-1)/(win_num-1)*length(sig_ds)/Fs;

sig_dur = length(sig_ds) / Fs;
stft = freq_arr;
lpass = zeros(length(stft(:, 1)), 1);

low1 = round(100 / (Fs/2) * (nffT/2+1));
low2 = round(1000/ (Fs/2) * (nffT/2+1));
for n = 1:length(stft(:, 1))
    lpass(n) = sum( stft(n, low1:low2));
end

high1 = round(4000/ (Fs/2) * (nffT/2+1));
high2 = round(6000/ (Fs/2) * (nffT/2+1));
hpass = zeros(length(stft(:, 1)), 1);
for n = 1:length(stft(:, 1))
    hpass(n) = sum( stft(n, high1:high2));
end


%imagesc( T, F, log(stft)' ); %plot the log spectrum
% figure;
%imagesc( (1:size(mfcc_arr2, 1)), mel2frq(mc), mfcc_arr2' ); %plot the log spectrum


end

