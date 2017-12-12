function [ success, middle_index ] = freq_analysis(tea_dv, pass_dv, sil_dv, lpass, hpass, total_time )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
%% Assumptions
init_sil = 0.3;%s
sil_thresh = 15;


success = 0;
middle_index = [0 0; 0 0]; %% 'help' and 'pass'



pass_band_len = length(lpass);
mfcc_len = length(tea_dv);


%%  NN Classifier
init_n = init_sil/total_time*length(sil_dv);
tea_pass_bias = 0; 
% From observation, 'tea's MFCC have a constant bias over 'pass's MFCC, so
% subtracting the bias out
for n = 1:init_n
    tea_pass_bias = tea_pass_bias + tea_dv(n) - pass_dv(n);
end
tea_pass_bias = tea_pass_bias / init_n;

classifier = zeros(mfcc_len,1);
for n = 1:mfcc_len
    if sil_dv(n)+sil_thresh < pass_dv(n) && sil_dv(n)+sil_thresh < tea_dv(n)
        classifier(n) = 0;
    else
        if tea_dv(n) <= pass_dv(n)+tea_pass_bias
            classifier(n) = 1;
        end
        if tea_dv(n) > pass_dv(n)+tea_pass_bias
            classifier(n) = 2;
        end
    end 
end
%% 'Median Filtering'
% Smooth out incongruencies
for n = 2:mfcc_len-1
    if classifier(n-1) == classifier(n+1)
        classifier(n) = classifier(n-1);
    end
end

%% Remove rogue peaks
% Remove Random peaks in Silence
fil_size = 4;
for n = (1+fil_size):(mfcc_len-fil_size)
    [M,F] = mode(classifier(n-fil_size:n+fil_size));

    if M == 0 && F >= fil_size*3/2    
        classifier(n) = 0;
    end
end

%% Find widest peaks
% Remove the leading and trailing silence
t1 = find(classifier,1,'first');
t2 = find(classifier,1,'last');
classifier_trun = classifier(t1:t2);

% We assume that 'helps and 'pass' occurs in this region (n1:n2). The words 'or'
% 'a', and 'evening' shows up in the classifier 50% of the time, but occur
% outside of the truncate region.
n1 = round(0.3*length(classifier_trun));
n2 = round(0.82*length(classifier_trun));

min_n_len = 6;
mi = [0 0];
index = 1;
counter = 0;
for n = n1:n2
    if classifier_trun(n) == 2
        counter = counter + 1;
    else
        if counter > min_n_len
            mi(index,:) = [n-counter, n-1];
            index = index + 1;
        end
        counter = 0;
    end
end
if counter > min_n_len
     if counter > mi(1,2) - mi(1,1) + 1
        mi(index,:) = [n-counter, n-1];
     end
end


%% We expect two or three peaks to show, return the two longest peaks
mi = mi + t1 -1; % Adjusts the length accordingly
if size(mi, 1) == 2 
    success = true;
    middle_index = mi;
end
if size(mi, 1) == 3
    success = true;
    middle_index(1,:) = mi(1,:);
    if mi(2,2)-mi(2,1) > mi(3,2)-mi(3,1)
        middle_index(2,:) = mi(2,:);
    else
        middle_index(2,:) = mi(3,:);
    end
end
t4 = [1:mfcc_len]/mfcc_len * total_time;
t5 = [1:pass_band_len]/pass_band_len * total_time;

%% Tea Extraction

n2 = middle_index(1,1) - t1 + 1;
n1 = round(0.2*length(classifier_trun));
mi = [0 0];
index = 1;
counter = 0;
for n = n1:n2-1
    if classifier_trun(n) == 1
        counter = counter + 1;
    else
         if counter > min_n_len
            mi(index,:) = [n-counter, n-1];
            index = index + 1;
         end
        counter = 0;
    end
end
if counter > min_n_len
     if counter > mi(1,2) - mi(1,1) + 1
        mi(index,:) = [n-counter, n-1];
     end
end
tea_index = [0 0];
if size(mi, 1) == 1
    tea_index = mi;
else
    range_mi = mi(:,2) - mi(:,1);
    [~, max_in] = max(range_mi);
    tea_index = [ mi(max_in(1), 1) mi(max_in(1), 2)];
end
middle_index(3,:) = tea_index;

% figure;
% hold on;
% plot(tea_dv);
% plot(pass_dv);
% plot(sil_dv);
% plot(middle_index(1,1), 0, 'b*');
% plot(middle_index(1,2), 0, 'b*');
% 
% 
% plot(n1+t1-1, 0, 'g*');
% plot(n2+t1-1, 0, 'g*');
% plot(t1-1+ tea_index(1,1), 0, 'r*');
% plot(t1-1+ tea_index(1,2), 0, 'r*');


end

