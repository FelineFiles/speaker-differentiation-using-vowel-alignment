function [ dv ] = generateDiff( M, v )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
v = v(:)';

dv = zeros(length(M), 1);
for n = 1: length(M)
    d = M(n,:) - v;
    
    dv(n) = sqrt(sum(d .* d));

end

