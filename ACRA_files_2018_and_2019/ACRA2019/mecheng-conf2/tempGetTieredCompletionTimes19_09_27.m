clear all
close all

load('D:\temp\TrackerFull19_09_26.mat');

rg = 1:1600;

plot(sort(est_completion(rg)/60/60))

comp_mask = ispresent(rg);

inds=rg(~comp_mask);
the_times=est_completion(inds)/60/60;

low_times_mask = the_times < 1.25;
low_inds = inds(low_times_mask);

high_times_mask = ~low_times_mask;
high_inds = inds(high_times_mask);

save('D:\temp\HighLowTimes19_09_27.mat','low_inds','high_inds');