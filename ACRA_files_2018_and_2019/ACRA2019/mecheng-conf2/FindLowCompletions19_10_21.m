clear all
close all

% Want a list of indices which have small numbers of completed trials

load('D:\temp\temp_lost_it_19_10_22.mat');

b=sum(logs.ispresent,2,'omitnan');

inds_lowcomp = find(b<20);

this_script = mfilename;

save('D:\temp\inds_lowcomp19_10_23.mat','inds_lowcomp','this_script')