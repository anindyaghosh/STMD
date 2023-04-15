clear all
close all
% 
load_dir='D:\temp\';
load([load_dir 'temp_lost19_09_18.mat']);

paramfile=load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_09_16.mat');

% Work out the average time to failure for different inner_ix. Separate
% facilitation tracker from other

numgind=25488;
num_outer = numel(unique(paramfile.outer_ix));

fac_outer = false(num_outer,1);
prob_outer = false(num_outer,1);

for k=1:num_outer
    prob_outer(k) = strcmp(paramfile.outer_var.sets.tracker{k},'prob');
    fac_outer(k) = strcmp(paramfile.outer_var.sets.tracker{k},'facilitation');
end

fac_outer_ix = find(fac_outer);
prob_outer_ix = find(prob_outer);

mask_fac = ismember(paramfile.outer_ix,fac_outer_ix);
mask_prob = ismember(paramfile.outer_ix,prob_outer_ix);

fac_results = t_lost_it(mask_fac,:);
prob_results = t_lost_it(mask_prob,:);

% Remove rows with NaN
% 
% reduce_fac_mask = isnan(sum(fac_results,2));
% reduce_prob_mask = isnan(sum(prob_results,2));
% 
% reduced_prob_results = prob_results(~reduce_prob_mask,:);

prob_mean = mean(prob_results,2);

plot(sort(prob_mean));

% 