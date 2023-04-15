clear all
close all

load('E:\PHD\TEST\TrackerParamFile19_10_03.mat');

fixed.fac.output_turn_threshold = fixed.prob.output_threshold_scalar;

this_script = mfilename;

description='This file is the same as TrackerParamFile19_10_03.mat but with a different fixed.fac.output_turn_threshold value';

save('E:\PHD\TEST\TrackerParamFile19_10_06.mat',...
    'belief_vel_fixed','fixed','outer_var',...
    'params_range','params_set','inner_var',...
    'outer_ix','inner_ix','this_script','description')