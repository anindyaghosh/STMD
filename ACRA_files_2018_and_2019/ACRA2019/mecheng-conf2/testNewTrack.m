clear all
close all

param_settings_file = 'D:\simfiles\conf2\trackerinfo\TrackerParamFile19_09_16.mat';
paramfile = load(param_settings_file);

numgind = 18576;
mask = false(numgind,1);

full_tracker_num = paramfile.outer_var.sets.tracker_num(paramfile.outer_ix);
full_saccades = paramfile.outer_var.sets.do_saccades(paramfile.outer_ix);
full_predictive = paramfile.outer_var.sets.do_predictive_turn(paramfile.outer_ix);
full_distractors = paramfile.outer_var.sets.draw_distractors(paramfile.outer_ix);

do_gind = find(full_tracker_num == 2 &...
    full_saccades == true &....
    full_predictive == false &...
    full_distractors == false,1,'first');

groupname='testit';
hours_to_run=1;
tic
in_toc = toc;

TrackerModel19_09_16(param_settings_file, do_gind, groupname, hours_to_run, in_toc)




 