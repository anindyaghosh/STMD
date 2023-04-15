clear all
close all
params=load('D:\temp\TrackerParamFile19_09_26.mat');

hfov=params.outer_var.sets.hfov(params.outer_ix);
image_num=params.outer_var.sets.image_num(params.outer_ix);
tracker_num=params.outer_var.sets.tracker_num(params.outer_ix);

draw_distractors=params.outer_var.sets.draw_distractors(params.outer_ix);
do_predictive_turn=params.outer_var.sets.do_predictive_turn(params.outer_ix);
do_saccades=params.outer_var.sets.do_saccades(params.outer_ix);

saccade_duration = params.inner_var.saccade_duration;
assumed_fwd_vel = params.inner_var.assumed_fwd_vel;

mask = hfov == 40 &...
    image_num == 1 & ...
    tracker_num == 1 &...
    ~draw_distractors & ...
    do_predictive_turn &...
    do_saccades &....
    saccade_duration == 0.1;

gind_range=1:5760;
gind=gind_range(mask)