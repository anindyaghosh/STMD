clear all
close all
clc

load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');

gind_range = 1:numel(outer_ix);

gind=388;

ix=outer_ix(gind);

hfov = outer_var.sets.hfov(ix);
image_name = outer_var.sets.image_name{ix};
tracker = outer_var.sets.tracker{ix};
distractors = outer_var.sets.draw_distractors(ix);
saccades = outer_var.sets.do_saccades(ix);
predictive_turn = outer_var.sets.do_predictive_turn(ix);
saccade_duration = inner_var.saccade_duration(gind);
assumed_fwd_vel = inner_var.assumed_fwd_vel(gind);

disp(['Gind = ' num2str(gind)])
disp(['hfov: ' num2str(hfov)])
disp(['image_name: ' image_name])
disp(['tracker: ' tracker])
disp(['distractors: ' num2str(distractors)])
disp(['saccades: ' num2str(saccades)])
disp(['predictive_turn: ' num2str(saccades)])
disp(['saccade_duration: ' num2str(saccade_duration)])
disp(['assumed_fwd_vel: ' num2str(assumed_fwd_vel)])

figure(1);
subplot(2,3,1)
x = outer_var.sets.tracker_num(outer_ix);
plot(x,'.')
title('tracker_num','Interpreter','none')
subplot(2,3,2)
x = outer_var.sets.image_num(outer_ix);
plot(x,'.')
title('image_num','Interpreter','none')
subplot(2,3,3)
x = outer_var.sets.draw_distractors(outer_ix);
plot(x,'.')
title('draw_distractors','Interpreter','none')
subplot(2,3,4)
x = outer_var.sets.do_saccades(outer_ix);
plot(x,'.')
title('do_saccades','Interpreter','none')
subplot(2,3,5)
x = inner_var.saccade_duration;
plot(gind_range,x,'.')
title('saccade_duration','Interpreter','none')
subplot(2,3,6)
x = inner_var.assumed_fwd_vel;
plot(gind_range,x,'.')
title('assumed_fwd_vel','Interpreter','none')