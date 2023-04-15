clear all
close all

load('E:\PHD\TEST\TrackerParamFile19_09_16.mat');
load('E:\PHD\TEST\temp_lost19_09_18_all.mat'); % logs

prob_mask = outer_var.sets.tracker_num(outer_ix) == 1;
fac_mask = outer_var.sets.tracker_num(outer_ix) == 2;

prob_results = logs.t_lost_it.first(prob_mask,:);
fac_results = logs.t_lost_it.first(fac_mask,:);

[best_mean_prob,I_prob] = max(mean(prob_results,2));
[best_mean_fac,I_fac] = max(mean(fac_results,2));

% Find the settings corresponding to the best performer
prob_inners = inner_ix(prob_mask);
fac_inners = inner_ix(fac_mask);

best_set.prob.pos_std_scalar = params_set.prob.pos_std_scalar(prob_inners(I_prob));
best_set.prob.output_threshold_scalar = params_set.prob.output_threshold_scalar(prob_inners(I_prob));
best_set.prob.acquisition_threshold = params_set.prob.acquisition_threshold(prob_inners(I_prob));
best_set.prob.vel_std_scalar = params_set.prob.vel_std_scalar(prob_inners(I_prob));
best_set.prob.turn_threshold = params_set.prob.turn_threshold(prob_inners(I_prob));
best_set.prob.reliability = params_set.prob.reliability(prob_inners(I_prob));
best_set.prob.lost_position_threshold = params_set.prob.lost_position_threshold(prob_inners(I_prob));
best_set.prob.min_turn_gap = params_set.prob.min_turn_gap(prob_inners(I_prob));
best_set.prob.tix_ignore_scalar = params_set.prob.tix_ignore_scalar(prob_inners(I_prob));
best_set.prob.output_integration = params_set.prob.output_integration(prob_inners(I_prob));
best_set.prob.likelihood_threshold = params_set.prob.likelihood_threshold(prob_inners(I_prob));

best_set.fac.gain = params_set.fac.gain(fac_inners(I_prob));
best_set.fac.sigma = params_set.fac.sigma(fac_inners(I_prob));
best_set.fac.kernel_spacing = params_set.fac.kernel_spacing(fac_inners(I_prob));
best_set.fac.emd_timeconstant = params_set.fac.emd_timeconstant(fac_inners(I_prob));
best_set.fac.lpf_timeconstant = params_set.fac.lpf_timeconstant(fac_inners(I_prob));
best_set.fac.distance_turn_threshold = params_set.fac.distance_turn_threshold(fac_inners(I_prob));
best_set.fac.direction_predict_distance = params_set.fac.direction_predict_distance(fac_inners(I_prob));
best_set.fac.direction_predict_threshold = params_set.fac.direction_predict_threshold(fac_inners(I_prob));
best_set.fac.direction_turn_threshold = params_set.fac.direction_turn_threshold(fac_inners(I_prob));
best_set.fac.tix_between_saccades = params_set.fac.tix_between_saccades(fac_inners(I_prob));
best_set.fac.output_integration = params_set.fac.output_integration(fac_inners(I_prob));

disp(['Best prob mean: ' num2str(best_mean_prob)])
disp(['Best fac mean: ' num2str(best_mean_fac)])

% subplot(2,2,1)
% plot(sort(mean(prob_results,2)))
% subplot(2,2,2)
% plot(sort(mean(fac_results,2)))

% Look at influence of distractors and saccades

distract_mask = outer_var.sets.draw_distractors(outer_ix);
predict_mask = outer_var.sets.do_predictive_turn(outer_ix);
saccade_mask = outer_var.sets.do_saccades(outer_ix);

save('E:\PHD\conf2\data\best_set.mat','best_set');

f=figure(1);
f.Name = 'Influence of Distractors';
subplot(2,2,1)
plot(sort(mean(logs.t_lost_it.first(prob_mask & distract_mask,:),2)))
title('prob, distract')
subplot(2,2,2)
plot(sort(mean(logs.t_lost_it.first(prob_mask & ~distract_mask,:),2)))
title('prob, nodistract')
subplot(2,2,3)
plot(sort(mean(logs.t_lost_it.first(fac_mask & distract_mask,:),2)))
title('fac, distract')
subplot(2,2,4)
plot(sort(mean(logs.t_lost_it.first(fac_mask & ~distract_mask,:),2)))
title('fac, nodistract')

f=figure(2);
f.Name = 'Influence of predictive turn';
subplot(2,2,1)
plot(sort(mean(logs.t_lost_it.first(prob_mask & predict_mask,:),2)))
title('prob, predict')
subplot(2,2,2)
plot(sort(mean(logs.t_lost_it.first(prob_mask & ~predict_mask,:),2)))
title('prob, nopredict')
subplot(2,2,3)
plot(sort(mean(logs.t_lost_it.first(fac_mask & predict_mask,:),2)))
title('fac, predict')
subplot(2,2,4)
plot(sort(mean(logs.t_lost_it.first(fac_mask & ~predict_mask,:),2)))
title('fac, nopredict')

f=figure(3);
f.Name = 'Influence of non-instant saccade';
subplot(2,2,1)
plot(sort(mean(logs.t_lost_it.first(prob_mask & saccade_mask,:),2)))
title('prob, saccade')
subplot(2,2,2)
plot(sort(mean(logs.t_lost_it.first(prob_mask & ~saccade_mask,:),2)))
title('prob, instant')
subplot(2,2,3)
plot(sort(mean(logs.t_lost_it.first(fac_mask & saccade_mask,:),2)))
title('fac, saccade')
subplot(2,2,4)
plot(sort(mean(logs.t_lost_it.first(fac_mask & ~saccade_mask,:),2)))
title('fac, instant')
