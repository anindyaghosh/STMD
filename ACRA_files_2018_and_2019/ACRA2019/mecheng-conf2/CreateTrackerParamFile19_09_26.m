clear all

% The purpose of this parameter file is to test a single set of tracker
% parameters (found to be the best on the other test set) against a wider
% range of test conditions

% 19_09_24: This is identical to 19_09_22 except that some of the
% facilitation parameters are changed

fixed.target_movement = 'rotational';
fixed.pixel2PR = 12;
fixed.sigma_hw_degree = 1.4;
fixed.cs_kernel_size = 5;
% fixed.hfov = 40;
% fixed.vfov = 20;
fixed.Ts=0.001;
% fixed.saccade_duration = 0.05;
fixed.make_histograms = false;
fixed.showimagery = false;
fixed.traj_range = 1:100;
fixed.tix_after_saccade = 30;
fixed.trajfilename = 'trajectories19_09_06.mat';
fixed.image_name = 'HDR_Botanic_RGB.png';
fixed.obs_model_filename = 'obs_model_4_botanic.mat';

% Fixed tracker parameters based on best parameters from the 19_09_16 run
% load('E:\PHD\conf2\data\best_set.mat'); % Creates best_set
load('D:\simfiles\conf2\trackerinfo\best_set.mat'); % Creates best_set

fixed.prob.pos_std_scalar = best_set.prob.pos_std_scalar;
fixed.prob.output_threshold_scalar = best_set.prob.output_threshold_scalar;
fixed.prob.acquisition_threshold = best_set.prob.acquisition_threshold;
fixed.prob.vel_std_scalar = best_set.prob.vel_std_scalar;
fixed.prob.turn_threshold = 0;
fixed.prob.reliability = best_set.prob.reliability;
fixed.prob.lost_position_threshold = best_set.prob.lost_position_threshold;
fixed.prob.min_turn_gap = best_set.prob.min_turn_gap;
fixed.prob.tix_ignore_scalar = best_set.prob.tix_ignore_scalar;
fixed.prob.output_integration = best_set.prob.output_integration;
fixed.prob.likelihood_threshold = best_set.prob.likelihood_threshold;

fixed.fac.gain = best_set.fac.gain;
fixed.fac.sigma = 7/2.35482; % Looked best from ZB's interface paper
fixed.fac.kernel_spacing = 5; % Spacing used by ZB
fixed.fac.emd_timeconstant = best_set.fac.emd_timeconstant; % matches ZB
fixed.fac.lpf_timeconstant = best_set.fac.lpf_timeconstant; % ZB said this varies according to velocity ratios
fixed.fac.distance_turn_threshold = best_set.fac.distance_turn_threshold; % Matches ZB
fixed.fac.direction_predict_distance = best_set.fac.direction_predict_distance; % Matches ZB
fixed.fac.direction_predict_threshold = 0.1; % Set to match ZB implementation
fixed.fac.direction_turn_threshold = 0; % Set to match ZB implementation
fixed.fac.tix_between_saccades = best_set.fac.tix_between_saccades; % Matches ZB implementation
fixed.fac.output_integration = best_set.fac.output_integration;
fixed.fac.output_turn_threshold = 0.01;

image_names = {'HDR_Botanic_RGB.png',...
    'HDR_Bushes_RGB.png',...
    'HDR_Creek Bed_RGB.png',...
    'HDR_Field_RGB.png',...
    'HDR_Fountain_RGB.png',...
    'HDR_Hill_RGB.png',...
    'HDR_House_RGB.png',...
    'HDR_Libary_RGB.png',...
    'HDR_Outdoor_RGB.png',...
    'HDR_Park_RGB.png',...
    'HDR_Rock Garden_RGB.png',...
    'HDR_Rubble_RGB.png',...
    'HDR_Shadow_RGB.png',...
    'HDR_Tree_RGB.png',...
    'HDR_Walkway_RGB.png'};

% Outer:
% *distractors
% *saccades
% tracker
% hfov
% do predictive

% Inner sets for distractors and saccades

outer_var.ranges.hfov = [40 60 90]; % vfov to be defined as hfov/2
outer_var.ranges.image_num = 1:numel(image_names);

outer_var.ranges.tracker = {'prob','prob_fixed','facilitation','facilitation_dark'};
outer_var.ranges.tracker_num = 1:numel(outer_var.ranges.tracker);

outer_var.ranges.draw_distractors = [false true];
outer_var.ranges.draw_distractors_num = 1:numel(outer_var.ranges.draw_distractors);

outer_var.ranges.do_predictive_turn = [false true];
outer_var.ranges.do_predictive_turn_num = 1:numel(outer_var.ranges.do_predictive_turn);

outer_var.ranges.do_saccades = [false true];
outer_var.ranges.do_saccades_num = 1:numel(outer_var.ranges.do_saccades);

[   outer_var.sets.hfov,...
    outer_var.sets.image_num,...
    outer_var.sets.tracker_num,...
    outer_var.sets.draw_distractors_num,...
    outer_var.sets.do_predictive_turn_num,...
    outer_var.sets.do_saccades_num]=ndgrid(...
    outer_var.ranges.hfov,...
    outer_var.ranges.image_num,...
    outer_var.ranges.tracker_num,...
    outer_var.ranges.draw_distractors_num,...
    outer_var.ranges.do_predictive_turn_num,...
    outer_var.ranges.do_saccades_num);

outer_var.sets.image_name = cell(size(outer_var.sets.image_num));
outer_var.sets.tracker = cell(size(outer_var.sets.tracker_num));
outer_var.sets.draw_distractors = false(size(outer_var.sets.tracker));
outer_var.sets.do_predictive_turn = false(size(outer_var.sets.tracker));
outer_var.sets.do_saccades = false(size(outer_var.sets.tracker));

for k=1:numel(outer_var.sets.tracker_num)
    outer_var.sets.image_name{k} = image_names{outer_var.sets.image_num(k)};
    outer_var.sets.tracker{k} = outer_var.ranges.tracker{outer_var.sets.tracker_num(k)};
    outer_var.sets.draw_distractors(k) = outer_var.ranges.draw_distractors(outer_var.sets.draw_distractors_num(k));
    outer_var.sets.do_predictive_turn(k) = outer_var.ranges.do_predictive_turn(outer_var.sets.do_predictive_turn_num(k));
    outer_var.sets.do_saccades(k) = outer_var.ranges.do_saccades(outer_var.sets.do_saccades_num(k));
end

params_range.distractors.num_distractors = 20;
params_range.distractors.min_distractor_hd = 0;
params_range.distractors.min_distractor_forward = 0.4;
params_range.distractors.max_distractor_hd = 1.2;
params_range.distractors.max_distractor_forward = 2;
params_range.distractors.max_distractor_elevation_mag = 0.15;
params_range.distractors.luminance_min = 0;
params_range.distractors.luminance_max = 1;

[params_set.distractors.num_distractors,...
params_set.distractors.min_distractor_hd,...
params_set.distractors.min_distractor_forward,...
params_set.distractors.max_distractor_hd,...
params_set.distractors.max_distractor_forward,...
params_set.distractors.max_distractor_elevation_mag,...
params_set.distractors.luminance_min,...
params_set.distractors.luminance_max]=ndgrid(...
    params_range.distractors.num_distractors,...
    params_range.distractors.min_distractor_hd,...
    params_range.distractors.min_distractor_forward,...
    params_range.distractors.max_distractor_hd,...
    params_range.distractors.max_distractor_forward,...
    params_range.distractors.max_distractor_elevation_mag,...
    params_range.distractors.luminance_min,...
    params_range.distractors.luminance_max);

% For each outer_ix, fill in the inner_ix which applies

assumed_fwd_vel_range = [1 3 6];
saccade_duration_range = [0.025 0.05 0.1];

inds_count = 0;
for k=1:numel(outer_var.sets.hfov)
    inner_var =[];
    if(outer_var.sets.do_saccades(k) && outer_var.sets.draw_distractors(k))
        [x_saccade_duration,x_assumed_fwd_vel] = ndgrid(saccade_duration_range,assumed_fwd_vel_range);
        inds_count = inds_count + numel(x_saccade_duration);
    elseif(outer_var.sets.do_saccades(k) && ~outer_var.sets.draw_distractors(k))
        x_saccade_duration = saccade_duration_range;
        inds_count = inds_count + numel(x_saccade_duration);
    elseif(~outer_var.sets.do_saccades(k) && outer_var.sets.draw_distractors(k))
        x_assumed_fwd_vel = assumed_fwd_vel_range;
        inds_count = inds_count + numel(x_assumed_fwd_vel);
    else
        % No saccades, no distractors so this just has 1 run
        inds_count = inds_count + 1;
    end
end

outer_ix = NaN(inds_count,1);
inner_ix = NaN(inds_count,1);

inner_var.saccade_duration = NaN(inds_count,1);
inner_var.assumed_fwd_vel = NaN(inds_count,1);

curr_ind=1;

for k = 1:numel(outer_var.sets.hfov)
    if(outer_var.sets.do_saccades(k) && outer_var.sets.draw_distractors(k))
        [x_saccade_duration,x_assumed_fwd_vel] = ndgrid(saccade_duration_range,assumed_fwd_vel_range);
        inds_range = curr_ind:curr_ind+numel(x_saccade_duration) -1;
        inner_var.saccade_duration(inds_range) = x_saccade_duration(:);
        inner_var.assumed_fwd_vel(inds_range) = x_assumed_fwd_vel(:);
        
    elseif(outer_var.sets.do_saccades(k) && ~outer_var.sets.draw_distractors(k))
        x_saccade_duration = saccade_duration_range;
        inds_range = curr_ind:curr_ind+numel(x_saccade_duration) -1;
        inner_var.saccade_duration(inds_range) = x_saccade_duration(:);
        inner_var.assumed_fwd_vel(inds_range) = NaN;
        
    elseif(~outer_var.sets.do_saccades(k) && outer_var.sets.draw_distractors(k))
        x_assumed_fwd_vel = assumed_fwd_vel_range;
        inds_range = curr_ind:curr_ind+numel(x_assumed_fwd_vel) -1;
        inner_var.saccade_duration(inds_range) = NaN;
        inner_var.assumed_fwd_vel(inds_range) = x_assumed_fwd_vel(:);
        
    else
        inds_range = curr_ind;
        inner_var.saccade_duration(inds_range) = NaN;
        inner_var.assumed_fwd_vel(inds_range) = NaN;
    end

    outer_ix(inds_range) = k;
    inner_ix(inds_range) = 1:numel(inds_range);
    curr_ind = curr_ind + numel(inds_range);
end

load('E:\PHD\TEST\belief_vel_fixed.mat');

this_script = mfilename;
save('E:\PHD\conf2\data\trackerinfo\TrackerParamFile19_09_26.mat',...
    'this_script','outer_ix','inner_ix','fixed','params_set','params_range','outer_var','inner_var','belief_vel_fixed')
    