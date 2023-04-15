% function Model2019_19_09_04(grp_index,grp_file,groupname)

% pixel2PR, sigma_hw_pixel, cs_kernel_size to be specified in group sheet
% clear all
% close all

% 19_09_24: Introduced a threshold on the output before turning for
% facilitation. Fixed a bug with facilitation saccades (saccade_end_tix was not being set when do_saccades was false).

% 19_09_25: There were some issues with the facilitation pursuit. Also, had
% to introduce a period of time where the facilitation tracker ignores
% inputs after turns

% 19_09_30: Fixed issue where belief.pos could become negative

% 19_10_03: Regularising the pursuit strategies so that all trackers use
% the same turn threshold

% 19_10_08: No substantive changes to the simulations. However, there was
% an issue where when terminating early due to running out of time, trials
% were being marked complete. Fixed that, and have written in a check to
% see if early termination occurred. In that case, the trial will be
% over-written.

% 19_10_14: Trials which terminated early prior to 751 ticks elapsing were
% not being correctly characterised as wrongfully terminated. That is fixed
% in this version. A large proportion of the data set was affected.

% 19_10_16: There were some cases of what looked like early termination
% where what had happened is the target came back into the screen in
% between output_integration ticks (i.e. for at most 3 consecutive frames).
% The decision to end early is only evaluated every output_integration
% ticks. To address this, now checking whether t_th_rel_history viewed
% through integrated_tix_history appears to be entirely off screen from 500
% ticks before end until the end.

% 19_10_25: The performance of my facilitation implementation was abysmal.
% So, adapting ZB implementation to see if that does any better. That has
% its own quirks in terms of how the ESTMD model is actually implemented.

% function TrackerModel19_10_25(param_settings_file, gind, groupname, hours_to_run, in_toc)
% disp(['Now running g: ' num2str(gind)]);

% % To be provided via function call
clear all
close all
tic
in_toc = toc;
gind=1080;
% data_dir = 'D:\temp\';
hours_to_run = 16/60;
% param_settings_file = 'D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat';
param_settings_file = 'E:\PHD\TEST\TrackerParamFile19_10_03.mat';
% E:\PHD\TEST
groupname='something';

which_pc='mrblack';
settings.showimagery = true;
% 
% which_pc='phoenix';
% settings.showimagery = false;

disp('Use_ZB_ESTMD has temporary false setting')
settings.Use_ZB_ESTMD = false;

if(strcmp(which_pc,'phoenix'))
    settings.trajfile_loc = '/fast/users/a1119946/simfiles/conf2/data/trackerinfo/';
    settings.im_loc = '/fast/users/a1119946/Texture/';
    settings.obs_model_loc = '/fast/users/a1119946/simfiles/conf2/data/trackerinfo/';
    data_dir = ['/fast/users/a1119946/simfiles/conf2/data/' groupname '/'];
elseif(strcmp(which_pc,'mrblack'))
    settings.trajfile_loc = 'E:\PHD\conf2\data\trackerinfo\';
    settings.im_loc = 'E:\PHD\texture\';
    settings.obs_model_loc = 'E:\PHD\conf2\data\trackerinfo\';
    data_dir = ['E:\PHD\conf2\data\' groupname '\'];
elseif(strcmp(which_pc,'mecheng'))
    addpath('D:\scripts\dev\conf2\');
    settings.trajfile_loc = 'D:\simfiles\conf2\trackerinfo\';
    settings.im_loc = 'D:\simfiles\texture\';
    settings.obs_model_loc = 'D:\simfiles\conf2\trackerinfo\';
    data_dir = ['D:\simfiles\conf2\test\' groupname '\'];
end

savename = [data_dir 'trackres_' num2str(gind) '.mat'];
toc_to_die = in_toc + hours_to_run*60*60;
param_settings=load(param_settings_file);

if(~exist(data_dir,'dir'))
    mkdir(data_dir)
end

settings.time_spent_so_far = 0;
settings.target_movement = param_settings.fixed.target_movement;
settings.pixel2PR = param_settings.fixed.pixel2PR;
settings.sigma_hw_degree = param_settings.fixed.sigma_hw_degree;
settings.cs_kernel_size = param_settings.fixed.cs_kernel_size;
settings.Ts = param_settings.fixed.Ts;
settings.make_histograms = param_settings.fixed.make_histograms;
settings.traj_range = param_settings.fixed.traj_range;
settings.tix_after_saccade = param_settings.fixed.tix_after_saccade;
% disp('Over-riding tix_after_saccade')
% settings.tix_after_saccade = 40;
settings.trajfilename = param_settings.fixed.trajfilename;
settings.obs_model_filename = param_settings.fixed.obs_model_filename;

% Outer index settings
settings.outer_ix=param_settings.outer_ix(gind);
settings.inner_ix=param_settings.inner_ix(gind);

% Vary hfov and saccade_duration
% settings.hfov = param_settings.outer_var.sets.hfov(settings.outer_ix);
% settings.vfov = param_settings.outer_var.sets.hfov(settings.outer_ix)/2;
disp('Over-riding hfov and vfov')
settings.hfov = 90;
settings.vfov = 45;

settings.saccade_duration = param_settings.inner_var.saccade_duration(gind);

% Vary image name
settings.image_name = param_settings.outer_var.sets.image_name{settings.outer_ix};
settings.impath = [settings.im_loc settings.image_name];

% settings.tracker = param_settings.outer_var.sets.tracker{settings.outer_ix};
disp('Over-riding settings.tracker')
settings.tracker = 'ZB_fac';
distractorparams.draw_distractors = param_settings.outer_var.sets.draw_distractors(settings.outer_ix);
settings.do_predictive_turn = param_settings.outer_var.sets.do_predictive_turn(settings.outer_ix);
settings.do_saccades = param_settings.outer_var.sets.do_saccades(settings.outer_ix);

settings.turn_distance_threshold = param_settings.fixed.turn_distance_threshold;
settings.use_ZB_tracker = true;

if(strcmp(settings.tracker,'prob') || strcmp(settings.tracker,'prob_fixed'))
    settings.output_integration = param_settings.fixed.prob.output_integration;
    facparams=[];
    % Tracker parameters
    trackerparams.pos_std_scalar = param_settings.fixed.prob.pos_std_scalar;
    trackerparams.output_threshold_scalar = param_settings.fixed.prob.output_threshold_scalar;
    trackerparams.vel_std_scalar = param_settings.fixed.prob.vel_std_scalar;
    trackerparams.acquisition_threshold = param_settings.fixed.prob.acquisition_threshold; % What the max probability has to evaluate out to before we assume a target is acquired
    trackerparams.turn_threshold = param_settings.fixed.prob.turn_threshold;
    trackerparams.reliability = param_settings.fixed.prob.reliability; % An assumption about how likely the measurement is to correspond to the target
    trackerparams.lost_position_threshold = param_settings.fixed.prob.lost_position_threshold;
    trackerparams.min_turn_gap = param_settings.fixed.prob.min_turn_gap;
    trackerparams.tix_ignore_scalar = param_settings.fixed.prob.tix_ignore_scalar;
    trackerparams.likelihood_threshold = param_settings.fixed.prob.likelihood_threshold;
    
    % Load obs model
    trackerparams.obs_model = load([settings.obs_model_loc settings.obs_model_filename]);
    
    % Derived tracker parameters
    trackerparams.pos_std = trackerparams.pos_std_scalar*settings.output_integration; % Gaussian around the observed location
    trackerparams.output_threshold = trackerparams.output_threshold_scalar * settings.output_integration; % Model outputs <= this are ignored.
    trackerparams.vel_std = settings.output_integration * settings.Ts * trackerparams.vel_std_scalar; % How much the velocity may have changed in the time since the last tracker tick
    trackerparams.vel_resolution = trackerparams.vel_std; % Resolve velocity belief at a resolution equal to one standard deviation of our velocity evolution model
    trackerparams.vel_pts = 0:trackerparams.vel_resolution:max(trackerparams.obs_model.t_vel_range);
    trackerparams.tix_ignore_after_turn = floor(trackerparams.tix_ignore_scalar / settings.Ts / settings.output_integration);
    
    % Other
    angle_pts = 0:30:330;
    
    rightward_angles = angle_pts < 90 | angle_pts > 270;
    rightward_angles = rightward_angles(:);
    leftward_angles = angle_pts > 90 | angle_pts < 270;
    leftward_angles=leftward_angles(:);
    
    % Since the resolution of velocity is 1 std deviation, this kernel is
    % +/- 4 std deviations
    trackerparams.vel_evolution_kernel = normpdf(-4:4,0,1);
    trackerparams.vel_evolution_kernel = trackerparams.vel_evolution_kernel / sum(trackerparams.vel_evolution_kernel(:));
    
    trackerparams.pos_grid_spacing = 0.5;
    [trackerparams.Y,trackerparams.X] = ndgrid(1:trackerparams.pos_grid_spacing:settings.vfov,1:trackerparams.pos_grid_spacing:settings.hfov); % Grid coordinates for screen
    
    % Carry out linear interpolation of the data-driven prob_R_v to get a
    % finer resolution prob_R_v
    
    [resp_levels,~]=size(trackerparams.obs_model.prob_R_v);
    
    trackerparams.prob_R_v = NaN(resp_levels,numel(trackerparams.vel_pts));
    % Below the lowest velocity, just set p = p(lowest)
    low_mask = trackerparams.vel_pts <= min(trackerparams.obs_model.t_vel_range);
    trackerparams.prob_R_v(:,low_mask) = repmat(trackerparams.obs_model.prob_R_v(:,1),1,sum(low_mask));
    
    % Above the highest velocity, set p = p(highest)
    high_mask = trackerparams.vel_pts >= max(trackerparams.obs_model.t_vel_range);
    trackerparams.prob_R_v(:,high_mask) = repmat(trackerparams.obs_model.prob_R_v(:,end),1,sum(high_mask));
    
    % For everything else, use a linear interpolation between the points
    % for which there is actual data to derive probabilities
    for vix=find(low_mask == false,1,'first'):find(high_mask == false,1,'last')
        I_lower = find(trackerparams.vel_pts(vix) >= trackerparams.obs_model.t_vel_range,1,'last');
        I_upper = find(trackerparams.vel_pts(vix) < trackerparams.obs_model.t_vel_range,1,'first');
        
        prop= ( trackerparams.vel_pts(vix) - trackerparams.obs_model.t_vel_range(I_lower) ) /...
            ( trackerparams.obs_model.t_vel_range(I_upper) - trackerparams.obs_model.t_vel_range(I_lower) );
        
        trackerparams.prob_R_v(:,vix) = (1-prop)*trackerparams.obs_model.prob_R_v(:,I_lower) + prop*trackerparams.obs_model.prob_R_v(:,I_upper);
        % Due to numerical reasons the column sums may not be 1 but this cannot
        % be helped
    end
    
    % Pixel min/max distance map for a 5x5 kernel (distance from centre of
    % central pixel
    trackerparams.pixel_distance_kernel_size = 5;
    dmin=NaN(trackerparams.pixel_distance_kernel_size);
    dmax=NaN(trackerparams.pixel_distance_kernel_size);
    
    central_pixel = ceil(trackerparams.pixel_distance_kernel_size/2);
    
    for x=1:trackerparams.pixel_distance_kernel_size
        for y=1:trackerparams.pixel_distance_kernel_size
            xdist= abs(x-central_pixel);
            ydist= abs(y-central_pixel);
            
            if(xdist > 0)
                xstep = trackerparams.pos_grid_spacing*(0.5 + xdist-1);
            else
                xstep = 0;
            end
            
            if(ydist > 0)
                ystep = trackerparams.pos_grid_spacing*(0.5 + ydist-1);
            else
                ystep = 0;
            end
            
            dmin(x,y) = norm([xstep ystep]);
            if(x == central_pixel && y == central_pixel)
                dmax(x,y)= norm(trackerparams.pos_grid_spacing*[0.5 0.5]);
            else
                dmax(x,y)= norm([trackerparams.pos_grid_spacing*(xdist+0.5) trackerparams.pos_grid_spacing*(ydist+0.5)]);
            end
        end
    end
    
    % Identify which velocities should be summed across
    trackerparams.pos_velocities = cell(size(dmin));
    
    trackerparams.vmin = NaN(size(dmin));
    trackerparams.vmax = trackerparams.vmin;
    for p=1:numel(trackerparams.vmin)
        trackerparams.vmin(p) = dmin(p) / (settings.output_integration*settings.Ts);
        trackerparams.vmax(p) = dmax(p) / (settings.output_integration*settings.Ts);
    end
    
    % Masks for which velocities are required to move from pixel to pixel
    % in output_integration*Ts seconds
    for p=1:numel(trackerparams.pos_velocities)
        trackerparams.pos_velocities{p} = (trackerparams.vel_pts >= trackerparams.vmin(p) & trackerparams.vel_pts <= trackerparams.vmax(p) )';
    end
    
    % Need a mapping of the actual pixel coordinates into belief.pos
    trackerparams.pix_index_map = NaN(settings.vfov,settings.hfov); % The entries in this should be the indices of belief.pos that correspond to the pixels
    for p=1:settings.vfov*settings.hfov
        [ypos,xpos] = ind2sub([settings.vfov settings.hfov], p);
        mask = trackerparams.Y == ypos & trackerparams.X == xpos;
        trackerparams.pix_index_map(p) = find(mask == true,1,'first');
    end
    
    settings.trackerparams = trackerparams;
    
elseif(strcmp(settings.tracker,'facilitation') || strcmp(settings.tracker,'facilitation_dark'))
    settings.output_integration = param_settings.fixed.fac.output_integration;
    trackerparams=[];
    
    facparams.gain = param_settings.fixed.fac.gain;
    facparams.sigma = param_settings.fixed.fac.sigma;
    facparams.kernel_spacing = param_settings.fixed.fac.kernel_spacing;
    facparams.emd_timeconstant = param_settings.fixed.fac.emd_timeconstant;
    facparams.lpf_timeconstant = param_settings.fixed.fac.lpf_timeconstant;
    facparams.distance_turn_threshold = param_settings.fixed.fac.distance_turn_threshold;
    facparams.direction_predict_distance = param_settings.fixed.fac.direction_predict_distance;
    facparams.direction_predict_threshold = param_settings.fixed.fac.direction_predict_threshold;
    facparams.direction_turn_threshold = param_settings.fixed.fac.direction_turn_threshold;
    facparams.tix_between_saccades = param_settings.fixed.fac.tix_between_saccades;
    facparams.output_turn_threshold = param_settings.fixed.fac.output_turn_threshold;

%     facparams.tix_ignore_scalar = 0.05;
%     facparams.tix_ignore_after_turn = floor(facparams.tix_ignore_scalar / settings.Ts / settings.output_integration);
    
    settings.facparams = facparams;
elseif(strcmp(settings.tracker,'kalman') || strcmp(settings.tracker,'kalman_dark'))
    % Settings from param file
    settings.output_integration = param_settings.fixed.kf.output_integration;
    kfparams.Q = param_settings.fixed.kf.Q_initial;
    kfparams.R = param_settings.fixed.kf.R_initial;
    kfparams.output_threshold_scalar = param_settings.fixed.kf.output_threshold_scalar;
    kfparams.certainty_threshold = param_settings.fixed.kf.certainty_threshold;
    kfparams.acquired_ignore_distance = param_settings.fixed.kf.acquired_ignore_distance;
    kfparams.max_certainty = param_settings.fixed.kf.max_certainty;
    % Derived settings
    kfparams.output_threshold = kfparams.output_threshold_scalar * settings.output_integration;
elseif(strcmp(settings.tracker,'constgain') || strcmp(settings.tracker,'constgain_dark'))
    settings.output_integration = param_settings.fixed.constgain.output_integration;
    constgain.width = param_settings.fixed.constgain.width;
    constgain.value = param_settings.fixed.constgain.value;
    constgain.output_threshold_scalar = param_settings.fixed.constgain.output_threshold_scalar;
    % Gain values per output pixel. Set to 1 outside the central gain
    % field, and to constgain.value inside it
    constgain.field = ones(settings.vfov,settings.hfov);
    constgain.field(:,settings.hfov/2-constgain.width:settings.hfov/2 + constgain.width) = constgain.value;

    constgain.output_threshold = constgain.output_threshold_scalar * settings.output_integration;
    
elseif(strcmp(settings.tracker,'none') || strcmp(settings.tracker,'none_dark'))
    settings.output_integration = param_settings.fixed.none.output_integration;
    notrack.output_threshold_scalar = param_settings.fixed.none.output_threshold_scalar;
    % Derived threshold
    
    notrack.output_threshold = settings.output_integration * notrack.output_threshold_scalar;
    
elseif(strcmp(settings.tracker,'ZB_fac') || strcmp(settings.tracker,'ZB_fac_dark'))
    % setup for the ZB_fac tracker
%     settings.output_integration = param_settings.fixed.ZB_fac.output_integration;
%     ZB_fac.output_threshold = param_settings.ZB_fac.output_threshold;
    
    settings.output_integration = 1;
    ZB_fac.Threshold = 0.1;
    ZB_fac.output_threshold = 0;

    ZB_fac.wb = 0.05;
    ZB_fac.FacilitationGain =25; %root of facilitation kernel gain
    ZB_fac.RC_wb = 5; 
    ZB_fac.parameter.RC_fac_wb = 1;
    ZB_fac.Threshold = 0.1;
    ZB_fac.Ts = 0.05;
    
    % Buffer used by PhotoReceptor_feedin
    ZB_fac.dbuffer1 = cell(3,1);
    
    % Buffers used by RTC_feedin function
    ZB_fac.On_Delayed_Channel = cell(3,1);
    ZB_fac.Off_Delayed_Channel = ZB_fac.On_Delayed_Channel;
    ZB_fac.ON_Filtered_Delayed = ZB_fac.On_Delayed_Channel;
    ZB_fac.OFF_Filtered_Delayed = ZB_fac.On_Delayed_Channel;
    ZB_fac.ONbuffer = ZB_fac.On_Delayed_Channel;
    ZB_fac.OFFbuffer = ZB_fac.On_Delayed_Channel;
    
    ZB_fac.initialCol = 5;
    ZB_fac.initialRow = settings.vfov/2;
    
    % Used by facilitation
    ZB_fac.Delayed_Grid = [];
    ZB_fac.Facilitationbuffer = [];
    
    % Size of subsampled image
    ZB_fac.XDim = settings.hfov;
    ZB_fac.YDim = settings.vfov;

    ZB_fac.Facilitation_Mode='on';

    ZB_fac.Facilitation_Matrix=ones(ZB_fac.YDim, ZB_fac.XDim);
    ZB_fac.Facilitation_sigma=5/2.35482;

    [ZB_fac.Y,ZB_fac.X] = ndgrid(1:settings.vfov, 1:settings.hfov);
    
else
    error('Tracker setting invalid')
end

% Distractor parameters
distractorparams.num_distractors = param_settings.params_set.distractors.num_distractors;
distractorparams.min_distractor_hd = param_settings.params_set.distractors.min_distractor_hd;
distractorparams.min_distractor_forward = param_settings.params_set.distractors.min_distractor_forward;
distractorparams.max_distractor_hd = param_settings.params_set.distractors.max_distractor_hd;
distractorparams.max_distractor_forward = param_settings.params_set.distractors.max_distractor_forward;
distractorparams.max_distractor_elevation_mag = param_settings.params_set.distractors.max_distractor_elevation_mag;
distractorparams.luminance_min = param_settings.params_set.distractors.luminance_min;
distractorparams.luminance_max = param_settings.params_set.distractors.luminance_max;

% Assumed fwd_vel being varied
distractorparams.assumed_fwd_vel = param_settings.inner_var.assumed_fwd_vel(gind);
distractors.initialised=false;
distractors.to_init = false;

settings.use_prob_tracker = false;
settings.use_facilitation_tracker = false;
settings.use_kf_tracker = false;
settings.use_constgain_tracker = false;
settings.use_no_tracker = false;

if(~settings.make_histograms)
    if(strcmp(settings.tracker,'prob') || strcmp(settings.tracker,'prob_fixed'))
        settings.use_prob_tracker = true;
        if(strcmp(settings.tracker,'prob_fixed'))
            settings.do_velocity_update = false;
        else
            settings.do_velocity_update = true;
        end
        settings.use_light_and_dark = false;
    elseif(strcmp(settings.tracker,'facilitation') || strcmp(settings.tracker,'facilitation_dark'))
        settings.use_facilitation_tracker = true;
        if(strcmp(settings.tracker,'facilitation'))
            settings.use_light_and_dark = true;
        else
            settings.use_light_and_dark = false;
        end
    elseif(strcmp(settings.tracker,'kalman') || strcmp(settings.tracker,'kalman_dark'))
        settings.use_kf_tracker = true;
        if(strcmp(settings.tracker,'kalman'))
            settings.use_light_and_dark = true;
        else
            settings.use_light_and_dark = false;
        end
    elseif(strcmp(settings.tracker,'constgain') || strcmp(settings.tracker,'constgain_dark'))
        settings.use_constgain_tracker = true;
        if(strcmp(settings.tracker,'constgain'))
            settings.use_light_and_dark = true;
        else
            settings.use_light_and_dark = false;
        end
    elseif(strcmp(settings.tracker,'none') || strcmp(settings.tracker,'none_dark'))
        settings.use_no_tracker = true;
        if(strcmp(settings.tracker,'none'))
            settings.use_light_and_dark = true;
        else
            settings.use_light_and_dark = false;
        end
    elseif(strcmp(settings.tracker,'ZB_fac') || strcmp(settings.tracker,'ZB_facdark'))
        settings.use_ZB_tracker = true;
        if(strcmp(settings.tracker,'ZB_fac'))
            settings.use_light_and_dark = true;
        else
            settings.use_light_and_dark = false;
        end
    else
        error('Tracker selection invalid')
    end
    
    if(strcmp(settings.target_movement,'rotational'))
        settings.target_rotational = true;
        settings.target_position_based = false;
    elseif(strcmp(settings.target_movement,'position_based'))
        settings.target_rotational = false;
        settings.target_position_based = true;
    else
        error('Target movement setting invalid')
    end
end

if(settings.make_histograms)
    %     t_vel_range=20:40:400;
    t_vel_range=20:20:800;
    num_tests=numel(t_vel_range);
else
    %     traj_range = 1;
    num_tests = numel(settings.traj_range);
    trajfile = load([settings.trajfile_loc settings.trajfilename]);
end

pad_size=floor(settings.cs_kernel_size/2);

% Set up rendering

t_width = 1; % Degrees
t_height = 1;
t_value = 0;

frames=1;
delay=1;
drawmode = 'pixels';
degree_per_PR = 1;

% Draw setup
[V,...
    H,...
    vpix,...
    hpix,...
    src,...
    panpix,...
    kernel_size,...
    h,...
    v,...
    kernel,...
    alp,...
    theta,...
    blurbound,...
    hdist,...
    t_pixelleft,...
    t_pixelright,...
    t_pixeldown,...
    t_pixelup]=JJ_Draw_Setup19_09_05(...
    settings.hfov,...
    settings.vfov,...
    settings.impath,...
    t_width,...
    t_height,...
    settings.pixel2PR,...
    degree_per_PR,...
    settings.sigma_hw_degree);

% Combined photoreceptor and high pass filter coefficients
pr_len=single(10);
pr_num=zeros(1,1,pr_len,'single');
pr_num(:)=[0    0.0001   -0.0012    0.0063   -0.0222    0.0609   -0.1013    0.2363   -0.3313    0.1524];
pr_den=zeros(1,1,pr_len,'single');
pr_den(:)=[ 1.0000   -5.1664   12.2955  -17.9486   17.9264  -12.8058    6.5661   -2.3291    0.5166   -0.0542];

pr_num_array=repelem(pr_num,settings.vfov,settings.hfov,1);
pr_den_array=repelem(pr_den,settings.vfov,settings.hfov,1);

% LMC kernel
lmc_kernel = single(1/9*[-1 -1 -1; -1 8 -1; -1 -1 -1]);

% NLAM
tau_on_up=0.01;
tau_on_down=0.1;
tau_off_up=0.01;
tau_off_down=0.1;

alpha_on_up=single(settings.Ts/(settings.Ts+tau_on_up));
alpha_on_down=single(settings.Ts/(settings.Ts+tau_on_down));
alpha_off_up=single(settings.Ts/(settings.Ts+tau_off_up));
alpha_off_down=single(settings.Ts/(settings.Ts+tau_off_down));

% C/S kernel

cs_kernel=zeros(settings.cs_kernel_size);

cs_kernel_val=-16/(4*settings.cs_kernel_size-4);

cs_kernel(:,1)      = cs_kernel_val;
cs_kernel(1,:)      = cs_kernel_val;
cs_kernel(:,end)    = cs_kernel_val;
cs_kernel(end,:)    = cs_kernel_val;

cs_kernel((settings.cs_kernel_size+1)/2,(settings.cs_kernel_size+1)/2)=2;

cs_kernel=single(cs_kernel);

% Low-pass "delay" filter
delay_lp=0.025;
lp_den_raw=[(1+2*delay_lp/settings.Ts), (1-2*delay_lp/settings.Ts)];
lp_num=single([1/lp_den_raw(1) 1/lp_den_raw(1)]);

lp_den=single(lp_den_raw(2)/lp_den_raw(1));

% Setup buffers;
if(distractorparams.draw_distractors)
    settings.num_simultaneous = 3;
else
    settings.num_simultaneous = 2;
end
pr_buffer = cell(settings.num_simultaneous,1);
input_buffer = pr_buffer;
LMC=pr_buffer;
delay_on_inbuff = pr_buffer;
delay_off_inbuff = pr_buffer;
delay_on_outbuff = pr_buffer;
delay_off_outbuff = pr_buffer;
fdsr_on_inbuff = pr_buffer;
fdsr_off_inbuff = pr_buffer;
fdsr_on_outbuff = pr_buffer;
fdsr_off_outbuff = pr_buffer;
alpha_on = pr_buffer;
alpha_off = pr_buffer;
alpha_on_mask = pr_buffer;
alpha_off_mask = pr_buffer;
on_chan = pr_buffer;
off_chan = pr_buffer;
dark_output_buffer = pr_buffer;
light_output_buffer = pr_buffer;

dark_output_integ = pr_buffer;
light_output_integ = pr_buffer;

if(~exist(savename,'file'))
    outer_log.data = cell(num_tests,1);
    outer_log.completed=false(num_tests,1);
    outer_log.completion_times = NaN(num_tests,1);
    nothingchanged=false;
else
    disp('Resuming')
    load(savename);
%     if(sum(outer_log.completed) == num_tests)
%         disp('File appears complete. Exiting')
%         return
%     end
    nothingchanged=true;
end

for test_ix = 1:num_tests;
    
    if(toc > toc_to_die)
        disp('Exiting due to time limit')
        break;
    end
    proceedwith=true;
    if(outer_log.completed(test_ix))
        % Check whether there was a premature termination before
        I_NaN = find(isnan(outer_log.data{test_ix}.t_th_rel_history),1,'first');
        wrongful_termination=false;
        if(~isempty(I_NaN))
            % Indicates that the trial terminated early, now see
            % whether it was unjustified
            if(I_NaN < 751)
                % Obviously if it takes 500 ticks for a correct
                % termination and termination was less than 500 ticks
                % after spin_ticks then clearly it was a wrongful
                % termination
                wrongful_termination = true;
            else
                test_range = I_NaN-500:I_NaN;
                % Only base the analysis on tix that are in integrated_tix_history
                test_range = test_range(ismember(test_range,outer_log.data{test_ix}.integrated_tix_history));
                
                % Test whether the target was on screen at all during the
                % 500 ticks prior to termination
                wrongful_termination = sum(abs(outer_log.data{test_ix}.t_th_rel_history(test_range)) < settings.hfov/2) > 0;
            end
        end
        if(wrongful_termination)
            proceedwith=true;
            nothingchanged=false;
            outer_log.completed(test_ix) = false;
            disp(['Redoing ' num2str(test_ix) ' due to early termination'])
        else
            proceedwith=false;
            disp(['Skipping test: ' num2str(test_ix)])
        end
    end
    
    if(proceedwith)
        nothingchanged=false;
        toc_commencing = toc;
        rng(0); % Fixed seed
        % Because the seed is fixed, the distractor movements will always be the
        % same given the same turns.
        % Therefore, not necessary to save this information out.
        
        if(settings.make_histograms)
            %     resp_levels=20;
            resp_thresholds=[0:0.01:0.5 100];
            t_al_pos_range=[-settings.vfov/2+2 0 settings.vfov/2-2]';
            num_targets=numel(t_al_pos_range);
            obs_th_pos=0;
            
            % A target will be drawn at each of these angles
            hist_R_v = zeros(numel(resp_thresholds)-1,1); % Create single histograms for velocities and combine these later
            t_vel = t_vel_range(test_ix)*ones(num_targets,1);
        else
            % Actually following a trajectory, so just load that
            traj_current = settings.traj_range(test_ix);
            traj=trajfile.traj{traj_current};
            
            obs_th_pos=0;
            
            % Spin a bit to initialise the filters and then introduce the
            % target along the set trajectory
            spin_obs_vel=60;
            spin_obs_angle=15;
            spin_ticks=ceil(spin_obs_angle/spin_obs_vel/settings.Ts); % Time to spin the background
            spinning=true;
            obs_destination = spin_obs_angle;
            max_ticks = numel(traj.t_th_traj) + spin_ticks;
            
            num_targets=1;
            t_th_pos=0;
            last_turn_tix = 0;
            last_turn_integrated = 0;
            
            executing_saccade = false;
            saccade_end_tix =0;
            
            % Flush logs
            logging=[];
            
            logging.distractors_init_history = NaN(max_ticks,1);
            logging.max_output_history = logging.distractors_init_history;
            logging.t_th_rel_history = logging.distractors_init_history;
            logging.integrated_tix_history = logging.distractors_init_history;
            logging.saccading_history = logging.distractors_init_history;
            logging.obs_th_pos_history = logging.distractors_init_history;
            logging.max_target_history = logging.distractors_init_history;
            logging.max_distractor_history = logging.distractors_init_history;
            
            %             logging.distractors_hd_history = NaN(max_ticks,distractorparams.num_distractors);
            %             logging.distractors_fwd_history = logging.distractors_hd_history;
            %             logging.distractors_th_history = logging.distractors_hd_history;
            %             logging.distractors_al_history = logging.distractors_hd_history;
            
            if(settings.use_prob_tracker)
                % Initial settings for the probabilistic tracker
                acquired=false;
                % Just assume uniform priors for everything for now
                belief.angle = ones(numel(angle_pts),1) / numel(angle_pts);
                
                if(settings.do_velocity_update)
                    belief.vel = ones(numel(trackerparams.vel_pts),1) / numel(trackerparams.vel_pts);
                else
                    belief.vel = param_settings.belief_vel_fixed;
                end
                belief.pos = ones(size(trackerparams.Y)) / numel(trackerparams.Y);
                
                logging.max_certainty_history = NaN(max_ticks,1);
                logging.maxbelief_pos_history = logging.max_certainty_history;
                
            elseif(settings.use_facilitation_tracker)
                % Set up the facilitation tracker
                
                % EMD on ESTMD settings
                
                fac.emd_n1 = (settings.Ts*facparams.emd_timeconstant) / (settings.Ts*facparams.emd_timeconstant + 2);
                fac.emd_n2 = (settings.Ts*facparams.emd_timeconstant) / (settings.Ts*facparams.emd_timeconstant + 2);
                fac.emd_d2 = -(settings.Ts*facparams.emd_timeconstant-2) / (settings.Ts*facparams.emd_timeconstant + 2);
                
                % Facilitation low-pass settings
                
                fac.lpf_n1=facparams.lpf_timeconstant*settings.Ts/(settings.Ts*facparams.lpf_timeconstant+2);
                fac.lpf_n2=facparams.lpf_timeconstant*settings.Ts/(settings.Ts*facparams.lpf_timeconstant+2);
                fac.lpf_d2=-(settings.Ts*facparams.lpf_timeconstant-2)/(settings.Ts*facparams.lpf_timeconstant+2);
                
                % Create buffers
                fac.ind = 1;
                % (in ZB implementation, there were 100 cycles of lowpass filter with 1 as input
                % So, the initial state after 100 cycles
                % wouldn't exactly be 1, but close enough
                fac.input_buffer = ones(settings.vfov,settings.hfov,2); % Facilitation initialised with 1
                fac.output_buffer = fac.input_buffer;
                
                fac.emd_input_buffer = zeros(settings.vfov,settings.hfov,2);
                fac.emd_output_buffer = fac.input_buffer;
                
                % Place kernel centers s.t.
                % 1) one of the kernels is at the center of the image
                % 2) no kernel has its center on the border
                img_center_x = settings.hfov/2;
                img_center_y = settings.vfov/2;
                
                fac.kernels_x = floor([sort(img_center_x:-facparams.kernel_spacing:2) img_center_x+facparams.kernel_spacing:facparams.kernel_spacing:settings.hfov-1]);
                fac.kernels_y = floor([sort(img_center_y:-facparams.kernel_spacing:2) img_center_y+facparams.kernel_spacing:facparams.kernel_spacing:settings.vfov-1]);
                
                logging.max_fac_output_history = NaN(max_ticks,1);
                logging.max_fac_output_h_history = logging.max_fac_output_history;
                logging.max_fac_output_v_history = logging.max_fac_output_history;
                logging.lr_dir_output_history = logging.max_fac_output_history;
                logging.ud_dir_output_history = logging.max_fac_output_history;
            elseif(settings.use_kf_tracker)
                % Kalman filter setup
                dt = settings.Ts*settings.output_integration;
                kfparams.A =    [1 0 dt 0;... %xpos = xpos + dt*xvel
                    0 1 0 dt;... %ypos = ypos + dt*yvel
                    0 0 1 0;... %xvel = xvel
                    0 0 0 1]; % yvel = yvel
                kfparams.H = diag([1 1 0 0]); % Measures the position states but not velocity
                x_post = [settings.hfov/2;settings.vfov/2;0;0]; % Assuming the target is in the centre of the screen and is motionless
                P_post = param_settings.fixed.kf.P_post_initial;
                certainty=0;
                
                logging.kf.certainty_history = NaN(max_ticks,1);
                logging.kf.x_post = NaN(max_ticks,4);
            end
        end
        
        dark_target_mask_buffer = false(settings.vfov,settings.hfov,settings.output_integration);
        light_and_dark_target_mask_buffer = dark_target_mask_buffer;
        
        dark_distractor_mask_buffer = dark_target_mask_buffer;
        light_and_dark_distractor_mask_buffer = dark_target_mask_buffer;
        
        for k=1:settings.num_simultaneous
            % Long PR and input image buffer
            pr_buffer{k} = zeros(settings.vfov,settings.hfov,10,'single');
            input_buffer{k} = pr_buffer{k};
            
            % Unbuffered
            LMC{k}=zeros(settings.vfov,settings.hfov,'single');
            on_chan{k} = LMC{k};
            off_chan{k} = LMC{k};
            
            alpha_on{k}=zeros(settings.vfov,settings.hfov,'single');
            alpha_off{k}=alpha_on{k};
            
            alpha_on_mask{k}=false(settings.vfov,settings.hfov);
            alpha_off_mask{k}=alpha_on_mask{k};
            
            % 2-buffer
            delay_on_inbuff{k} = zeros(settings.vfov,settings.hfov,2,'single');
            delay_off_inbuff{k} = delay_on_inbuff{k};
            delay_on_outbuff{k} = delay_on_inbuff{k};
            delay_off_outbuff{k} = delay_on_inbuff{k};
            
            fdsr_on_inbuff{k} = delay_on_inbuff{k};
            fdsr_off_inbuff{k} = delay_on_inbuff{k};
            fdsr_on_outbuff{k} = delay_on_inbuff{k};
            fdsr_off_outbuff{k} = delay_on_inbuff{k};
            
            % Long output buffers
            dark_output_buffer{k} = zeros(settings.vfov,settings.hfov,settings.output_integration,'single');
            light_output_buffer{k} = dark_output_buffer{k};
            
        end
        
        % onepath=linspace(-hfov/2,hfov/2,500);
        % t_th_trajectory=repmat(onepath(:),5,1);
        %
        % [tix_todo,num_targets]=size(t_th_trajectory);
        
        if(settings.make_histograms)
            num_targets=numel(t_al_pos_range);
            t_th_pos=zeros(num_targets,1);
            
            t_value = zeros(num_targets,1);
            
            spin_obs_vel=60;
            spin_obs_angle = 15;
            spin_ticks=ceil(spin_obs_angle/spin_obs_vel/Ts); % Time to spin the background
            spinning=true;
            obs_destination = spin_obs_angle;
            
            spins_completed = 0;
            spins_todo = ceil(360 / spin_obs_angle);
            
            est_ticks = spins_todo* (settings.hfov/t_vel(1) / settings.Ts + spin_ticks); % Spin time + test time
            max_ticks = est_ticks * 1.2;
        end
        
        exiting_mid_test = false;
        for tix=1:max_ticks
            if(mod(tix,200) == 0)
                disp([num2str(test_ix) ': ' num2str(tix) '/' num2str(max_ticks) ' ' num2str(toc)])
                if(toc > toc_to_die)
                    disp('Exiting mid-test due to time limit')
                    exiting_mid_test = true;
                    break;
                end
            end
            % Update indices
            pr_ind=1+mod(tix-1,pr_len);
            fdsr_ind=1+mod(tix-1,2);
            out_ind=1+mod(tix-1,settings.output_integration);
            
            % ########################
            %   Generate a new frame
            % ########################
            
            % Update target positions
            if(settings.make_histograms)
                if(spinning)
                    if(obs_th_pos < obs_destination)
                        obs_th_pos = obs_th_pos + spin_obs_vel*Ts;
                        t_al_pos=90*ones(num_targets,1);
                    else
                        spinning=false;
                        obs_destination = obs_destination + spin_obs_angle;
                        t_al_pos = t_al_pos_range;
                        t_th_pos = -settings.hfov/2 * ones(num_targets,1) + obs_th_pos;
                    end
                else
                    % Targets move left to right
                    t_th_pos = t_th_pos + t_vel*Ts;
                    if(t_th_pos > obs_th_pos + settings.hfov/2)
                        spinning=true;
                        spins_completed=spins_completed +  1;
                        if(spins_completed > spins_todo)
                            break;
                        end
                    end
                end
            else
                if(spinning)
                    if(obs_th_pos < obs_destination)
                        obs_th_pos = obs_th_pos + spin_obs_vel*settings.Ts;
                        t_al_pos=90*ones(num_targets,1);
                    else
                        spin_stop_tick = tix-1;
                        spinning=false;
                        last_tracked=tix;
                        
                        th_offset = -settings.hfov/2 + obs_th_pos; % The left side of the screen when the target appears
                        
                        traj_step = tix - spin_stop_tick;
                        
                        if(settings.target_rotational)
                            t_th_pos = traj.t_th_traj(traj_step) + th_offset;
                            t_al_pos = traj.t_al_traj(traj_step);
                        elseif(settings.target_position_based)
                            % TODO: write conversion between current
                            % heading, global position etc
                        end
                    end
                else
                    traj_step = tix - spin_stop_tick;
                    if(settings.target_rotational)
                        t_th_pos = traj.t_th_traj(traj_step) + th_offset;
                        t_al_pos = traj.t_al_traj(traj_step);
                    end
                end
            end
            %         t_th_pos = t_th_trajectory(tix,:);
            
            logging.saccading_history(tix) = executing_saccade;
            
            if(settings.do_saccades)
                if(executing_saccade)
                    % Currently turning as part of a saccade, so update the
                    % observer position accordingly
                    % This is relative so that it operates correctly bearing in mind the clamping of obs_th_pos
                    obs_th_pos = obs_th_pos + saccade_traj(saccade_index)-saccade_traj(saccade_index-1);
                    saccade_index = saccade_index + 1;
                    if(saccade_index > numel(saccade_traj))
                        % Reached the end of the saccade
                        executing_saccade = false;
                        saccade_end_tix = tix;
                        distractors.to_init=true;
                    end
                end
            end
            
            if(distractors.to_init && distractorparams.draw_distractors)
                distractors.initialised=true;
                % Create a new set of distractors
                b = 2*(2-randi(2,[distractorparams.num_distractors,1]))-1; % Left or right (-1 or 1)
                % Set the horizontal distances
                distractors.hd = b.*(distractorparams.min_distractor_hd + (distractorparams.max_distractor_hd - distractorparams.min_distractor_hd)*rand([distractorparams.num_distractors 1]));
                % Calculate the actual minimum forward distances distances
                distractor_forward_real_min = abs(distractors.hd(:)) ./ tand(settings.hfov/2);
                distractor_forward_floor = max(distractor_forward_real_min(:),distractorparams.min_distractor_forward);
                % Set forward distance
                distractors.fwd = distractor_forward_floor + (distractorparams.max_distractor_forward - distractor_forward_floor) .* rand([distractorparams.num_distractors,1]);
                % Set distractor luminance
                distractors.luminance = distractorparams.luminance_min + (distractorparams.luminance_max - distractorparams.luminance_min) .* rand([distractorparams.num_distractors,1]);
                % Set distractor elevation
                distractors.elevation = distractorparams.max_distractor_elevation_mag*(-1 + 2*rand([distractorparams.num_distractors,1]));
                distractors.to_init = false;
            end
            
            % Ensure obs_th_pos is kept in the range -180 180)
            if(obs_th_pos < -180)
                obs_th_pos = obs_th_pos + 360;
            elseif(obs_th_pos > 180)
                obs_th_pos = obs_th_pos - 360;
            end
            
            if(~settings.make_histograms)
                logging.obs_th_pos_history(tix) = obs_th_pos;
            end
            
            t_th_rel=t_th_pos-repmat(obs_th_pos,size(t_th_pos));
            % Corce t_th_rel into [0 360]
            t_th_rel = mod(180+t_th_rel,360)-180;
            logging.t_th_rel_history(tix)=t_th_rel;
            
            h_pixelpos=round(hdist*tand(t_th_rel)+hpix/2);
            v_pixelpos=round(hdist*tand(t_al_pos)./cosd(t_th_rel)+vpix/2);
            
            ind_logic=false(vpix,hpix);
            
            % Background
            offset=floor(obs_th_pos/360*panpix);
            
            ind=sub2ind(size(src),V(:),round(offset+H(:))); % Select the appropriate pixels to fill the background
            ind(isnan(ind))=1;
            img=reshape(src(ind),vpix,hpix); % img now contains the background
            img_white = ones(size(img)); % Pure white background
            img_distractors = ones(size(img));
            
            % Draw targets on to main image and img_white
            for targ_ix=1:num_targets % For each target
                if(t_th_rel(targ_ix) > -settings.hfov/2 && t_th_rel(targ_ix) < settings.hfov/2 && t_al_pos(targ_ix) < settings.vfov/2 && t_al_pos(targ_ix) > -settings.vfov/2) % Check whether the target should be drawn
                    if(strcmp(drawmode,'pixels')) %In this mode, the target has a constant pixel size, which means the angles change as it goes. Movement is by angles.
                        % Draw the target
                        x_left= max(h_pixelpos(targ_ix)-t_pixelleft,1);
                        x_right= min(h_pixelpos(targ_ix)+t_pixelright,hpix);
                        y_up= max(v_pixelpos(targ_ix)-t_pixelup,1);
                        y_down= min(v_pixelpos(targ_ix)+t_pixeldown,vpix);
                        ind_logic=false(size(ind_logic));
                        ind_logic(y_up:y_down,x_left:x_right)=true;
                    end
                    img(ind_logic)=t_value(targ_ix);
                    img_white(ind_logic)=t_value(targ_ix);
                end
            end
            
            % Draw the distractors on to main image and img_distractors
            if(distractorparams.draw_distractors && distractors.initialised)
                
                dist_th_rel = atand(distractors.hd(:) ./ distractors.fwd(:));
                dist_al_rel = atand(distractors.elevation(:) ./ (distractors.hd(:).^2 + distractors.fwd(:).^2).^0.5);
                
                dist_h_pixelpos = round(hdist*tand(dist_th_rel)+hpix/2);
                dist_v_pixelpos = round(hdist*tand(dist_al_rel)./cosd(dist_al_rel)+vpix/2);
                
                for distract_ix = 1:distractorparams.num_distractors
                    
                    x_left = max(dist_h_pixelpos(distract_ix)-t_pixelleft,1);
                    x_right = min(dist_h_pixelpos(distract_ix)+t_pixelright,hpix);
                    y_up = max(dist_v_pixelpos(distract_ix)-t_pixelup,1);
                    y_down = min(dist_v_pixelpos(distract_ix)+t_pixeldown,vpix);
                    
                    ind_logic = false(size(ind_logic));
                    ind_logic(y_up:y_down,x_left:x_right) = true;
                    img(ind_logic)= distractors.luminance(distract_ix);
                    img_distractors(ind_logic) = distractors.luminance(distract_ix);
                end
                
                % Move distractors
                distractors.fwd = distractors.fwd - distractorparams.assumed_fwd_vel*settings.Ts;
            end
            
            logging.distractors_init_history(tix) = distractors.initialised;
            
            if(~settings.Use_ZB_ESTMD)
                % Blur and subsample
                frame=zeros(size(h));
                frame_white=frame;
                if(distractorparams.draw_distractors)
                    frame_distractors=frame;
                end
                for pix_ix=1:numel(h)
                    ref_h=h(pix_ix);
                    ref_v=v(pix_ix);
                    frame(pix_ix)=sum(sum(img(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
                        ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));

                    frame_white(pix_ix)=sum(sum(img_white(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
                        ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));

                    if(distractorparams.draw_distractors)
                        frame_distractors(pix_ix)=sum(sum(img_distractors(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
                            ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));
                    end
                end

                input_buffer{1}(:,:,pr_ind) = frame;
                input_buffer{2}(:,:,pr_ind) = frame_white;
                if(distractorparams.draw_distractors)
                    input_buffer{3}(:,:,pr_ind) = frame_distractors;
                end

                % ########################
                % End of frame generation
                % ########################

                % For each background variant
                for k=1:settings.num_simultaneous

                    % PHOTORECEPTOR FILTER
                    den_ix=[pr_ind-1:-1:1 pr_len:-1:pr_ind+1];
                    num_ix=[pr_ind:-1:1 pr_len:-1:pr_ind+1];

                    pr_buffer{k}(:,:,pr_ind)= -sum(pr_buffer{k}(:,:,den_ix).*pr_den_array(:,:,2:pr_len),3)+...
                        sum(input_buffer{k}(:,:,num_ix).*pr_num_array(:,:,:),3);

                    % LMC and highpass
                    LMC{k}=conv2(padarray(pr_buffer{k}(:,:,pr_ind),[1,1],'symmetric'),lmc_kernel,'valid'); % Using same to match the simulink runs

                    % FDSR
                    % Take rectified LMC outputs
                    fdsr_on_inbuff{k}(:,:,fdsr_ind)=max(LMC{k}(:,:),0);
                    fdsr_off_inbuff{k}(:,:,fdsr_ind)=max(-LMC{k}(:,:),0);

                    alpha_on_mask{k}=fdsr_on_inbuff{k}(:,:,fdsr_ind) >= fdsr_on_outbuff{k}(:,:,3-fdsr_ind);
                    alpha_off_mask{k}=fdsr_off_inbuff{k}(:,:,fdsr_ind) >= fdsr_off_outbuff{k}(:,:,3-fdsr_ind);

                    alpha_on{k}= alpha_on_up.*alpha_on_mask{k} + alpha_on_down.*~alpha_on_mask{k};
                    alpha_off{k}= alpha_off_up.*alpha_off_mask{k} + alpha_off_down.*~alpha_off_mask{k};

                    fdsr_on_outbuff{k}(:,:,fdsr_ind)=fdsr_on_inbuff{k}(:,:,fdsr_ind).*alpha_on{k} + fdsr_on_outbuff{k}(:,:,3-fdsr_ind).*(1-alpha_on{k});
                    fdsr_off_outbuff{k}(:,:,fdsr_ind)=fdsr_off_inbuff{k}(:,:,fdsr_ind).*alpha_off{k} + fdsr_off_outbuff{k}(:,:,3-fdsr_ind).*(1-alpha_off{k});

                    % Subtraction of FDSR from input and C/S
                    on_chan{k}=max(conv2(padarray(max(fdsr_on_inbuff{k}(:,:,fdsr_ind)-fdsr_on_outbuff{k}(:,:,3-fdsr_ind),0),[pad_size pad_size],'symmetric'),cs_kernel,'valid'),0);
                    off_chan{k}=max(conv2(padarray(max(fdsr_off_inbuff{k}(:,:,fdsr_ind)-fdsr_off_outbuff{k}(:,:,3-fdsr_ind),0),[pad_size pad_size],'symmetric'),cs_kernel,'valid'),0);

                    % Delay filter
                    delay_on_inbuff{k}(:,:,fdsr_ind) = on_chan{k};
                    delay_off_inbuff{k}(:,:,fdsr_ind) = off_chan{k};

                    delay_on_outbuff{k}(:,:,fdsr_ind) = lp_num(1)* delay_on_inbuff{k}(:,:,fdsr_ind) +...
                        lp_num(2)*delay_on_inbuff{k}(:,:,3-fdsr_ind) - ...
                        lp_den*delay_on_outbuff{k}(:,:,3-fdsr_ind);

                    delay_off_outbuff{k}(:,:,fdsr_ind) = lp_num(1)* delay_off_inbuff{k}(:,:,fdsr_ind) +...
                        lp_num(2)*delay_off_inbuff{k}(:,:,3-fdsr_ind) - ...
                        lp_den*delay_off_outbuff{k}(:,:,3-fdsr_ind);

                    dark_output_buffer{k}(:,:,out_ind) = on_chan{k} .*delay_off_outbuff{k}(:,:,fdsr_ind);
                    light_output_buffer{k}(:,:,out_ind) = off_chan{k} .* delay_on_outbuff{k}(:,:,fdsr_ind);

                end
            else
                % img, img_white and img_distractors must be processed
                % using the ZB implementation of the ESTMD model
                
                % In the reference implementation the images were defined
                % with 8 bit intensity inputs whereas the rendering here
                % produces intensities in the range 0 1
                % However, in the ZB implementation the photoreceptor
                % function scaled by 1/255 anyway so just passing in the
                % 0-1 values generated here is fine.
                
                input_buffer{1} = img;
                input_buffer{2} = img_white;
                input_buffer{3} = img_distractors;
                
                dbuffer = cell(num_simultaneous,1);
                for k=1:num_simultaneous
                    dbuffer{k} = zeros(YDim, XDim, length(b_LogNormal));
                end

                for k=1:num_simultaneous
                    opticOut=OpticBlur(Input, ZB_pixels_per_degree);
                    SubSampledGreen=Subsample(opticOut, ZB_pixel2PR);
                    
                    % Cannot use ZB "PhotoReceptor" function because it
                    % used persistent variables, so had to expand it out
                    % here
%                     PR_Output=PhotoReceptor(SubSampledGreen, ZB_XDim, ZB_YDim);
                    %Lognormal filter parameters
                    b_LogNormal=[0    0.0001   -0.0011    0.0052   -0.0170    0.0439   -0.0574    0.1789   -0.1524];
                    a=[ 1.0000   -4.3331    8.6847  -10.7116    9.0004   -5.3058    2.1448   -0.5418    0.0651];
%                     persistent dbuffer1
%                     if isempty(dbuffer1)
%                        dbuffer1=zeros(YDim,XDim,length(b_LogNormal));
%                     end
                    %filter the data with a LogNormal Filter which has the same
                    %properties as PhotoReceptors
                    [PR_Output,dbuffer1]=IIRFilter(b_LogNormal,a,Green2,dbuffer1);
                    
                    LMC_Output=LMC(PR_Output, ZB_XDim,ZB_YDim);
                    RTC_Output=RTC(LMC_Output, ZB_Ts);
                    ESTMD_OUT=ESTMD(RTC_Output, ZB_Facilitation_Matrix, ZB_Facilitation_Mode, ZB_start, ZB_Threshold);
                end
                
            end
            
            % Extract masks for target responses using the white background to identify
            % target
            dark_target_mask = dark_output_buffer{2}(:,:,out_ind) > 0.01;
            light_and_dark_target_mask = (dark_output_buffer{2}(:,:,out_ind)+light_output_buffer{2}(:,:,out_ind)) > 0.01;

            dark_target_mask_buffer(:,:,out_ind)=dark_target_mask;
            light_and_dark_target_mask_buffer(:,:,out_ind)=light_and_dark_target_mask;

            if(distractorparams.draw_distractors)
                dark_distractor_mask_buffer(:,:,out_ind) = dark_output_buffer{3}(:,:,out_ind) > 0.01;
                light_and_dark_distractor_mask_buffer(:,:,out_ind) = (dark_output_buffer{3}(:,:,out_ind)+light_output_buffer{3}(:,:,out_ind))  > 0.01;
            end
            
            if(mod(tix,settings.output_integration)==0) % Time to update the tracker or histograms as the case may be
                for k=1:settings.num_simultaneous
                    % Aggregate outputs over output_integration time steps
                    dark_output_integ{k} = sum(dark_output_buffer{k},3);
                    light_output_integ{k} = sum(light_output_buffer{k},3);
                end
                integrated_tix = floor(tix/settings.output_integration);
                logging.integrated_tix_history(integrated_tix) = tix;
                
                if(settings.make_histograms)
                    % Aggregate the target outputs
                    target_response_frame=sum(dark_output_buffer{1}.*dark_target_mask_buffer,3);
                    % Create a frame showing where there were any target
                    % responses
                    target_mask_frame = sum(dark_target_mask_buffer,3) > 0;
                    % Extract aggregated target response levels
                    target_responses = target_response_frame(target_mask_frame);
                    
                    % Increment histogram
                    hist_R_v=hist_R_v + histcounts(target_responses,resp_thresholds)';
                    if(showimagery)
                        f=figure(3);
                        bar(hist_R_v)
                        drawnow
                    end
                else
                    if(~spinning && tix > spin_ticks + 30)
                        % Extract target and distractor maximum responses
                        % for logging
                        
                        % Aggregate the target outputs
                        
                        if(~settings.use_light_and_dark)
                            target_response_frame=sum(dark_output_buffer{1}.*dark_target_mask_buffer,3);
                            % Create a frame showing where there were any target
                            % responses
                            target_mask_frame = sum(dark_target_mask_buffer,3) > 0;
                            % Extract aggregated target response levels
                            target_responses = target_response_frame(target_mask_frame);
                            if(~isempty(target_responses))
                                logging.max_target_history(tix) = max(target_responses(:));
                            end
                        else
                            target_response_frame=sum((dark_output_buffer{1}+light_output_buffer{1}).*light_and_dark_target_mask_buffer,3);
                            % Create a frame showing where there were any target
                            % responses
                            target_mask_frame = sum(light_and_dark_target_mask_buffer,3) > 0;
                            % Extract aggregated target response levels
                            target_responses = target_response_frame(target_mask_frame);
                            if(~isempty(target_responses))
                                logging.max_target_history(tix) = max(target_responses(:));
                            end
                        end
                        
                        if(distractorparams.draw_distractors)
                            if(~settings.use_light_and_dark)
                                % Aggregate the distractor outputs
                                distractor_response_frame=sum(dark_output_buffer{1}.*dark_distractor_mask_buffer,3);
                                % Create a frame showing where there were any target
                                % responses
                                distractor_mask_frame = sum(dark_distractor_mask_buffer,3) > 0;
                                % Extract aggregated target response levels
                                distractor_responses = distractor_response_frame(distractor_mask_frame);
                                if(~isempty(distractor_responses))
                                    logging.max_distractor_history(tix) = max(distractor_responses(:));
                                end
                            else
                                % Aggregate the distractor outputs
                                distractor_response_frame=sum((light_output_buffer{1} + dark_output_buffer{1}).*light_and_dark_distractor_mask_buffer,3);
                                % Create a frame showing where there were any target
                                % responses
                                distractor_mask_frame = sum(light_and_dark_distractor_mask_buffer,3) > 0;
                                % Extract aggregated target response levels
                                distractor_responses = distractor_response_frame(distractor_mask_frame);
                                if(~isempty(distractor_responses))
                                    logging.max_distractor_history(tix) = max(distractor_responses(:));
                                end
                            end
                        end
                        
                        if(settings.use_prob_tracker)
                            if(acquired &&...
                                    (tix-saccade_end_tix) > settings.tix_after_saccade)
                                % Find the output which agrees best with the current
                                % hypothesis about target states
                                
                                % Calculate predicted velocity using current belief
                                % and our velocity evolution model
                                if(settings.do_velocity_update)
                                    predict.vel = conv(belief.vel,trackerparams.vel_evolution_kernel,'same');
                                else
                                    predict.vel = belief.vel;
                                end
                                
                                % Find probability that the response agrees with
                                % our position belief given our velocity beliefs
                                prob_R_pos = NaN(settings.vfov,settings.hfov);
                                %                                 old_tick=toc;
                                for pix_ix=1:numel(prob_R_pos) % For each actual pixel
                                    if(dark_output_integ{1}(pix_ix) < trackerparams.output_threshold)
                                        prob_R_pos(pix_ix)=0;
                                    else
                                        curr_resp = dark_output_integ{1}(pix_ix);
                                        resp_bin=find(curr_resp >= trackerparams.obs_model.resp_thresholds,1,'last');
                                        
                                        % sum over v of p(R|v) p(v)
                                        %                             prob_R_v(p) = sum(trackerparams.prob_R_v(resp_bin,:)' .* predict.vel);
                                        
                                        % Evaluate probability of response given
                                        % position
                                        % p(R at pos) = sum over pos of p(R | v = required) p(pos)
                                        % where v=required is a mask of velocities
                                        % required to get from the old pos to the new
                                        % ones
                                        
                                        % p(R | pos) = sum over v of p(R|v)*p(v|pos)
                                        
                                        probs_in_bin = trackerparams.prob_R_v(resp_bin,:);
                                        probs_in_bin = probs_in_bin(:) / sum(probs_in_bin(:)); % Normalise so that sum (p(R)) = 1
                                        % Create a kernel to apply to the position
                                        % belief
                                        
                                        prob_R_pos_kernel = NaN(size(trackerparams.pos_velocities));
                                        for p=1:numel(prob_R_pos_kernel)
                                            vmask = trackerparams.pos_velocities{p};
                                            % Calculate probability of the velocity
                                            % being in the required range and then
                                            % giving the response observed
                                            prob_R_pos_kernel(p) = sum(probs_in_bin .* vmask .* predict.vel); % TODO: check this
                                        end
                                        
                                        % Belief.pos is at a finer resolution than the
                                        % model outputs but we only want prob_R_pos to
                                        % be the same resolution as the outputs
                                        prob_R_pos(pix_ix) = Correl_Patch19_09_08(prob_R_pos_kernel,belief.pos,trackerparams.pix_index_map(pix_ix));
                                    end
                                end
                                %                         disp(num2str(toc - old_tick))
                                
                                % Use the most likely measurement for inference
                                [max_prob,I]=max(prob_R_pos(:));
                                
                                if(max_prob > trackerparams.likelihood_threshold)
                                    [maxpos.v,maxpos.h] = ind2sub([settings.vfov,settings.hfov],I);
                                    max_output = dark_output_integ{1}(maxpos.v,maxpos.h);
                                else
                                    max_output=0;
                                end
                                
                                % Find probability that the response agrees with
                                % our position belief
                                % For simplicity just going to integrate across the
                                % velocities that could have resulted in being on a
                                % given pixel assuming that these are all equally
                                % likely to have occurred
                                
                                % TODO: more elaborate pixel movement model based
                                % on the length of a line segment from the centre
                                % of the central pixel through another pixel at
                                % different angles
                            else
                                % Just take the maximum
                                [max_output,I]=max(dark_output_integ{1}(:));
                                [maxpos.v,maxpos.h] = ind2sub([settings.vfov,settings.hfov],I);
                            end
                            
                            if(settings.do_velocity_update)
                                % Change velocity belief based on dynamics
                                belief.vel = conv(belief.vel,trackerparams.vel_evolution_kernel,'same');
                                
                                % If the maximum model output is over threshold and we
                                % haven't turned recently then condition on it
                                if(max_output > trackerparams.output_threshold &&...
                                        (tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                        ~executing_saccade)
                                    % Condition velocity belief based on response
                                    
                                    % ############################################
                                    % Update belief about velocity based on model
                                    % output value
                                    % ############################################
                                    
                                    % Determine which resp_threshold bin to use
                                    resp_bin=find(max_output >= trackerparams.obs_model.resp_thresholds,1,'last');
                                    
                                    % Get the velocity probabilities for that bin
                                    prob_v = trackerparams.prob_R_v(resp_bin,:)';
                                    prob_v = prob_v / sum(prob_v(:));
                                    
                                    % Condition the velocity magnitude belief based on the
                                    % observation model
                                    belief.vel = trackerparams.reliability*(prob_v .* belief.vel) +(1-trackerparams.reliability)*belief.vel;
                                end
                                
                                % Normalise velocity belief
                                belief.vel = belief.vel / sum(belief.vel(:));
                            end
                            
                            % Generate a kernel based on the likelihood
                            % of our previous position beliefs transitioning to
                            % new ones
                            position_prediction_kernel = NaN(trackerparams.pixel_distance_kernel_size);
                            [Y_mini,X_mini] = ndgrid(1:trackerparams.pixel_distance_kernel_size,1:trackerparams.pixel_distance_kernel_size);
                            centre_x=ceil(trackerparams.pixel_distance_kernel_size/2);
                            centre_y=ceil(trackerparams.pixel_distance_kernel_size/2);
                            d_predict = ((trackerparams.pos_grid_spacing * (Y_mini - centre_y)).^2 + (trackerparams.pos_grid_spacing*(X_mini - centre_x)).^2 ).^0.5;
                            v_predict = d_predict / (settings.output_integration*settings.Ts);
                            
                            for p=1:numel(position_prediction_kernel)
                                % Match the prediction velocity to the
                                % discretised belief points
                                [~,vel_match] = min(abs(v_predict(p) - trackerparams.vel_pts));
                                position_prediction_kernel(p) = belief.vel(vel_match);
                            end
                            % Normalise the prediction kernel
                            position_prediction_kernel = position_prediction_kernel / sum(position_prediction_kernel(:));
                            
                            % Predict position
                            belief.pos = conv2(belief.pos,position_prediction_kernel,'same');
                            
                            if(max_output > trackerparams.output_threshold &&...
                                    ~executing_saccade &&...
                                    (tix-saccade_end_tix) > settings.tix_after_saccade)
                                % Condition position belief on the detection
                                % location
                                d=((trackerparams.Y-maxpos.v).^2 + (trackerparams.X-maxpos.h).^2).^0.5;
                                pdist=gaussmf(d,[trackerparams.pos_std 0]);
                                belief.pos = (trackerparams.reliability*belief.pos) .* ((1-trackerparams.reliability)*pdist);
                            end
                            
                            % Normalise belief.pos
                            belief.pos = belief.pos / sum(belief.pos(:));
                            
                            % Determine the degree of certainty about the states
                            [maxbelief_pos,I_pos] = max(belief.pos(:));
                            max_certainty= maxbelief_pos*max(belief.vel(:));
                            
                            if(max_certainty >= trackerparams.acquisition_threshold)
                                acquired=true;
                            elseif(maxbelief_pos < trackerparams.lost_position_threshold)
                                acquired=false;
                            end
                            
                            last_detection.v = maxpos.v;
                            last_detection.h = maxpos.h;
                            last_detection.tix = integrated_tix;
                            
                            % Pursuit logic
                            [max_y,max_x] = ind2sub(size(belief.pos),I_pos);
                            % Convert max_x into actual screen coordinates
                            max_x_screen = trackerparams.X(1,max_x);
                            
                            if(     acquired  &&...
                                    max_certainty > trackerparams.turn_threshold &&...
                                    (tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                if(max_x_screen - settings.hfov/2 > settings.turn_distance_threshold)
                                    
                                    % Since we will be blacking out a period of time
                                    % after the turn, move the belief position
                                    % predictively assuming some typical velocity
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = max_x_screen - settings.hfov/2 + predict_angle + 3;
                                    else
                                        turn_deg = max_x_screen - settings.hfov/2;
                                    end
                                    
                                    % Execute turn such that the predicted target position will be a
                                    % little to the left of center
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    % Centralise the belief position
                                    predict_deg = floor(max_x_screen - settings.hfov/2);
                                    
                                    %left_mask = trackerparams.X <= settings.hfov+(predict_deg-1);
                                    %right_mask = left_mask(end:-1:1,:);
                                    
                                    left_mask = trackerparams.X <=settings.hfov-(predict_deg-1);
                                    right_mask = left_mask(:,end:-1:1);
                                    
                                    % Shift belief leftward
                                    belief.pos(left_mask) = belief.pos(right_mask);
                                    belief.pos(~left_mask) = 0;
                                    rem = max(0,1-sum(belief.pos(:)));
                                    belief.pos(~left_mask) = rem / numel(belief.pos(~left_mask));
                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                    
                                elseif(max_x_screen - settings.hfov/2 < -settings.turn_distance_threshold)
                                    % Leftward turn
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = max_x_screen - settings.hfov/2 - predict_angle - 3;
                                    else
                                        turn_deg = max_x_screen - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    predict_deg = floor(max_x_screen - settings.hfov/2);
                                    
                                    % Shift belief rightward
                                    left_mask = trackerparams.X <= settings.hfov+(predict_deg-1);
                                    right_mask = left_mask(:,end:-1:1);
                                    belief.pos(right_mask) = belief.pos(left_mask);
                                    belief.pos(~left_mask) = 0;
                                    rem = max(0,1-sum(belief.pos(:)));
                                    belief.pos(~right_mask) = rem / numel(belief.pos(~right_mask));
                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                end
                            end
                            
                            logging.max_certainty_history(tix) = max_certainty;
                            logging.maxbelief_pos_history(tix) = maxbelief_pos;
                            logging.max_output_history(tix) = max_output;
                            
                        elseif(settings.use_facilitation_tracker)
                            
                            % Multiply ESTMD output (summed light and dark) with low-pass filtered facilitation
                            if(settings.use_light_and_dark)
                                output_facilitated = (light_output_integ{1}+dark_output_integ{1}).*fac.output_buffer(:,:,3-fac.ind);
                            else
                                output_facilitated = dark_output_integ{1}.*fac.output_buffer(:,:,3-fac.ind);
                            end
                            
                            % Create low-pass filtered version of image
                            fac.emd_input_buffer(:,:,3-fac.ind) = output_facilitated;
                            fac.emd_output_buffer(:,:,fac.ind)=...
                                fac.emd_n1 * fac.emd_input_buffer(:,:,fac.ind) +...
                                fac.emd_n2 * fac.emd_input_buffer(:,:,3-fac.ind) +...
                                fac.emd_d2 * fac.emd_output_buffer(:,:,3-fac.ind);
                            
                            % Left-right EMD (delayed left by undelayed
                            % right minus delayed right by undelayed left)
                            fac.emd_LR_result = fac.emd_output_buffer(:,1:end-1,fac.ind) .* fac.emd_input_buffer(:,2:end,fac.ind) -... % undelayed right . delayed left
                                fac.emd_output_buffer(:,2:end,fac.ind) .* fac.emd_input_buffer(:,1:end-1,fac.ind); % delayed right with undelayed left
                            
                            % Up-down EMD (delayed top with undelayed
                            % bottom -...
                            fac.emd_UD_result = fac.emd_output_buffer(1:end-1,:,fac.ind) .* fac.emd_input_buffer(2:end,:,fac.ind) -...
                                fac.emd_output_buffer(2:end,:,fac.ind) .* fac.emd_input_buffer(1:end-1,:,fac.ind);
                            
                            % Calculate direction output
                            fac.direction_leftright = max(fac.emd_LR_result(:)) + min(fac.emd_LR_result(:)); % > 0 -> right
                            fac.direction_updown = max(fac.emd_UD_result(:)) + min(fac.emd_UD_result(:)); % > 0 -> down
                            
                            % Get maximum position based on facilitated ESTMD outputs,
                            % then predict based on EMD outputs
                            [max_output,I]=max(output_facilitated(:));
                            [maxpos.v,maxpos.h] = ind2sub([settings.vfov,settings.hfov],I);
                            
                            % Parse left/right EMD
                            if(fac.direction_leftright > facparams.direction_predict_threshold)
                                % Moving right
                                inpt.x = maxpos.h + facparams.direction_predict_distance;
                            elseif(fac.direction_leftright < -facparams.direction_predict_threshold)
                                % moving left
                                inpt.x = maxpos.h - facparams.direction_predict_distance;
                            else
                                % No meaningful direction so just apply fac
                                % where the maximum is
                                inpt.x = maxpos.h;
                            end
                            
                            % Parse up/down EMD
                            if(fac.direction_updown > facparams.direction_predict_threshold)
                                % Moving right
                                inpt.y = maxpos.v + facparams.direction_predict_distance;
                            elseif(fac.direction_updown < -facparams.direction_predict_threshold)
                                % moving left
                                inpt.y = maxpos.v - facparams.direction_predict_distance;
                            else
                                % No meaningful direction so just apply fac
                                % where the maximum is
                                inpt.y = maxpos.v;
                            end
                            
                            fac.matrix = zeros(settings.vfov,settings.hfov);
                            fac.patch_gain=NaN(numel(fac.kernels_y),numel(fac.kernels_x));
                            
                            for kh=1:numel(fac.kernels_x)
                                for kv=1:numel(fac.kernels_y)
                                    % For the patch, evaluate distance to input point
                                    d_input = norm([fac.kernels_x(kh)-inpt.x, fac.kernels_y(kv)-inpt.y]);
                                    fac.patch_gain(kv,kh)=gaussmf(d_input,[facparams.sigma 0]);
                                end
                            end
                            
                            % Apply the patches
                            for kh=1:numel(fac.kernels_x)
                                for kv=1:numel(fac.kernels_y)
                                    left_x = max(fac.kernels_x(kh)-facparams.kernel_spacing,1);
                                    right_x = min(fac.kernels_x(kh)+(facparams.kernel_spacing-1),settings.hfov);
                                    top_y = max(fac.kernels_y(kv)-facparams.kernel_spacing,1);
                                    bottom_y = min(fac.kernels_y(kv)+(facparams.kernel_spacing-1),settings.vfov);
                                    fac.matrix(top_y:bottom_y,left_x:right_x) = fac.matrix(top_y:bottom_y,left_x:right_x)+fac.patch_gain(kv,kh)*ones(size([fac.matrix(top_y:bottom_y,left_x:right_x)]));
                                end
                            end
                            
                            % Apply facilitation gain. In ZB's implementation the gain seems to be
                            % squared because it is applied separately on
                            % signals that are subsequently multiplied.
                            % 1 is added to fac.matrix so doing that here
                            % also
                            
                            % Newly introducing a period of ignoring the
                            % model outputs after a turn
                            if(tix - saccade_end_tix > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                fac.matrix = 1 + fac.matrix * facparams.gain;
                            else
                                fac.matrix = ones(size(fac.matrix));
                            end
                            
                            % Low-pass filter the facilitation matrix
                            fac.input_buffer(:,:,fac.ind) = fac.matrix;
                            fac.output_buffer(:,:,fac.ind) = fac.input_buffer(:,:,fac.ind)*fac.lpf_n1 + fac.input_buffer(:,:,3-fac.ind)*fac.lpf_n2 + fac.lpf_d2*fac.output_buffer(:,:,3-fac.ind);
                            
                            fac.ind = 3 - fac.ind;
                            
                            % Facilitation pursuit logic
                            % Turn if the target is more than 5 degrees
                            % from center and the direction says to do so
                            
                            % Whereas in the ZB implementation that I have,
                            % the direction was calculated with
                            % max(direction(:)) + min(direction(:))
                            
                            logging.lr_dir_output_history(tix) = fac.direction_leftright;
                            logging.ud_dir_output_history(tix) = fac.direction_updown;
                            
                            logging.max_fac_output_history(tix) = max_output;
                            logging.max_fac_output_h_history(tix) = maxpos.h;
                            logging.max_fac_output_v_history(tix) = maxpos.v;
                            
                            if(max_output > facparams.output_turn_threshold)
                                if(maxpos.h - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        fac.direction_leftright > facparams.direction_turn_threshold &&...
                                        (tix - saccade_end_tix) > settings.tix_after_saccade &&...
                                        ~executing_saccade &&...
                                        (tix > spin_ticks+20))
                                    % Execute a rightward saccade/turn s.t. the
                                    % field of view is centered on the maximum
                                    % position
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = max(settings.tix_after_saccade * settings.Ts,...
                                                settings.saccade_duration+settings.tix_after_saccade*settings.Ts);
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_angle = settings.tix_after_saccade * settings.output_integration*settings.Ts * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 + predict_angle + 5;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix = tix;
                                    end
                                    
                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                    
                                elseif(maxpos.h - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        fac.direction_leftright < -facparams.direction_turn_threshold &&...
                                        (tix - saccade_end_tix) > (tix - saccade_end_tix) > settings.tix_after_saccade &&...
                                        ~executing_saccade &&...
                                        tix > spin_ticks+20)
                                    % Execute a leftward saccade s.t. the
                                    % field of view is centered on the maximum
                                    % position
                                    
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = max(settings.tix_after_saccade * settings.output_integration*settings.Ts,...
                                                settings.saccade_duration+settings.tix_after_saccade*settings.Ts);
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_angle = settings.tix_after_saccade * settings.output_integration*settings.Ts * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 - predict_angle - 5;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix = tix;
                                    end
                                    
                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                end
                            end
                        elseif(settings.use_kf_tracker)
                            % ################################
                            %       KALMAN FILTER UPDATE
                            % ################################
                            
                            % Prediction step
                            x_priori=kfparams.A*x_post;
                            P_priori=kfparams.A*P_post*kfparams.A' + kfparams.Q;
                            
                            if((tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                % Work out where the maximum output is
                                if(settings.use_light_and_dark)
                                    output_to_use = light_output_integ{1}+dark_output_integ{1};
                                else
                                    output_to_use = dark_output_integ{1};
                                end
                                [max_output,I]=max(output_to_use(:));
                                [maxpos.v,maxpos.h] = ind2sub(size(output_to_use),I);
                                
                                % Determine whether to do a measurement
                                % update or not
                                do_measurement=false;
                                
                                if(max_output >= kfparams.output_threshold)
                                    if(certainty > kfparams.certainty_threshold) % "acquired"
                                        % Compare x_priori to the maxpos and
                                        % see how well they agree
                                        dist = norm(x_priori(1:2) - [maxpos.h;maxpos.v]);
                                        if(dist > kfparams.acquired_ignore_distance)
                                            % It's too far away from our
                                            % expectations
                                            do_measurement=false;
                                            certainty = certainty - 1;
                                        else
                                            % It's close enough so we should
                                            % do a measurement update with it
                                            do_measurement=true;
                                            certainty = min(kfparams.max_certainty, certainty + 1);
                                        end
                                    else
                                        % We should do a measurement-based
                                        % update because the response was
                                        % strong and we haven't acquired
                                        % the target yet
                                        do_measurement = true;
                                        dist = norm(x_priori(1:2) - [maxpos.h;maxpos.v]);
                                        if(dist <= kfparams.acquired_ignore_distance)
                                            % We can be a bit more certain that
                                            % we are tracking the target now
                                            certainty = min(kfparams.max_certainty, certainty + 1);
                                        else
                                            % Less certain
                                            certainty = max(0,certainty - 1);
                                        end
                                    end
                                else
                                    % The response is not strong enough so
                                    % ignore it
                                    do_measurement=false;
                                    certainty = max(0,certainty - 1);
                                end

                                if(do_measurement)
                                    z = [maxpos.h; maxpos.v; 0; 0];
                                    K=P_priori*kfparams.H'*(kfparams.H*P_priori*kfparams.H'+kfparams.R)^-1;
                                    x_post=x_priori+K*(z-kfparams.H*x_priori);
                                    P_post=(eye(4)-K*kfparams.H)*P_priori;
                                else
                                    % Just using the predicted states
                                    x_post = x_priori;
                                    P_post = P_priori;
                                end
                            else
                                % Just using the predicted states because
                                % there's no measurement to use
                                x_post = x_priori;
                                P_post = P_priori;
                            end
                            
                            % ################################
                            %   KALMAN FILTER TURN DECISIONS
                            % ################################
                            
                            if((tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                if(x_post(1) - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        maxpos.h - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        max_output > kfparams.output_threshold)
                                    % Rightward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 + predict_angle + 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    predict_deg = maxpos.h - settings.hfov/2;
                                    
                                    % Centralise the horizontal state
                                    x_post(1) = x_post(1) - predict_deg;
                                    distractors.initialised=false;
                                    
                                elseif(x_post(1) - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        maxpos.h - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        max_output > kfparams.output_threshold)
                                    % Leftward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 - predict_angle - 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    predict_deg = maxpos.h - settings.hfov/2;
                                    
                                    % Shift states rightward
                                    x_post(1) = x_post(1) - predict_deg;
                                    distractors.initialised=false;
                                end
                            end
                            
                            logging.kf.certainty_history(tix) = certainty;
                            logging.kf.x_post(tix,:) = x_post;
                            logging.max_output_history(tix) = max_output;
                        elseif(settings.use_constgain_tracker)
                            
                            if(settings.use_light_and_dark)
                                output_to_use = light_output_integ{1}+dark_output_integ{1};
                            else
                                output_to_use = dark_output_integ{1};
                            end
                            % Apply gain
                            output_to_use = output_to_use .* constgain.field; 
                            
                            % Find max
                            [max_output,I]=max(output_to_use(:));
                            [maxpos.v,maxpos.h] = ind2sub(size(output_to_use),I);
                            
                            % Decide whether to turn
                            if((tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                if(maxpos.h - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        max_output > constgain.output_threshold)
                                    % Rightward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 + predict_angle + 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    distractors.initialised=false;
                                    
                                elseif(maxpos.h - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        max_output > constgain.output_threshold)
                                    % Leftward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 - predict_angle - 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    distractors.initialised=false;
                                end
                            end
                        elseif(settings.use_no_tracker)
                            if(settings.use_light_and_dark)
                                output_to_use = light_output_integ{1}+dark_output_integ{1};
                            else
                                output_to_use = dark_output_integ{1};
                            end
                            [max_output,I]=max(output_to_use(:));
                            [maxpos.v,maxpos.h] = ind2sub(size(output_to_use),I);
                            
                            % Decide whether to turn
                            if((tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                if(maxpos.h - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        max_output > notrack.output_threshold)
                                    % Rightward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 + predict_angle + 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    distractors.initialised=false;
                                    
                                elseif(maxpos.h - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        max_output > notrack.output_threshold)
                                    % Leftward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 - predict_angle - 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    distractors.initialised=false;
                                end
                            end
                        elseif(settings.use_ZB_tracker)

                            
                            ZB_fac.test = 1;
                            ZB_fac.start = 1;
                            
%                            XDim = settings.hfov;
%                            YDim = settings.vfov;
                            
%                            SubSampledGreen = frame;
%                             
%                            [PR_Output,ZB_fac.dbuffer1{1}]=PhotoReceptor_feedin(SubSampledGreen, ZB_fac.XDim, ZB_fac.YDim,ZB_fac.dbuffer1{1});
%                            LMC_Output=LMC(PR_Output, ZB_fac.XDim,ZB_fac.YDim);
%                            return
%                            RTC_Output=RTC(LMC_Output, Ts);
                            
                            if(settings.use_light_and_dark)
                                RTC_Output = light_output_integ{1} + dark_output_integ{1};
                            else
                                RTC_Output = dark_output_integ{1};
                            end

                            ESTMD_OUT=ESTMD(RTC_Output, ZB_fac.Facilitation_Matrix, ZB_fac.Facilitation_Mode, ZB_fac.start, ZB_fac.Threshold);
                            
                           [ZB_fac.Direction_Horizontal,ZB_fac.Direction_Vertical]=Direction(ESTMD_OUT,  ZB_fac.RC_wb, ZB_fac.Ts,ZB_fac.XDim, ZB_fac.YDim);
                           [ col_index, row_index] = Target_Location(ESTMD_OUT, ZB_fac.YDim);
                           [default, col_index2,row_index2] = Velocity_Vector(ZB_fac.test, col_index, row_index,ZB_fac.Direction_Horizontal,ZB_fac.Direction_Vertical, ZB_fac.XDim);
%                            return
                           
%                            if i<=Delay
%                                default=1;
%                            end
% 
%                            if i<Delay+1
%                                col_index2=initialCol;
%                                row_index2=initialRow;
%                            end

                           ZB_fac.Grid=FacilitationGrid(col_index2,row_index2, ESTMD_OUT, ZB_fac.Facilitation_sigma, ZB_fac.FacilitationGain);
                           [ZB_fac.Facilitation_Matrix, ZB_fac.Delayed_Grid, ZB_fac.Facilitationbuffer]=FacilitationMatrix_feedin(ZB_fac.Grid, col_index2, default, ZB_fac.Ts, ZB_fac.wb, ZB_fac.Delayed_Grid, ZB_fac.Facilitationbuffer);
                           
                           maxpos.h = col_index;
                           maxpos.v = row_index;
                           max_output = max(ESTMD_OUT(:));
 
                           if((tix-saccade_end_tix) > settings.tix_after_saccade &&...
                                    ~executing_saccade)
                                if(maxpos.h - settings.hfov/2 > settings.turn_distance_threshold &&...
                                        max_output > ZB_fac.output_threshold)
                                    % Rightward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 + predict_angle + 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end
                                    
                                    % Remap the facilitation matrix
                                    % Centralise the belief position
                                    predict_deg = floor(maxpos.h - settings.hfov/2);
                                    
                                    left_mask = ZB_fac.X <=settings.hfov-(predict_deg-1);
                                    right_mask = left_mask(:,end:-1:1);
                                    
                                    left_mask_big = repmat(left_mask,1,1,2);
                                    right_mask_big = repmat(right_mask,1,1,2);
                                    
                                    % Shift facilitation matrix and buffers
                                    % leftward
                                    ZB_fac.Facilitation_Matrix(left_mask) = ZB_fac.Facilitation_Matrix(right_mask);
                                    ZB_fac.Facilitation_Matrix(~left_mask) = 0;
                                    ZB_fac.Delayed_Grid(left_mask) = ZB_fac.Delayed_Grid(right_mask);
                                    ZB_fac.Delayed_Grid(~left_mask) = 0;
                                    ZB_fac.Facilitationbuffer(left_mask_big) = ZB_fac.Facilitationbuffer(right_mask_big);
                                    ZB_fac.Facilitationbuffer(~left_mask_big) = 0;
                                    
                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                    
                                elseif(maxpos.h - settings.hfov/2 < -settings.turn_distance_threshold &&...
                                        max_output > ZB_fac.output_threshold)
                                    % Leftward turn required
                                    if(settings.do_predictive_turn)
                                        if(settings.do_saccades)
                                            predict_time = settings.saccade_duration+settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        else
                                            predict_time = settings.tix_after_saccade*settings.Ts;
                                            predict_angle = predict_time * 60;
                                        end
                                        turn_deg = maxpos.h - settings.hfov/2 - predict_angle - 3;
                                    else
                                        turn_deg = maxpos.h - settings.hfov/2;
                                    end
                                    
                                    if(settings.do_saccades)
                                        % Perform the turn gradually using a
                                        % trajectory reflective of saccades
                                        saccade_traj = GenerateSaccade19_09_12(obs_th_pos,obs_th_pos + turn_deg,settings.saccade_duration);
                                        saccade_index=2; % Skip the first entry because the observer position won't be updated until the next timestep
                                        executing_saccade=true;
                                        distractors.to_init = false;
                                    else
                                        obs_th_pos = obs_th_pos + turn_deg;
                                        distractors.to_init = true;
                                        saccade_end_tix=tix;
                                    end

                                    % Remap the facilitation matrix
                                    % rightward
                                    predict_deg = floor(maxpos.h - settings.hfov/2);
                                    
                                    % Shift belief rightward
                                    facnew = ones(size(ZB_fac.Facilitation_Matrix));
                                    left_mask = ZB_fac.X <= settings.hfov+(predict_deg-1);
                                    right_mask = left_mask(:,end:-1:1);
                                    
                                    left_mask_big = repmat(left_mask,1,1,2);
                                    right_mask_big = repmat(right_mask,1,1,2);
                                    
                                    ZB_fac.Facilitation_Matrix(right_mask) = ZB_fac.Facilitation_Matrix(left_mask);
                                    ZB_fac.Facilitation_Matrix(~right_mask) = 0;
                                    ZB_fac.Delayed_Grid(right_mask) = ZB_fac.Delayed_Grid(left_mask);
                                    ZB_fac.Delayed_Grid(~right_mask) = 0;
                                    ZB_fac.Facilitationbuffer(right_mask_big) = ZB_fac.Facilitationbuffer(left_mask_big);
                                    ZB_fac.Facilitationbuffer(~right_mask_big) = 0;

                                    last_turn_integrated = integrated_tix;
                                    last_turn_tix = tix;
                                    distractors.initialised=false;
                                end
                           end
                        end
                    end
                    
                    % Detect a loss of tracking to terminate the trial early,
                    % if appropriate
                    if(~settings.make_histograms && tix > spin_ticks)
                        if(abs(t_th_rel) < settings.hfov/2) % Positioned in the screen
                            last_tracked=tix;
                        else
                            if(tix - last_tracked > 500)
                                % Tracking was lost more than 200 ticks ago
                                break;
                            end
                        end
                    end
                    
                    if(settings.showimagery && tix > spin_ticks + 30)
                        f=figure(1);
                        if(settings.use_prob_tracker)
                            subplot(4,2,1)
                            imagesc(belief.pos)
                            title('belief position')
                            subplot(4,2,2)
                            plot(trackerparams.vel_pts,belief.vel,'b-',[1 1]*traj.t_vel_traj(tix),[0 max(belief.vel(:))],'r-')
                            title('Velocity belief')
                            subplot(4,2,5)
                            plot(logging.max_certainty_history(~isnan(logging.max_certainty_history)))
                            if(acquired)
                                title('acquired')
                            else
                                title('not acquired')
                            end
                            subplot(4,2,6)
                            plot(logging.max_output_history(~isnan(logging.max_output_history)))
                            
                        elseif(settings.use_facilitation_tracker)
                            subplot(4,2,1)
                            imagesc(fac.output_buffer(:,:,fac.ind))
                            title('Facilitation matrix')
                            subplot(4,2,2)
                            plot(logging.max_fac_output_history)
                            subplot(4,2,5)
                            imagesc(fac.emd_LR_result);
                            title('L-R EMD')
                            subplot(4,2,6)
                            imagesc(fac.emd_UD_result);
                            title('U-D EMD')
                        elseif(settings.use_kf_tracker)
                            subplot(4,2,1)
                            temp_img=zeros([settings.vfov, settings.hfov]);
                            x_draw = min(max(1,floor(x_post(1))),settings.hfov);
                            y_draw = min(max(1,floor(x_post(2))),settings.vfov);
                            temp_img(y_draw,x_draw)=1;
                            imagesc(temp_img)
                            title('Estimated position')
                            subplot(4,2,2)
                            plot(logging.max_output_history(~isnan(logging.max_output_history)))
                            title('Output history')
                            subplot(4,2,5)
                            plot(logging.kf.certainty_history(~isnan(logging.kf.certainty_history)))
                            title('Certainty')
                        elseif(settings.use_ZB_tracker)
                            subplot(4,2,1)
                            imagesc(ZB_fac.Facilitation_Matrix)
                            subplot(4,2,2)
                            temp_img=zeros([settings.vfov, settings.hfov]);
                            x_draw = min(max(1,floor(col_index)),settings.hfov);
                            y_draw = min(max(1,floor(row_index)),settings.vfov);
                            temp_img(y_draw,x_draw)=1;
                            imagesc(temp_img)
                        end
                        subplot(4,2,3)
                        imagesc(dark_output_integ{1})
                        title('output')
                        subplot(4,2,4)
                        imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
                        title('input')
                        subplot(4,2,7)
                        plot(logging.t_th_rel_history)
                        subplot(4,2,8)
                        imshow(img_white)
                        if(distractors.initialised)
                            title('Distractors on')
                        elseif(executing_saccade)
                            title('Saccading')
                        end
                        drawnow
                        
                    end
                end
            end
            %         if(~settings.make_histograms)
            %             if(settings.use_prob_tracker)
            %                 settings.trackerparams = trackerparams;
            %             elseif(settings.use_facilitation_tracker)
            %                 settings.facparams = facparams;
            %             end
            %
            %             save(savename,...
            %                 'traj','settings','logging')
            %         end
            
            
            
            if(settings.make_histograms)
                save([outdir 'integ_' num2str(settings.output_integration) '-' num2str(t_vel_range(test_ix)) '.mat'],'t_vel','hist_R_v','resp_thresholds','output_integration');
            end
        end
        if(~exiting_mid_test)
            outer_log.data{test_ix}=logging;
            outer_log.completed(test_ix) = true;
            outer_log.completion_times(test_ix)= toc - toc_commencing;
        end
    end
end

if(~nothingchanged)
    outer_log.settings = settings;
    outer_log.total_time_spent = sum(outer_log.completion_times,'omitnan');
    outer_log.est_time_to_completion = (num_tests - sum(outer_log.completed(:)))/sum(outer_log.completed(:)) * outer_log.total_time_spent;
    save(savename,'outer_log');
end
