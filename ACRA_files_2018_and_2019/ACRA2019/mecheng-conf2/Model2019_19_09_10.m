% function Model2019_19_09_04(grp_index,grp_file,groupname)

% pixel2PR, sigma_hw_pixel, cs_kernel_size to be specified in group sheet
clear all
close all

traj_range =  1:100;

pixel2PR=12;
sigma_hw_degree=1.4;
cs_kernel_size=5;
output_integration=4; % How many output frames to aggregate when running tracker
make_histograms=false;
showimagery=false;
hfov = 40;
vfov = 20;
Ts=0.001;

which_pc='mecheng';

if(strcmp(which_pc,'mrblack'))
    trajfilename='E:\PHD\conf2\data\trackerinfo\trajectories19_09_06.mat';
    impath='E:\PHD\texture\HDR_Botanic_RGB.png';
    %     obs_model_filename = 'E:\PHD\conf2\data\trackerinfo\obs_model_botanic.mat';
    obs_model_filename = 'E:\PHD\conf2\data\trackerinfo\obs_model_4_botanic.mat';
    outdir='E:\PHD\conf2\data\results\';
elseif(strcmp(which_pc,'mecheng'))
    %     trajfilename='D:\simfiles\conf2\trackerinfo\trajectories_inscreen_19_09_09.mat';
    trajfilename='D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat';
    impath='D:\simfiles\texture\HDR_Botanic_RGB.png';
    %     obs_model_filename = 'E:\PHD\conf2\data\trackerinfo\obs_model_botanic.mat';
    obs_model_filename = 'D:\simfiles\conf2\trackerinfo\obs_model_4_botanic.mat';
    outdir='D:\simfiles\conf2\results\';
end

settings.pixel2PR = pixel2PR;
settings.sigma_hw_degree = sigma_hw_degree;
settings.cs_kernel_size = cs_kernel_size;
settings.output_integration = output_integration;
settings.make_histograms = make_histograms;
settings.showimagery = showimagery;
settings.hfov = hfov;
settings.vfov = vfov;
settings.Ts = Ts;
settings.which_pc = which_pc;
settings.impath=impath;
settings.trajfilename=trajfilename;
settings.obs_model_filename = obs_model_filename;

% impath = 'D:\simfiles\texture\HDR_Botanic_RGB.png';

if(make_histograms)
    %     t_vel_range=20:40:400;
    t_vel_range=20:20:800;
    num_tests=numel(t_vel_range);
else
    %     traj_range = 1;
    num_tests = numel(traj_range);
    trajfile = load(trajfilename);
    
    trackerparams.obs_model = load(obs_model_filename);
    
    % Tracker parameters
    trackerparams.pos_std = 0.5*output_integration; % Gaussian around the observed location
    trackerparams.output_threshold = 0.002*output_integration; % Model outputs <= this are ignored.
    trackerparams.acquisition_threshold = 0.5e-5; % What the max probability has to evaluate out to before we assume a target is acquired
    trackerparams.vel_std = output_integration * Ts * 80; % How much the velocity may have changed in the time since the last tracker tick
    trackerparams.vel_resolution = trackerparams.vel_std; % Resolve velocity belief at a resolution equal to one standard deviation of our velocity evolution model
    trackerparams.vel_pts = 0:trackerparams.vel_resolution:max(trackerparams.obs_model.t_vel_range);
    trackerparams.turn_threshold = 1e-5;
    trackerparams.reliability = 0.99; % An assumption about how likely the measurement is to correspond to the target
    trackerparams.lost_position_threshold = 0.002;
    trackerparams.min_turn_gap = 10;
    trackerparams.tix_ignore_after_turn = floor(0.05 / Ts / output_integration);
    
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
    [trackerparams.Y,trackerparams.X] = ndgrid(1:trackerparams.pos_grid_spacing:vfov,1:trackerparams.pos_grid_spacing:hfov); % Grid coordinates for screen
    
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
        trackerparams.vmin(p) = dmin(p) / (output_integration*Ts);
        trackerparams.vmax(p) = dmax(p) / (output_integration*Ts);
    end
    
    % Masks for which velocities are required to move from pixel to pixel
    % in output_integration*Ts seconds
    for p=1:numel(trackerparams.pos_velocities)
        trackerparams.pos_velocities{p} = (trackerparams.vel_pts >= trackerparams.vmin(p) & trackerparams.vel_pts <= trackerparams.vmax(p) )';
    end
    
    % Need a mapping of the actual pixel coordinates into belief.pos
    trackerparams.pix_index_map = NaN(vfov,hfov); % The entries in this should be the indices of belief.pos that correspond to the pixels
    for p=1:vfov*hfov
        [ypos,xpos] = ind2sub([vfov hfov], p);
        mask = trackerparams.Y == ypos & trackerparams.X == xpos;
        trackerparams.pix_index_map(p) = find(mask == true,1,'first');
    end
    
    trackerparams.likelihood_threshold = 1e-8;
end

% bg_edges=[0 logspace(-12,0,100)];

% locroot='D:\simfiles\conf2\';
% groupname='test';
%
% dataDir=[locroot groupname '/data/'];
% if(~exist(dataDir,'dir'))
%     mkdir(dataDir);
% end
%
% savename=[dataDir 'gind' num2str(grp_index) '.mat'];
% if(exist(savename,'file'))
%     disp(['Found existing ' savename '. Exiting.'])
%     return
% end

pad_size=floor(cs_kernel_size/2);

% Set up rendering

t_width = 1; % Degrees
t_height = 1;
t_value = 0;

fwhm = 1.4; % blur half-width, in degrees
frames=1;
delay=1;
drawmode = 'pixels';
% t_al_pos = [0 9]; % Vertical angle, 1 entry per target
% t_th_pos = [0 10]; % Horizontal angle, 1 entry per target (absolute angle)
pixel_per_degree = 12;
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
    hfov,...
    vfov,...
    impath,...
    t_width,...
    t_height,...
    pixel_per_degree,...
    degree_per_PR,...
    fwhm);

% Combined photoreceptor and high pass filter coefficients
pr_len=single(10);
pr_num=zeros(1,1,pr_len,'single');
pr_num(:)=[0    0.0001   -0.0012    0.0063   -0.0222    0.0609   -0.1013    0.2363   -0.3313    0.1524];
pr_den=zeros(1,1,pr_len,'single');
pr_den(:)=[ 1.0000   -5.1664   12.2955  -17.9486   17.9264  -12.8058    6.5661   -2.3291    0.5166   -0.0542];

pr_num_array=repelem(pr_num,vfov,hfov,1);
pr_den_array=repelem(pr_den,vfov,hfov,1);

% LMC kernel
lmc_kernel = single(1/9*[-1 -1 -1; -1 8 -1; -1 -1 -1]);

% NLAM
tau_on_up=0.01;
tau_on_down=0.1;
tau_off_up=0.01;
tau_off_down=0.1;

alpha_on_up=single(Ts/(Ts+tau_on_up));
alpha_on_down=single(Ts/(Ts+tau_on_down));
alpha_off_up=single(Ts/(Ts+tau_off_up));
alpha_off_down=single(Ts/(Ts+tau_off_down));

% C/S kernel

cs_kernel=zeros(cs_kernel_size);

cs_kernel_val=-16/(4*cs_kernel_size-4);

cs_kernel(:,1)      = cs_kernel_val;
cs_kernel(1,:)      = cs_kernel_val;
cs_kernel(:,end)    = cs_kernel_val;
cs_kernel(end,:)    = cs_kernel_val;

cs_kernel((cs_kernel_size+1)/2,(cs_kernel_size+1)/2)=2;

cs_kernel=single(cs_kernel);

% Low-pass "delay" filter
delay_lp=0.025;
lp_den_raw=[(1+2*delay_lp/Ts), (1-2*delay_lp/Ts)];
lp_num=single([1/lp_den_raw(1) 1/lp_den_raw(1)]);

lp_den=single(lp_den_raw(2)/lp_den_raw(1));

% Setup buffers;
pr_buffer = cell(2,1);
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

for test_ix = 1:num_tests;
    
    proceedwith=true;
    if(~make_histograms)
        savename=[outdir 'track_' num2str(output_integration) '-' num2str(traj_range(test_ix)) '.mat'];
        if(exist(savename,'file'))
            disp(['Skipping ' savename])
            proceedwith=false;
        end
    end
    
    if(proceedwith)
        if(make_histograms)
            %     resp_levels=20;
            resp_thresholds=[0:0.01:0.5 100];
            t_al_pos_range=[-vfov/2+2 0 vfov/2-2]';
            num_targets=numel(t_al_pos_range);
            obs_th_pos=0;
            
            % A target will be drawn at each of these angles
            hist_R_v = zeros(numel(resp_thresholds)-1,1); % Create single histograms for velocities and combine these later
            t_vel = t_vel_range(test_ix)*ones(num_targets,1);
        else
            % Actually following a trajectory, so just load that
            traj_current = traj_range(test_ix);
            traj=trajfile.traj{traj_current};
            
            obs_th_pos=0;
            
            % Spin a bit to initialise the filters and then introduce the
            % target along the set trajectory
            spin_obs_vel=60;
            spin_obs_angle=15;
            spin_ticks=ceil(spin_obs_angle/spin_obs_vel/Ts); % Time to spin the background
            spinning=true;
            obs_destination = spin_obs_angle;
            max_ticks = numel(traj.t_th_traj) + spin_ticks;
            
            num_targets=1;
            t_th_pos=0;
            
            % Initial settings for the tracker
            acquired=false;
            % Just assume uniform priors for everything for now
            
            belief.angle = ones(numel(angle_pts),1) / numel(angle_pts);
            
            belief.vel = ones(numel(trackerparams.vel_pts),1) / numel(trackerparams.vel_pts);
            belief.pos = ones(size(trackerparams.Y)) / numel(trackerparams.Y);
            
            max_certainty_history = NaN(max_ticks,1);
            maxbelief_pos_history = max_certainty_history;
            max_output_history = max_certainty_history;
            t_th_rel_history = max_certainty_history;
            last_turn = 0;
            obs_th_pos_history = obs_th_pos;
        end
        
        dark_target_responses_buffer = zeros(vfov,hfov,output_integration,'single');
        light_target_responses_buffer = dark_target_responses_buffer;
        
        dark_target_mask_buffer = false(vfov,hfov,output_integration);
        light_target_mask_buffer = dark_target_mask_buffer;
        
        for k=1:2
            % Long PR and input image buffer
            pr_buffer{k} = zeros(vfov,hfov,10,'single');
            input_buffer{k} = pr_buffer{k};
            
            % Unbuffered
            LMC{k}=zeros(vfov,hfov,'single');
            on_chan{k} = LMC{k};
            off_chan{k} = LMC{k};
            
            alpha_on{k}=zeros(vfov,hfov,'single');
            alpha_off{k}=alpha_on{k};
            
            alpha_on_mask{k}=false(vfov,hfov);
            alpha_off_mask{k}=alpha_on_mask{k};
            
            % 2-buffer
            delay_on_inbuff{k} = zeros(vfov,hfov,2,'single');
            delay_off_inbuff{k} = delay_on_inbuff{k};
            delay_on_outbuff{k} = delay_on_inbuff{k};
            delay_off_outbuff{k} = delay_on_inbuff{k};
            
            fdsr_on_inbuff{k} = delay_on_inbuff{k};
            fdsr_off_inbuff{k} = delay_on_inbuff{k};
            fdsr_on_outbuff{k} = delay_on_inbuff{k};
            fdsr_off_outbuff{k} = delay_on_inbuff{k};
            
            % Long output buffers
            dark_output_buffer{k} = zeros(vfov,hfov,output_integration,'single');
            light_output_buffer{k} = dark_output_buffer{k};
            
        end
        
        % onepath=linspace(-hfov/2,hfov/2,500);
        % t_th_trajectory=repmat(onepath(:),5,1);
        %
        % [tix_todo,num_targets]=size(t_th_trajectory);
        
        if(make_histograms)
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
            
            est_ticks = spins_todo* (hfov/t_vel(1) / Ts + spin_ticks); % Spin time + test time
            max_ticks = est_ticks * 1.2;
        end
        
        tic
        for tix=1:max_ticks
            if(mod(tix,200) == 0)
                disp([num2str(tix) '/' num2str(max_ticks) ' ' num2str(toc)])
            end
            % Update indices
            pr_ind=1+mod(tix-1,pr_len);
            fdsr_ind=1+mod(tix-1,2);
            out_ind=1+mod(tix-1,output_integration);
            
            % ########################
            %   Generate a new frame
            % ########################
            
            % Update target positions
            if(make_histograms)
                if(spinning)
                    if(obs_th_pos < obs_destination)
                        obs_th_pos = obs_th_pos + spin_obs_vel*Ts;
                        t_al_pos=90*ones(num_targets,1);
                    else
                        spinning=false;
                        obs_destination = obs_destination + spin_obs_angle;
                        t_al_pos = t_al_pos_range;
                        t_th_pos = -hfov/2 * ones(num_targets,1) + obs_th_pos;
                    end
                else
                    % Targets move left to right
                    t_th_pos = t_th_pos + t_vel*Ts;
                    if(t_th_pos > obs_th_pos + hfov/2)
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
                        obs_th_pos = obs_th_pos + spin_obs_vel*Ts;
                        t_al_pos=90*ones(num_targets,1);
                    else
                        spin_stop_tick = tix-1;
                        spinning=false;
                        
                        th_offset = -hfov/2 + obs_th_pos; % The left side of the screen when the target appears
                        
                        traj_step = tix - spin_stop_tick;
                        t_th_pos = traj.t_th_traj(traj_step) + th_offset;
                        t_al_pos = traj.t_al_traj(traj_step);
                        
                    end
                else
                    traj_step = tix - spin_stop_tick;
                    t_th_pos = traj.t_th_traj(traj_step) + th_offset;
                    t_al_pos = traj.t_al_traj(traj_step);
                end
            end
            %         t_th_pos = t_th_trajectory(tix,:);
            
            % Ensure obs_th_pos is kept in the range -180 180)
            if(obs_th_pos < -180)
                obs_th_pos = obs_th_pos + 360;
            elseif(obs_th_pos > 180)
                obs_th_pos = obs_th_pos - 360;
            end
            
            if(~make_histograms)
                obs_th_pos_history(tix) = obs_th_pos;
            end
            
            t_th_rel=t_th_pos-repmat(obs_th_pos,size(t_th_pos));
            % Corce t_th_rel into [0 360]
            t_th_rel = mod(180+t_th_rel,360)-180;
            t_th_rel_history(tix)=t_th_rel;
            
            h_pixelpos=round(hdist*tand(t_th_rel)+hpix/2);
            v_pixelpos=round(hdist*tand(t_al_pos)./cosd(t_th_rel)+vpix/2);
            
            ind_logic=false(vpix,hpix);
            
            % Background
            offset=floor(obs_th_pos/360*panpix);
            
            ind=sub2ind(size(src),V(:),round(offset+H(:))); % Select the appropriate pixels to fill the background
            ind(isnan(ind))=1;
            img=reshape(src(ind),vpix,hpix); % img now contains the background
            img_white = ones(size(img)); % Pure white background
            
            % Draw targets
            for targ_ix=1:num_targets % For each target
                if(t_th_rel(targ_ix) > -hfov/2 && t_th_rel(targ_ix) < hfov/2 && t_al_pos(targ_ix) < vfov/2 && t_al_pos(targ_ix) > -vfov/2) % Check whether the target should be drawn
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
            
            
            % Blur and subsample
            frame=zeros(size(h));
            frame_white=frame;
            for pix_ix=1:numel(h)
                ref_h=h(pix_ix);
                ref_v=v(pix_ix);
                frame(pix_ix)=sum(sum(img(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
                    ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));
                frame_white(pix_ix)=sum(sum(img_white(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
                    ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));
            end
            
            input_buffer{1}(:,:,pr_ind) = frame;
            input_buffer{2}(:,:,pr_ind) = frame_white;
            
            % ########################
            % End of frame generation
            % ########################
            
            % For each background variant
            for k=1:2
                
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
            
            % Extract masks for target responses using the white background to identify
            % target
            dark_target_mask = dark_output_buffer{2}(:,:,out_ind) > 0.01;
            light_target_mask = light_output_buffer{2}(:,:,out_ind) > 0.01;
            
            dark_target_mask_buffer(:,:,out_ind)=dark_target_mask;
            light_target_mask_buffer(:,:,out_ind)=light_target_mask;
            
            %     curr_out_dark = dark_output_buffer{2}(:,:,out_ind);
            %     curr_out_light = light_output_buffer{2}(:,:,out_ind);
            %
            %     curr_target_dark = curr_out_dark(dark_target_mask);
            %     curr_target_light = curr_out_light(light_target_mask);
            %
            %     dark_target_responses_buffer(:,:,out_ind) = curr_out_dark{1}(:,:,out_ind);
            %     light_target_responses_buffer(:,:,out_ind) = curr_out_light{1}(:,:,out_ind);
            
            %     % Just show outputs and frames for now
            %
            %         if(showimagery)
            %             if(mod(tix,10) == 0)
            %                 f=figure(1);
            %                 subplot(2,1,1)
            %                 imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
            %                 subplot(2,1,2)
            %                 imagesc(dark_output_buffer{1}(:,:,out_ind));
            %             end
            %         end
            
            if(mod(tix,output_integration)==0) % Time to update the tracker or histograms as the case may be
                for k=1:2
                    % Aggregate outputs over output_integration time steps
                    dark_output_integ{k} = sum(dark_output_buffer{k},3);
                    light_output_integ{k} = sum(light_output_buffer{k},3);
                end
                integrated_tix = floor(tix/output_integration);
                
                
                
                if(make_histograms)
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
                    if(~spinning)
                        
                        if(acquired)
                            % Find the output which agrees best with the current
                            % hypothesis about target states
                            
                            % Calculate predicted velocity using current belief
                            % and our velocity evolution model
                            predict.vel = conv(belief.vel,trackerparams.vel_evolution_kernel,'same');
                            
                            % Find probability that the response agrees with
                            % our position belief given our velocity beliefs
                            prob_R_pos = NaN(vfov,hfov);
                            old_tick=toc;
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
                                [maxpos.v,maxpos.h] = ind2sub([vfov,hfov],I);
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
                            [maxpos.v,maxpos.h] = ind2sub([vfov,hfov],I);
                        end
                        
                        % Change velocity belief based on dynamics
                        belief.vel = conv(belief.vel,trackerparams.vel_evolution_kernel,'same');
                        
                        % If the maximum model output is over threshold and we
                        % haven't turned recently then condition on it
                        if(max_output > trackerparams.output_threshold && (integrated_tix - last_turn) > trackerparams.tix_ignore_after_turn)
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
                        
                        % Generate a kernel based on the likelihood
                        % of our previous position beliefs transitioning to
                        % new ones
                        position_prediction_kernel = NaN(trackerparams.pixel_distance_kernel_size);
                        [Y_mini,X_mini] = ndgrid(1:trackerparams.pixel_distance_kernel_size,1:trackerparams.pixel_distance_kernel_size);
                        centre_x=ceil(trackerparams.pixel_distance_kernel_size/2);
                        centre_y=ceil(trackerparams.pixel_distance_kernel_size/2);
                        d_predict = ((trackerparams.pos_grid_spacing * (Y_mini - centre_y)).^2 + (trackerparams.pos_grid_spacing*(X_mini - centre_x)).^2 ).^0.5;
                        v_predict = d_predict / (output_integration*Ts);
                        
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
                        
                        if(max_output > trackerparams.output_threshold && (integrated_tix - last_turn) > trackerparams.tix_ignore_after_turn)
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
                        
                        if(showimagery)
                            f=figure(1);
                            subplot(4,2,1)
                            imagesc(belief.pos)
                            title('belief position')
                            subplot(4,2,2)
                            plot(trackerparams.vel_pts,belief.vel,'b-',[1 1]*traj.t_vel_traj(tix),[0 max(belief.vel(:))],'r-')
                            title('Velocity belief')
                            subplot(4,2,3)
                            imagesc(dark_output_integ{1})
                            title('output')
                            subplot(4,2,4)
                            imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
                            title('input')
                            subplot(4,2,5)
                            plot(max_certainty_history)
                            if(acquired)
                                title('acquired')
                            else
                                title('not acquired')
                            end
                            subplot(4,2,6)
                            plot(maxbelief_pos_history)
                            subplot(4,2,7)
                            plot(t_th_rel_history)
                            drawnow
                        end
                        
                        last_detection.v = maxpos.v;
                        last_detection.h = maxpos.h;
                        last_detection.tix = integrated_tix;
                        
                        % Pursuit logic
                        [max_y,max_x] = ind2sub(size(belief.pos),I_pos);
                        % Convert max_x into actual screen coordinates
                        max_x_screen = trackerparams.X(1,max_x);
                        
                        if(acquired  && max_certainty > trackerparams.turn_threshold && (integrated_tix - last_turn) > trackerparams.min_turn_gap)
                            if(max_x_screen > 2*hfov/3)
                                % Execute turn such that the target will be a
                                % little to the left of center
                                turn_deg = max_x_screen - hfov/2 + 6;
                                obs_th_pos = obs_th_pos + turn_deg;
                                
                                % Since we will be blacking out a period of time
                                % after the turn, move the belief position
                                % predictively assuming a velocity of 120deg/s
                                predict_angle = trackerparams.tix_ignore_after_turn * output_integration*Ts * 120;
                                predict_deg = turn_deg - floor(predict_angle);
                                
                                left_mask = trackerparams.X <=hfov-(predict_deg-1);
                                right_mask = trackerparams.X >= predict_deg;
                                % Shift belief leftward
                                belief.pos(left_mask) = belief.pos(right_mask);
                                rem = 1-sum(belief.pos(:));
                                belief.pos(~left_mask) = rem / numel(belief.pos(~left_mask));
                                last_turn = integrated_tix;
                            elseif(max_x_screen < hfov/3)
                                turn_deg = hfov/2 - max_x_screen + 6;
                                obs_th_pos = obs_th_pos - turn_deg;
                                
                                predict_angle = trackerparams.tix_ignore_after_turn * output_integration*Ts * 120;
                                predict_deg = turn_deg - floor(predict_angle);
                                
                                % Shift belief rightward
                                left_mask = trackerparams.X <=hfov-(predict_deg-1);
                                right_mask = trackerparams.X >= predict_deg;
                                belief.pos(right_mask) = belief.pos(left_mask);
                                rem = 1-sum(belief.pos(:));
                                belief.pos(~right_mask) = rem / numel(belief.pos(~right_mask));
                                last_turn = integrated_tix;
                            end
                        end
                        
                        max_certainty_history(tix) = max_certainty;
                        maxbelief_pos_history(tix) = maxbelief_pos;
                        max_output_history(tix) = max_output;
                        
                    end
                    
                end
            end
        end
        if(~make_histograms)
            save([outdir 'track_' num2str(output_integration) '-' num2str(traj_range(test_ix)) '.mat'],...
                'traj','trackerparams','t_th_rel_history','max_certainty_history','maxbelief_pos_history','settings',...
                'obs_th_pos_history','max_output_history')
        end
        
        if(make_histograms)
            save([outdir 'integ_' num2str(output_integration) '-' num2str(t_vel_range(test_ix)) '.mat'],'t_vel','hist_R_v','resp_thresholds','output_integration');
        end
    end
end
