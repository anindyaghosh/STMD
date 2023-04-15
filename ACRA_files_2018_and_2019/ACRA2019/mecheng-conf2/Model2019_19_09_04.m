% function Model2019_19_09_04(grp_index,grp_file,groupname)

% pixel2PR, sigma_hw_pixel, cs_kernel_size to be specified in group sheet
clear all
close all

pixel2PR=12;
sigma_hw_degree=1.4;
cs_kernel_size=5;
grp_index=1;
output_integration=10; % How many output frames to aggregate when running tracker
make_histograms=false;
showimagery=false;
traj_number = 1; % Which trajectory the target should follow
impath = 'D:\simfiles\texture\HDR_Botanic_RGB.png';

if(make_histograms)
    %     t_vel_range=20:40:400;
    t_vel_range=400:40:800;
    num_tests=numel(t_vel_range);
else
    traj_range = 1;
    num_tests = numel(traj_range);
    trajfile = load('D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat');
    
    trackerparams.obs_model = load('D:\simfiles\conf2\trackerinfo\obs_model_botanic.mat');
    vel_pts = trackerparams.obs_model.t_vel_range;
    angle_pts = 0:30:330;
    
    rightward_angles = angle_pts < 90 | angle_pts > 270;
    rightward_angles = rightward_angles(:);
    leftward_angles = angle_pts > 90 | angle_pts < 270;
    leftward_angles=leftward_angles(:);
    
    trackerparams.pos_std = 10; % Gaussian around the observed location
    trackerparams.output_threshold = 0.1; % Model outputs <= this are ignored.
    trackerparams.acquisition_threshold = 0.15; % What the max probability has to evaluate out to before we assume a target is acquired
    trackerparams.vel_std = 0.4; % How much the velocity may have changed in the time since the last tracker tick
    
    %     trackerparams.obs_model =
    
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
hfov = 40;
vfov = 20;
fwhm = 1.4; % blur half-width, in degrees
frames=1;
delay=1;
Ts=0.001;
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
        
        est_ticks = spins_todo* (hfov/t_vel / Ts + spin_ticks); % Spin time + test time
        max_ticks = est_ticks * 1.2;
    else
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
        belief.pos = ones(vfov,hfov) / (vfov*hfov);
        belief.vel = ones(numel(vel_pts),1) / numel(vel_pts);
        belief.angle = ones(numel(angle_pts),1) / numel(angle_pts);
        
        max_certainty_history = NaN(max_ticks,1);
        
        [Y,X]=ndgrid(1:vfov,1:hfov); % Used for distance finding
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
        
        t_th_rel=t_th_pos-repmat(obs_th_pos,size(t_th_pos));
        
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
        if(showimagery)
            if(mod(tix,10) == 0)
                f=figure(1);
                imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
                if(tix == 10); movegui(gcf,'west'); end
                f=figure(2);
                imagesc(dark_output_buffer{1}(:,:,out_ind));
                if(tix == 10); movegui(gcf,'east'); end
                title(num2str(tix))
                drawnow
            end
        end
        
        if(mod(tix,output_integration)==0) % Time to update the tracker or histograms as the case may be
            for k=1:2
                % Aggregate outputs over output_integration time steps
                dark_output_integ{k} = sum(dark_output_buffer{k},3);
                light_output_integ{k} = sum(light_output_buffer{k},3);
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
                        [max_output,I]=max(dark_output_integ{1}(:));
                        % Don't update during initial spin-up
                        if(~acquired)
                            % If the maximum model output is over threshold,
                            % condition on it
                            if(max_output > trackerparams.output_threshold)
                                [maxpos.v,maxpos.h] = ind2sub([vfov,hfov],I);
                                
                                % Generate distribution around last
                                % observation based on velocity
                                % distribution
                                if(exist('last_detection','var'))
                                    d_preacq = ((Y-last_detection.v).^2 + (X-last_detection.h).^2).^0.5;
                                    v_preacq = d_preacq / (output_integration*Ts);
                                    prob_v_preacq = NaN(size(v_preacq));
                                    
                                    for p=1:numel(prob_v_preacq)
                                        [~,vel_match] = min(abs(v_preacq(p) - vel_pts));
                                        prob_v_preacq(p) = belief.vel(vel_match);
                                    end
                                    prob_v_preacq = prob_v_preacq / sum(prob_v_preacq(:));
                                    
                                else
                                    % Just assume uniform
                                    prob_v_preacq = ones(vfov,hfov)/(vfov*hfov);
                                end

                                % Generate 2-D gaussian centered on the
                                % current observation
                                d=((Y-maxpos.v).^2 + (X-maxpos.h).^2).^0.5;
                                pdist=gaussmf(d,[trackerparams.pos_std 0]);
                                % Condition position belief via correlation of
                                % current belief with the position gaussian
                                % and velocity
                                belief.pos = belief.pos .* pdist .* prob_v_preacq;
                                % Normalise to sum 1
                                belief.pos = belief.pos ./ sum(belief.pos(:));
                                
%                                 f=figure(1);
% %                                 imagesc(dark_output_integ{1});
%                                 imagesc(prob_v_preacq)
%                                 title('Prob based on v')
%                                 movegui(gcf,'west')
%                                 f=figure(2);
%                                 imagesc(belief.pos)
%                                 movegui(gcf,'center');
%                                 f=figure(3);
%                                 imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
%                                 movegui(gcf,'east')
%                                 drawnow
%                                 waitforbuttonpress
                                
                                % Condition velocity belief based on model
                                % output value
                                
                                % Determine which resp_threshold bin to use
                                resp_bin=find(max_output >= trackerparams.obs_model.resp_thresholds,1,'last');
                                
                                % Get the velocity probabilities for that bin
                                prob_v = trackerparams.obs_model.prob_R_v(resp_bin,:);
                                
                                % Condition the velocity magnitude belief based on the
                                % observation model
                                belief.vel = prob_v' .* belief.vel;
                                
                                % Normalise to be sure
                                belief.vel = belief.vel / sum(belief.vel);
                                
                                % Condition the angle belief on the assumption
                                % that the target is headed to the visual midline
                                
                                if(maxpos.h < hfov/2)
                                    belief.angle = belief.angle .* rightward_angles;
                                else
                                    belief.angle = belief.angle .* leftward_angles;
                                end
                                belief.angle = belief.angle / sum(belief.angle);
                                
                                % Check to see whether we have passed the
                                % acquisition threshold
                                
                                % Determine certainty based on position and
                                % velocity
                                max_certainty = max(belief.pos(:)) * max(belief.vel(:));
                                if(max_certainty > trackerparams.acquisition_threshold)
                                    acquired = true;
                                    % Set a firm belief about the target
                                    % position
                                    [~,I]=max(belief.pos(:));
                                    [belief.firm_ypos, belief.firm_xpos]=ind2sub(size(belief.pos),I);
                                    % Set a firm belief about velocity
                                    % magnitude
                                    [~,I]=max(belief.vel(:));
                                    belief.firm_vel = vel_pts(I);
                                    % Set a firm belief about angle using
                                    % the last detection
                                    belief.firm_angle=atand((maxpos.v - last_detection.v) /(maxpos.h - last_detection.h));
                                    
                                    belief.firm_xvel = belief.firm_vel * cosd(belief.firm_angle);
                                    belief.firm_yvel = belief.firm_vel * sind(belief.firm_angle);
                                    
                                    disp(num2str([belief.firm_xvel;belief.firm_yvel;belief.firm_xpos;belief.firm_ypos]))
                                end
                                
                                max_certainty_history(integrated_tix) = max_certainty;
                                last_detection = maxpos;
                                %                                 f=figure(1);
                                %                                 plot(1:max_ticks,max_certainty_history)
                                %                                 f=figure(2);
                                %                                 imagesc(belief.pos)
                                %                                 f=figure(3);
                                %                                 imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
                                %                                 drawnow
                            end
                        else
                            % We now have a firm belief about where the
                            % target is, its velocity magnitude and angle.
                            
                            % Based on our belief, we now evaluate how
                            % likely the output of each pixel is to
                            % represent a true detection
                            
                            % Start by predicting the new target location based
                            % on our current beliefs
                            
                            hypo.ypos = belief.firm_ypos + belief.firm_vel * sind(belief.firm_angle);
                            hypo.xpos = belief.firm_xpos + belief.firm_vel * cosd(belief.firm_angle);
                            
                            % Create 2-d Gaussian around that hypothetical
                            % position
                            d=((Y-hypo.ypos).^2 + (X-hypo.xpos).^2).^0.5;
                            pdist=gaussmf(d,[trackerparams.pos_std 0]);
                            
                            % Evaluate a vector v_apparent for all pixels
                            d_x_apparent = X-belief.firm_xpos;
                            d_y_apparent = Y-belief.firm_ypos;
                            d_apparent=((d_y_apparent).^2 + (d_x_apparent).^2).^0.5;
                            
                            v_apparent.xvel = d_x_apparent ./ (output_integration * Ts);
                            v_apparent.yvel = d_y_apparent ./ (output_integration * Ts);
                            v_apparent.mag = d_apparent ./ (output_integration*Ts);
                            
                            prob_apparent_given_response=NaN(size(v_apparent.mag)); % TODO: move this outside the loop
                            
                            % Determine how well the response actually
                            % given out by the pixels matches the apparent
                            % velocity
                            for p=1:numel(v_apparent.mag)
                                % Find the closest vel_pt
                                [~,I_vel]= min( abs(v_apparent.mag(p) - vel_pts));
                                % Find the matching response level bin
                                I_resp = find(dark_output_integ{1}(p) > trackerparams.obs_model.resp_thresholds,1,'last');
                                if(isempty(I_resp))
                                    I_resp =1;
                                end
                                
                                % The probability of vel apparent being
                                % correct given the response:
                                prob_apparent_given_response(p) = trackerparams.obs_model.prob_R_v(I_resp,I_vel) / sum(trackerparams.obs_model.prob_R_v(:,I_vel));
                            end
                            % Normalise prob_apparent
                            prob_apparent_given_response = prob_apparent_given_response ./ sum(prob_apparent_given_response(:));
                            
                            % Determine how likely v_apparent is given our
                            % acceleration model and our previous beliefs
                            prob_apparent_given_dynamics = NaN(size(v_apparent.mag));
                            dv_y = v_apparent.yvel - belief.firm_yvel;
                            dv_x = v_apparent.xvel - belief.firm_xvel;
                            
                            for p=1:numel(v_apparent.mag)
                                % Assuming a gaussian distribution around
                                % the current velocity
                                dv = [dv_y(p) dv_x(p)];
                                prob_apparent_given_dynamics(p) = normpdf(norm(dv),0,trackerparams.vel_std);
                            end
                            % Normalise prob_apparent_given_dynamics
                            prob_apparent_given_dynamics = prob_apparent_given_dynamics ./ sum(prob_apparent_given_dynamics(:));
                            
                            % Now evaluate which detection event is the
                            % most likely to match our hypothesis
                            
                            prob_total = prob_apparent_given_dynamics .* prob_apparent_given_response;
%                             prob_total = prob_apparent_given_response;
                            [max_certainty,I_certainty] = max(prob_total(:));
                            [maxpos.v,maxpos.h]=ind2sub(size(prob_total),I_certainty);
                            
                            % Adopt the belief that the target is at that
                            % location, and has the apparent velocity
                            belief.firm_xpos = maxpos.h;
                            belief.firm_ypos = maxpos.v;
                            belief.firm_xvel = v_apparent.xvel(maxpos.v,maxpos.h);
                            belief.firm_yvel = v_apparent.yvel(maxpos.v,maxpos.h);
                            belief.firm_vel = v_apparent.mag(maxpos.v,maxpos.h);
                            belief.firm_angle = atand(belief.firm_yvel / belief.firm_xvel);
                            
                            imtemp=zeros(vfov,hfov);
                            imtemp(belief.firm_ypos,belief.firm_xpos)=1;
                            %                             imshow(prob_total*255,'InitialMagnification',1000)
                            f=figure(1);
                            imagesc(prob_total)
                            title('Total probability')
                            movegui(gcf,'west')
                            f=figure(2);
                            imagesc(prob_apparent_given_response);
                            title('Response')
                            movegui(gcf,'center')
                            f=figure(3);
%                             imshow(input_buffer{1}(:,:,pr_ind),'InitialMagnification',1000);
                            imagesc(prob_apparent_given_dynamics);
                            title('Dynamics')
%                             title('Input')
                            movegui(gcf,'east')
%                             imagesc(prob_total)
                            drawnow
                            waitforbuttonpress
                            
                            
                            % TEMPORARY
                            % Change obs_th_pos to try to track the target
                            
                            
                        end
                    end
                    
                end
            end
        end
    end
    
    if(make_histograms)
        save(['D:\simfiles\conf2\results\testix' num2str(t_vel_range(test_ix)) '.mat'],'t_vel','hist_R_v','resp_thresholds');
    end
end
