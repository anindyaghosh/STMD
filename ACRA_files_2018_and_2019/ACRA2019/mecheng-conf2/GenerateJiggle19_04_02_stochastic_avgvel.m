function im_ss=GenerateJiggle19_04_02_stochastic_avgvel(hfov,vfov,delay,frames,Ts,impath,obs_vel,t_vel,...
    t_width,t_height,t_value,drawmode,fwhm,pixel_per_degree,degree_per_PR,targ_dir,obs_th_start,step_size,desired_bg_avg_vel,rix,rixdir)

% clear all
% close all
% 
% hfov=20;
% vfov=40;
% delay=1000;
% frames=2000;
% Ts=0.001;
% 
% rix=1;
% rixdir='D:\simfiles\LongStochasticMovement\';
% 
% impath = 'D:\simfiles\texture\HDR_Botanic_RGB.png';
% obs_vel=0;
% t_vel=-110;
% t_width=1;
% t_height=1;
% t_value=0;
% drawmode='pixels';
% fwhm=1.4;
% pixel_per_degree=12;
% degree_per_PR=1;
% targ_dir='vertical';
% obs_th_start=0;
% desired_bg_avg_vel=100;
% step_size=50;
% 
% % Generate obs_path
% obs_path=obs_th_start+([0:frames-1]*obs_vel*Ts)';
% obs_path = obs_path+180;
% obs_path = mod(obs_path,360);
% obs_path = obs_path - 180;

% Load obs_path
a=load([rixdir 'move' num2str(rix) '.mat']);

% Sample obs_path at specified step_size
sampled_sig = a.orig_sig(1:step_size:end);
sampled_sig = sampled_sig(1:frames);

% Shift to zero mean
sampled_sig = sampled_sig - mean(sampled_sig);

% Estimate the mean velocity using f'(x) = (f(x+h) - f(x-h)) / 2h which has
% error O(h^2)

est_vel=(sampled_sig(3:end)-sampled_sig(1:end-2))/(2*Ts);
ampfactor=desired_bg_avg_vel/mean(abs(est_vel));
sampled_sig = sampled_sig*ampfactor;
obs_path = obs_th_start + sampled_sig + (0:numel(sampled_sig)-1)*Ts*obs_vel;
obs_path=obs_path(:);

% By this point, should have a signal with the desired average background
% velocity which has been formed from a signal with the requested step size

if(strcmp(targ_dir,'vertical'))
    t_al_rel=zeros(frames,1);


    t_al_rel(delay:frames)=(vfov/2-1)*sign(-t_vel)+[0:(frames-delay)]*t_vel*Ts;

    % Force t_al_rel into [-vfov/2,vfov/2]
    lapcount=floor(abs(t_al_rel+vfov/2*sign(t_vel))/vfov);
    if(t_vel > 0)
        t_al_rel = t_al_rel+vfov/2;
        t_al_rel = mod(t_al_rel,vfov);
        t_al_rel = t_al_rel - vfov/2;
        t_al_path=t_al_rel;
    else
        t_al_rel=t_al_rel+sign(-t_vel)*vfov/2;
        t_al_rel=mod(t_al_rel,vfov);
        t_al_rel=t_al_rel+sign(t_vel)*vfov/2;
        t_al_path=t_al_rel;
    end
    
    
    t_th_rel=-hfov/4 + hfov/4*mod(lapcount,3);
    t_th_path=obs_path+t_th_rel;
    
elseif(strcmp(targ_dir,'horizontal'))
    disp('Horizontal not fully written')
%     t_al_path=zeros(frames,1);
% 
%     t_th_rel=zeros(frames,1);
%     t_th_rel(delay-1)=obs_path(delay-1)+hfov/2;
%     t_th_rel(delay:frames)=-hfov/2+[0:(frames-delay)]*t_vel*Ts;
% 
%     % Force t_th_rel into [-hfov/2,hfov/2]
%     t_th_rel=t_th_rel+hfov/2;
%     t_th_rel=mod(t_th_rel,hfov);
%     t_th_rel=t_th_rel-hfov/2;
% 
%     t_th_path=obs_path+t_th_rel;
else
    disp('No valid target direction. Options are "horizontal" or "vertical". Exiting.')
end

% Add jiggle to observer movement

% jiggle_amp=0.35; % degrees
% jiggle_freq=5; % Hz
% deg_per_cycle=360;
% cycles_per_ms=jiggle_freq/1000;
% obs_path=obs_path + jiggle_amp*sind((0:frames-1)'*cycles_per_ms*deg_per_cycle);

% impath='D:\Johnstuff\Matlab\Data\texture\HDR_Botanic_RGB.png';
% t_width=1;
% t_height=1;
% t_value=0;

% drawmode='pixels';
% delay=1000;
% sigma_hw=1.4;
% pixel_per_degree=12;
% degree_per_PR_coarse=1;
% degree_per_PR_fine=0.25;

hfov_coarse=hfov;
vfov_coarse=vfov;
hfov_fine=1;
vfov_fine=1;
fwhm_coarse=fwhm;
fwhm_fine=1;
degree_per_PR_coarse=degree_per_PR;
degree_per_PR_fine=1;

t_al_pos_input=t_al_path;
t_th_pos_input=t_th_path;
obs_th_input=obs_path;
% tic
[  im_ss_coarse,im_ss_fine,...
            h_pixelpos_coarse,v_pixelpos_coarse,...
            h_pixelpos_fine,v_pixelpos_fine]=RenderRepeatPath18_06_12(...
            impath,...
            t_width,t_height,t_value,...
            hfov_coarse,vfov_coarse,fwhm_coarse,...
            hfov_fine,vfov_fine,fwhm_fine,...
            frames,delay,Ts,...
            drawmode,...
            obs_th_input,t_al_pos_input,t_th_pos_input,...
            pixel_per_degree,...
            degree_per_PR_coarse,degree_per_PR_fine);
        
im_ss=im_ss_coarse;
% end
% toc
%%
% [vc,hc,~]=size(im_ss_coarse);
% [vf,hf,~]=size(im_ss_fine);   
% figure(1)
% for f=delay:frames
% subplot(2,1,1)
% imshow(reshape(im_ss_coarse(:,:,f),[vc hc]),'InitialMagnification',1000)
% title(f)
% drawnow
% end

