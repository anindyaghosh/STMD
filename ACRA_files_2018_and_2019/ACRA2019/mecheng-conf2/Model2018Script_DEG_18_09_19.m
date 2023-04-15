% This script runs the model on the input video specified in the grp_file
% Changes from 18_09_16: the horizontal FOV value has been changed (from
% 84)

function Model2018Script_DEG_18_09_19(grp_index,grp_file,groupname)

% pixel2PR, sigma_hw_pixel, cs_kernel_size to be specified in group sheet

%############
%   SETUP
%############
% clear all
% close all
% grp_index=96789;
% grp_file='C:\Users\John\Desktop\PhD\paramfiles\CONF_DroneHighRes18_08_30.txt';
% groupname='TestHighRes';
% sigma_hw_pixel=0;
% pixel2PR=1;
% cs_kernel_size=101;

% vid_end='Pursuit1.mj2';

        
camera_fov=2*atand(tand(84/2)*3/sqrt(13)); % Converting 84° diagonal FOV to horizontal
% Note that for 3:2 sensor and 16:9 aspect ratio the full potential
% horizontal FOV is preserved
% Vertical FOV not required but could be calculated with:
% camera_vert_fov=2*atand(9/16*tand(camera_fov/2))

fdsr_lp=0.170;

part_size=600;
part_padding=150;

bg_edges=[0 logspace(-12,0,100)];

% locroot='E:\PHD\CONFdronefiles\';
% vid_dir='E:\PHD\CONFdronefiles\MJ2videos\';
% gt_dir='E:\PHD\CONFdronefiles\gtruth\';

locroot='/fast/users/a1119946/simfiles/dronefiles/';
vid_dir='/fast/users/a1119946/dronevids/';
gt_dir='/fast/users/a1119946/simfiles/dronefiles/gtruth/';

% locroot='C:\Users\John\Desktop\PhD\ConfDronefiles\';
% vid_dir='F:\Drone footage\MJ2 Format\';
% gt_dir='F:\MrBlack\CONFdronefiles\gtruth\';

dataDir=[locroot groupname '/data/'];
if(~exist(dataDir,'dir'))
    mkdir(dataDir);
end

savename=[dataDir 'gind' num2str(grp_index) '.mat'];
if(exist(savename,'file'))
    disp(['Found existing ' savename '. Exiting.'])
    return
end

file=fopen(grp_file,'r');
grp_info=textscan(file,'%f%s%f%f%f%f%f','Headerlines',1,'Delimiter','\t');
fclose(file);

input_video_end=char(grp_info{2}(grp_index));
partnum=grp_info{3}(grp_index);
numparts=grp_info{4}(grp_index);
pixel2PR=grp_info{5}(grp_index);
sigma_hw_pixel=grp_info{6}(grp_index);
cs_kernel_size=grp_info{7}(grp_index);

gtfile=[gt_dir input_video_end(1:end-4) '_gt.mat'];

Ts=0.001;

% Sizing
vidfile=[vid_dir input_video_end];
vread=VideoReader(vidfile);
% input_im=rand(1080,1920,100,'single');
vraw=get(vread,'Height');
hraw=get(vread,'Width');

sigma_pixel = sigma_hw_pixel / 2.35;
kernel_size = 2*(ceil(sigma_pixel));
if(~mod(kernel_size,2))
    kernel_size=kernel_size+1;
end
blurbound=single(floor(kernel_size/2));

newv=vraw-2*blurbound;
newh=hraw-2*blurbound;

hfov=floor(newh/pixel2PR);
vfov=floor(newv/pixel2PR);
newsize=[2*blurbound+(vfov-1)*pixel2PR+1,2*blurbound+(hfov-1)*pixel2PR+1];

% Success threshold for determining what is a target and what isn't
target_window = hfov/camera_fov; % How many subsampled pixels per degree
success_threshold = target_window;

pad_size=floor(cs_kernel_size/2);

% Photoreceptor filter coefficients
pr_len=single(9);
pr_num=zeros(1,1,pr_len,'single');
pr_num(:)=[0    0.0001   -0.0011    0.0052   -0.0170    0.0439   -0.0574    0.1789   -0.1524];
pr_den=zeros(1,1,pr_len,'single');
pr_den(:)=[ 1.0000   -4.3331    8.6847  -10.7116    9.0004   -5.3058    2.1448   -0.5418    0.0651];

pr_num_array=repelem(pr_num,vfov,hfov,1);
pr_den_array=repelem(pr_den,vfov,hfov,1);

% LMC kernel
lmc_kernel = single(1/9*[-1 -1 -1; -1 8 -1; -1 -1 -1]);

% NLAM
tau_on_up=0.005;
tau_on_down=0.35;
tau_off_up=0.005;
tau_off_down=0.35;

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
% delay_lp=0.025;
lp_den_raw=[(1+2*fdsr_lp/Ts), (1-2*fdsr_lp/Ts)];
lp_num=single([1/lp_den_raw(1) 1/lp_den_raw(1)]);

lp_den=single(lp_den_raw(2)/lp_den_raw(1));

% Setup buffers;
pr_buffer=zeros(vfov,hfov,9,'single');
input_buffer=pr_buffer;

hp_inbuff=zeros(vfov,hfov,2,'single');
hp_outbuff=hp_inbuff;
delay_on_inbuff=hp_inbuff;
delay_off_inbuff=hp_inbuff;
delay_on_outbuff=hp_inbuff;
delay_off_outbuff=hp_inbuff;

fdsr_on_inbuff=hp_inbuff;
fdsr_off_inbuff=hp_inbuff;

fdsr_on_outbuff=hp_inbuff;
fdsr_off_outbuff=hp_inbuff;

alpha_on=zeros(vfov,hfov,'single');
alpha_off=alpha_on;

alpha_on_mask=false(vfov,hfov);
alpha_off_mask=alpha_on_mask;

pr_ind=0;

framerate=get(vread,'Framerate');
frame_period=1/framerate;
duration=get(vread,'Duration');
frames_in_vid=1+framerate*duration;

frame_times=(0:frames_in_vid-1)*frame_period;
frame_modelstep=ceil(frame_times/Ts);

frame_buffer=zeros(vfov,hfov,2,'single');

vid_ticks=floor(duration*1000);

start_frame=(partnum-1)*part_size+1;
end_frame=min(1+partnum*part_size+part_padding,frames_in_vid-1);
% end_frame=100;
% start_frame=17343;
% end_frame=17352;

frames_todo=numel(start_frame:end_frame);
set(vread,'CurrentTime',(start_frame-1)/framerate);

framebuff_ind=1;
frame_num=zeros(1,2,'single');
frame_num(framebuff_ind)=start_frame;
frame_num(3-framebuff_ind)=start_frame+1;

% The other script blurred in uint8. Switching over to blurring in single
% precision.
newframe=single(readFrame(vread)/255);
% newframe=readFrame(vread);
if(sigma_pixel > 0)
    img=imgaussfilt(newframe,sigma_pixel,'FilterSize',kernel_size);
    frame_buffer(:,:,framebuff_ind)=img( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
else
    frame_buffer(:,:,framebuff_ind)=newframe( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
end
% frame_buffer(:,:,framebuff_ind)=frame_buffer(:,:,framebuff_ind)/255;

newframe=single(readFrame(vread)/255);
% newframe=readFrame(vread);
if(sigma_pixel > 0)
    img=imgaussfilt(newframe,sigma_pixel,'FilterSize',kernel_size);
    frame_buffer(:,:,3-framebuff_ind)=img( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
else
    frame_buffer(:,:,3-framebuff_ind)=newframe( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
end
% frame_buffer(:,:,3-framebuff_ind)=frame_buffer(:,:,3-framebuff_ind)/255;

vid_framenum=2;

interp_period=frame_times(frame_num(3-framebuff_ind)) - frame_times(frame_num(framebuff_ind));
interp_start_time=frame_times(frame_num(framebuff_ind));
interp_diff=(frame_buffer(:,:,3-framebuff_ind) - frame_buffer(:,:,framebuff_ind));

model_steps_per_frame=ceil(1000/framerate);

dark_output_buffer=zeros(vfov,hfov,model_steps_per_frame,'single');
light_output_buffer=dark_output_buffer;

maxlocs_DARK.row=zeros(frames_todo,10,'single');
maxlocs_DARK.column=maxlocs_DARK.row;
maxlocs_DARK.value=maxlocs_DARK.row;

maxlocs_LIGHT.row=maxlocs_DARK.row;
maxlocs_LIGHT.column=maxlocs_DARK.row;
maxlocs_LIGHT.value=maxlocs_DARK.row;

maxlocs_COMBO.row=maxlocs_DARK.row;
maxlocs_COMBO.column=maxlocs_DARK.row;
maxlocs_COMBO.value=maxlocs_DARK.row;

targresp.LIGHT=NaN(numel(start_frame:end_frame),1,'single');
targresp.DARK=targresp.LIGHT;
targresp.COMBO=targresp.LIGHT;

targrank.LIGHT=targresp.LIGHT;
targrank.DARK=targresp.LIGHT;
targrank.COMBO=targresp.LIGHT;

bg_counts.LIGHT=zeros(numel(bg_edges)-1,1,'single');
bg_counts.DARK=bg_counts.LIGHT;
bg_counts.COMBO=bg_counts.LIGHT;

simtime=frame_times(end_frame)-frame_times(start_frame);
simtime=ceil(simtime/Ts)*Ts;

ssgt=ConvertGroundTruth18_08_28(gtfile,pixel2PR,blurbound,hfov,vfov,start_frame,end_frame);

start_time=frame_times(start_frame);

flag_for_record=false;
tic
tix_todo=ceil(simtime/Ts);
for tix=1:tix_todo %model ticks
    if(mod(tix,200) == 0)
        disp([num2str(tix) '/' num2str(tix_todo) ' ' num2str(toc)])
    end
    curr_time=start_time+(tix-1)*Ts;
    % Update indices
    pr_ind=1+mod(tix-1,pr_len);
    hp_ind=1+mod(tix-1,2);
    out_ind=1+mod(tix-1,model_steps_per_frame);
    
    % Test whether it's time for a new frame
    if(curr_time >= frame_times(frame_num(3-framebuff_ind)))
        % Time for a new frame
%         newframe=readFrame(vread);
        newframe=single(readFrame(vread))/255;
        if(sigma_pixel > 0)
            img=imgaussfilt(newframe,sigma_pixel,'FilterSize',kernel_size);
            frame_buffer(:,:,framebuff_ind)=img( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
        else
            frame_buffer(:,:,framebuff_ind)=newframe( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
        end
%         frame_buffer(:,:,framebuff_ind)=frame_buffer(:,:,framebuff_ind)/255;

        vid_framenum=vid_framenum+1;
        frame_num(framebuff_ind)=frame_num(framebuff_ind)+2;
        
        framebuff_ind=framebuff_ind+1;
        framebuff_ind=1+mod(framebuff_ind-1,2);
        % Recalc interpolation parameters
        interp_period=frame_times(frame_num(3-framebuff_ind)) - frame_times(frame_num(framebuff_ind));
        interp_start_time=frame_times(frame_num(framebuff_ind));
        interp_diff=frame_buffer(:,:,3-framebuff_ind)-frame_buffer(:,:,framebuff_ind);
        
        flag_for_record=true;
    end
    
    if(curr_time == frame_times(frame_num(framebuff_ind)))
        % If the time matches the current frame, just use it
        input_buffer(:,:,pr_ind)=frame_buffer(:,:,framebuff_ind);
    else
        % Otherwise interpolate
        timediff=single((curr_time-interp_start_time)/interp_period);
        input_buffer(:,:,pr_ind)=min(max(frame_buffer(:,:,framebuff_ind)+interp_diff*timediff,0),1);
    end

    % PHOTORECEPTOR FILTER
    den_ix=[pr_ind-1:-1:1 pr_len:-1:pr_ind+1];
    num_ix=[pr_ind:-1:1 pr_len:-1:pr_ind+1];

    pr_buffer(:,:,pr_ind)= -sum(pr_buffer(:,:,den_ix).*pr_den_array(:,:,2:pr_len),3)+...
        sum(input_buffer(:,:,num_ix).*pr_num_array(:,:,:),3);

    % LMC and highpass
    hp_inbuff(:,:,hp_ind)=conv2(padarray(pr_buffer(:,:,pr_ind),[1,1],'symmetric'),lmc_kernel,'valid'); % Using same to match the simulink runs
    hp_outbuff(:,:,hp_ind)=hp_outbuff(:,:,3-hp_ind).*(5/6)+hp_inbuff(:,:,hp_ind)-hp_inbuff(:,:,3-hp_ind);

    % FDSR
    fdsr_on_inbuff(:,:,hp_ind)=max(hp_outbuff(:,:,hp_ind),0);
    fdsr_off_inbuff(:,:,hp_ind)=max(-hp_outbuff(:,:,hp_ind),0);
    
    alpha_on_mask=fdsr_on_inbuff(:,:,hp_ind) >= fdsr_on_outbuff(:,:,3-hp_ind);
    alpha_off_mask=fdsr_off_inbuff(:,:,hp_ind) >= fdsr_off_outbuff(:,:,3-hp_ind);
    
    alpha_on= alpha_on_up.*alpha_on_mask + alpha_on_down.*~alpha_on_mask;
    alpha_off= alpha_off_up.*alpha_off_mask + alpha_off_down.*~alpha_off_mask;
    
    fdsr_on_outbuff(:,:,hp_ind)=fdsr_on_inbuff(:,:,hp_ind).*alpha_on + fdsr_on_outbuff(:,:,3-hp_ind).*(1-alpha_on);
    fdsr_off_outbuff(:,:,hp_ind)=fdsr_off_inbuff(:,:,hp_ind).*alpha_off + fdsr_off_outbuff(:,:,3-hp_ind).*(1-alpha_off);

    % Subtraction of FDSR from input and C/S
    on_chan=max(conv2(padarray(max(fdsr_on_inbuff(:,:,hp_ind)-fdsr_on_outbuff(:,:,3-hp_ind),0),[pad_size pad_size],'symmetric'),cs_kernel,'valid'),0);
    off_chan=max(conv2(padarray(max(fdsr_off_inbuff(:,:,hp_ind)-fdsr_off_outbuff(:,:,3-hp_ind),0),[pad_size pad_size],'symmetric'),cs_kernel,'valid'),0);

    % Delay filter
    delay_on_inbuff(:,:,hp_ind) = on_chan;
    delay_off_inbuff(:,:,hp_ind) = off_chan;
    
    delay_on_outbuff(:,:,hp_ind) = lp_num(1)* delay_on_inbuff(:,:,hp_ind) +...
        lp_num(2)*delay_on_inbuff(:,:,3-hp_ind) - ...
        lp_den*delay_on_outbuff(:,:,3-hp_ind);
    
    delay_off_outbuff(:,:,hp_ind) = lp_num(1)* delay_off_inbuff(:,:,hp_ind) +...
        lp_num(2)*delay_off_inbuff(:,:,3-hp_ind) - ...
        lp_den*delay_off_outbuff(:,:,3-hp_ind);
    
    dark_output_buffer(:,:,out_ind) = on_chan .*delay_off_outbuff(:,:,hp_ind);
    light_output_buffer(:,:,out_ind) = off_chan .* delay_on_outbuff(:,:,hp_ind);
    
    if(flag_for_record)
        flag_for_record=false;
        % Find max etc
        bdark=sum(dark_output_buffer,3);
        blight=sum(light_output_buffer,3);
        bcombo=(bdark+blight)/2;
        
        [val_DARK,I_DARK]=sort(bdark(:),'descend');
        [val_LIGHT,I_LIGHT]=sort(blight(:),'descend');
        [val_COMBO,I_COMBO]=sort(bcombo(:),'descend');

        [ir,ic]=ind2sub(size(bdark),I_DARK(1:10));
        maxlocs_DARK.row(vid_framenum-1,:)=ir;
        maxlocs_DARK.column(vid_framenum-1,:)=ic;
        maxlocs_DARK.value(vid_framenum-1,:)=val_DARK(1:10);
        
        [ir,ic]=ind2sub(size(blight),I_LIGHT(1:10));
        maxlocs_LIGHT.row(vid_framenum-1,:)=ir;
        maxlocs_LIGHT.column(vid_framenum-1,:)=ic;
        maxlocs_LIGHT.value(vid_framenum-1,:)=val_LIGHT(1:10);
        
        [ir,ic]=ind2sub(size(bcombo),I_COMBO(1:10));
        maxlocs_COMBO.row(vid_framenum-1,:)=ir;
        maxlocs_COMBO.column(vid_framenum-1,:)=ic;
        maxlocs_COMBO.value(vid_framenum-1,:)=val_COMBO(1:10);
        
        % Record target response
        bg_mask=true(vfov,hfov);
        abs_frame_num= start_frame + vid_framenum -1;
        if(ssgt.gtbuttonfill(abs_frame_num) == 1)
            % Within a square with edges 2*success_threshold centred on the
            % ground truth, take max of all pixels within success_threshold
            % distance from ground truth.
            for k_x=floor(-success_threshold):ceil(success_threshold)
                for k_y=floor(-success_threshold):ceil(success_threshold)
                    curr_x=round(ssgt.gtx_ss(abs_frame_num)+k_x);
                    curr_y=round(ssgt.gty_ss(abs_frame_num)+k_y);
                    dist=(k_x^2+k_y^2)^0.5;
                    if(curr_x >= 1 && curr_x <= hfov &&...
                            curr_y >= 1 && curr_y <= vfov &&...
                            dist <= success_threshold)
                        % The point is within the bounds of the screen and
                        % is within success_threshold of the ground truth
                        bg_mask(curr_y,curr_x)=false; % exclude from bg
                        targresp.DARK(vid_framenum-1)=max(targresp.DARK(vid_framenum-1),bdark(curr_y,curr_x));
                        targresp.LIGHT(vid_framenum-1)=max(targresp.LIGHT(vid_framenum-1),blight(curr_y,curr_x));
                        targresp.COMBO(vid_framenum-1)=max(targresp.COMBO(vid_framenum-1),bcombo(curr_y,curr_x));
                    end 
                end
            end

            % Find how far down the ranks the target response is
            bdark_sort=sort(bdark(:),'descend');
            blight_sort=sort(blight(:),'descend');
            bcombo_sort=sort(bcombo(:),'descend');

            targrank.DARK(vid_framenum-1) = find(bdark_sort == targresp.DARK(vid_framenum-1),1,'first');
            targrank.LIGHT(vid_framenum-1) = find(blight_sort == targresp.LIGHT(vid_framenum-1),1,'first');
            targrank.COMBO(vid_framenum-1) = find(bcombo_sort == targresp.COMBO(vid_framenum-1),1,'first');    
        end
        
        % Record binned background responses
        bgresp.LIGHT=blight(bg_mask);
        bgresp.DARK=bdark(bg_mask);
        bgresp.COMBO=bcombo(bg_mask);

        bg_counts.LIGHT=bg_counts.LIGHT+histcounts(bgresp.LIGHT,bg_edges)';
        bg_counts.DARK=bg_counts.DARK+histcounts(bgresp.DARK,bg_edges)';
        bg_counts.COMBO=bg_counts.COMBO+histcounts(bgresp.COMBO,bg_edges)';
        
    end
end

% Chop off the padding at the start
maxlocs_DARK.row=maxlocs_DARK.row(part_padding+1:end-1,:);
maxlocs_DARK.column=maxlocs_DARK.column(part_padding+1:end-1,:);
maxlocs_DARK.value=maxlocs_DARK.value(part_padding+1:end-1,:);

maxlocs_LIGHT.row=maxlocs_LIGHT.row(part_padding+1:end-1,:);
maxlocs_LIGHT.column=maxlocs_LIGHT.column(part_padding+1:end-1,:);
maxlocs_LIGHT.value=maxlocs_LIGHT.value(part_padding+1:end-1,:);

maxlocs_COMBO.row=maxlocs_COMBO.row(part_padding+1:end-1,:);
maxlocs_COMBO.column=maxlocs_COMBO.column(part_padding+1:end-1,:);
maxlocs_COMBO.value=maxlocs_COMBO.value(part_padding+1:end-1,:);

targresp.LIGHT=targresp.LIGHT(part_padding+1:end-1);
targresp.DARK=targresp.DARK(part_padding+1:end-1);
targresp.COMBO=targresp.COMBO(part_padding+1:end-1);

targrank.LIGHT=targrank.LIGHT(part_padding+1:end-1);
targrank.DARK=targrank.DARK(part_padding+1:end-1);
targrank.COMBO=targrank.COMBO(part_padding+1:end-1);

save([dataDir 'gind' num2str(grp_index) '.mat'],...
    'maxlocs_DARK','maxlocs_LIGHT','maxlocs_COMBO',...
    'pixel2PR','sigma_hw_pixel','cs_kernel_size',...
    'fdsr_lp','targresp','bg_counts','bg_edges','targrank')

disp(['Full sim took ' num2str(toc) ' seconds for ' num2str(tix) ' ticks'])