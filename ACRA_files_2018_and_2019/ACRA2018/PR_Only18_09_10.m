function PR_Only18_09_10(grp_index,grp_file,groupname)

% pixel2PR, sigma_hw_pixel, cs_kernel_size to be specified in group sheet

% ############
%   SETUP
% ############
% grp_index=3121;
% grp_file='F:\MrBlack\CONFdronefiles\CONF_Droneparams18_08_09.txt';
% groupname='PR_Only';
% sigma_hw_pixel=18;
% pixel2PR=18;
% cs_kernel_size=101;

% vid_end='Pursuit1.mj2';

part_size=600;
part_padding=150;

locroot='/fast/users/a1119946/simfiles/dronefiles/';
vid_dir='/fast/users/a1119946/dronevids/';
% gt_dir='/fast/users/a1119946/simfiles/dronefiles/gtruth/';
% 
% vid_dir='F:\Drone footage\MJ2 Format\';
% locroot='F:\MrBlack\CONFdronefiles\';

dataDir=[locroot groupname '/data/'];
if(~exist(dataDir,'dir'))
    mkdir(dataDir);
end

savename=[dataDir 'gind' num2str(grp_index) '.mat'];
if(exist(savename,'file'))
    disp(['Found existing ' savename '. Exiting.'])
    return
end

if(strcmp(grp_file,'/fast/users/a1119946/simfiles/dronefiles/CONF_Droneparams18_08_09.txt'))
    file=fopen(grp_file,'r');
    grp_info=textscan(file,'%f%s%f%f%f%f','Headerlines',1,'Delimiter','\t');
    fclose(file);
elseif(strcmp(grp_file,'/fast/users/a1119946/simfiles/dronefiles/CONF_DroneHighRes18_08_30.txt'))
    file=fopen(grp_file,'r');
    grp_info=textscan(file,'%f%s%f%f%f%f%f','Headerlines',1,'Delimiter','\t');
    fclose(file);
else
    disp('Unknown group file')
    return
end

input_video_end=char(grp_info{2}(grp_index));
partnum=grp_info{3}(grp_index);
numparts=grp_info{4}(grp_index);
pixel2PR=grp_info{5}(grp_index);
sigma_hw_pixel=grp_info{6}(grp_index);

% gtfile=[gt_dir input_video_end(1:end-4) '_gt.mat'];

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
% 
% pad_size=floor(cs_kernel_size/2);

% Photoreceptor filter coefficients
pr_len=single(9);
pr_num=zeros(1,1,pr_len,'single');
pr_num(:)=[0    0.0001   -0.0011    0.0052   -0.0170    0.0439   -0.0574    0.1789   -0.1524];
pr_den=zeros(1,1,pr_len,'single');
pr_den(:)=[ 1.0000   -4.3331    8.6847  -10.7116    9.0004   -5.3058    2.1448   -0.5418    0.0651];

pr_num_array=repelem(pr_num,vfov,hfov,1);
pr_den_array=repelem(pr_den,vfov,hfov,1);

% Setup buffers;
pr_buffer=zeros(vfov,hfov,9,'single');
input_buffer=pr_buffer;

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

simtime=frame_times(end_frame)-frame_times(start_frame);
simtime=ceil(simtime/Ts)*Ts;

start_time=frame_times(start_frame);

pr_summed=zeros(frames_todo,1);
% pr_hist=zeros(vfov,hfov,frames_todo);

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

    if(flag_for_record)
        flag_for_record=false;
        pr_summed(vid_framenum)=sum(abs(pr_buffer(:)));
        pr_hist(:,:,vid_framenum)=sum(abs(pr_buffer),3);
%         imagesc(reshape(sum(abs(pr_buffer),3),vfov,hfov))
%         colorbar
%         drawnow
    end
end
pr_summed=pr_summed(151:end-1);
save(savename,'pr_summed','grp_index','grp_file','pixel2PR',...
'sigma_hw_pixel',...
'partnum',...
'numparts',...
'vfov',...
'hfov')
disp(['Full sim took ' num2str(toc) ' seconds for ' num2str(tix) ' ticks'])

% 
% %%
% bb=max(max(pr_hist,[],1),[],2);
% bb=bb(:);
% bb=bb(151:end-1);
% plot(150*frame_period+frame_period*(1:numel(bb)),bb)
% 
% %%
% a=load('F:\MrBlack\CONFdronefiles\PR_Only\data\gind3121.mat');
% b=load('F:\MrBlack\CONFdronefiles\PR_Only\data\gind3122.mat');
% prs=[a.pr_summed;b.pr_summed];
