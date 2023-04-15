% This function runs part of the video through a model which has both a
% light selective and dark selective version in it.
% 18_06_27: In this version the cs_kernel is now "normalised" such that the
% sum of the negative parts is always -16, to match the sum of negative
% parts in a 5x5 kernel.
%
% 18_09_17:     Previously only the target response at the actual ground
% truth location was being recorded. Now, the maximum response within the
% 3x3 region centred on ground truth is recorded.
function CONF_ProcessDroneVideoTARGRESP18_09_17(grp_index,grp_file,groupname,cs_kernel_size_range,fdsr_lp_range)


% grp_index=15637;
% % grp_file='D:\Johnstuff\Matlab\Data\paramfiles\DroneVidParams18_06_22.txt';
% grp_file='D:\Johnstuff\Matlab\Data\CONFdronefiles\CONF_Droneparams18_08_09.txt';
% groupname='DroneVidProcessed';
% cs_kernel_size_range=3:2:13; % Keep it odd
% fdsr_lp_range=[0.005 0.0250    0.0450    0.0650    0.0850];


% End of feed in
part_size=600; % How many frames per part
part_padding=150; % How many frames at the start for warm-up

bg_edges=[0 logspace(-12,0,100)];

% locroot='D:\Johnstuff\Matlab\Data\';
% vid_dir='D:\Johnstuff\Matlab\Data\GIC_Actual\Pursuit1\fragmented_video\';
% gt_dir='D:\Johnstuff\Matlab\Data\GIC_Actual\gtruth\';
locroot='/fast/users/a1119946/simfiles/dronefiles/';
vid_dir='/fast/users/a1119946/dronevids/';
gt_dir='/fast/users/a1119946/simfiles/dronefiles/gtruth/';
% locroot='C:\Users\John\Desktop\PhD\';
% vid_dir='F:\Drone footage\MJ2 Format\';
% gt_dir='F:\Drone footage\MJ2 Format\';

[cs_kernel_size_s,fdsr_lp_s]=meshgrid(cs_kernel_size_range,fdsr_lp_range);

% Index, name, part, numparts, pixel2PR, sigma_hw
file=fopen(grp_file,'r');
grp_info=textscan(file,'%f%s%f%f%f%f','Headerlines',1,'Delimiter','\t');
fclose(file);

input_video_end=char(grp_info{2}(grp_index));
partnum=grp_info{3}(grp_index);
numparts=grp_info{4}(grp_index);
pixel2PR=grp_info{5}(grp_index);
sigma_hw_pixel=grp_info{6}(grp_index);


gtfile=[gt_dir input_video_end(1:end-4) '_gt.mat'];

dataDir_base=[locroot groupname '/'];
if(~exist(dataDir_base,'dir'))
	mkdir(dataDir_base);
end
dataDir=[dataDir_base 'data/'];

if(~exist(dataDir,'dir'))
	mkdir(dataDir);
end

savename=[dataDir 'gind' num2str(grp_index) '.mat'];

if(exist(savename,'file'))
    disp(['Skipping ' savename]);
    return;
end

input_video=[vid_dir input_video_end];

thrownwarning=false;
Ts= 0.001;
func_name=mfilename;

trashdir=[dataDir_base 'trash/' num2str(grp_index) '/'];
dest=[dataDir_base 'testbed/' num2str(grp_index) '/'];

% Model setup
model_names={'Model_OnlyDark_18_06_22_forvideo_lightanddark'};
num_models=numel(model_names);

models=cell(1,num_models);
for t=1:num_models
    models{t}=[locroot char(model_names{t}) '.slx'];
    load_system(char(models{t}))
    set_param([char(model_names{t}) '/From Workspace'], 'SampleTime',num2str(Ts));
end
currDir=trashdir;
tmpDir=[trashdir 'temp' num2str(1)];
if(~exist(tmpDir,'dir'))
    mkdir(tmpDir);
end
cd(tmpDir);

if(~exist(dataDir,'dir'))
    mkdir(dataDir)
end

if(~exist(dest,'dir'))
    mkdir(dest)
end
cd(dest)

% model parameters
tcs=[0.005 0.350 0.005 0.350];

% Load video information
vread=VideoReader(input_video);
hraw=get(vread,'Width');
vraw=get(vread,'Height');
vid_duration=get(vread,'Duration');
framerate=get(vread,'FrameRate');
vid_frames=vid_duration*framerate;
if(floor(vid_frames) ~= vid_frames)
    disp('Warning: rounding vid_frames down')
    vid_frames=floor(vid_frames);
end
frame_period=1/framerate;
frameratio=1000/framerate;    % How many model frames per actual input frame

%Parameters for blurring and subsampling
% sigma_deg = sigma_hw/2.35;
% sigma_pixel = sigma_deg*pixel2PR;
sigma_pixel = sigma_hw_pixel / 2.35;
kernel_size = 2*(ceil(sigma_pixel));
if(~mod(kernel_size,2))
    kernel_size=kernel_size+1;
end
blurbound=floor(kernel_size/2);

newv=vraw-2*blurbound;
newh=hraw-2*blurbound;

hfov=floor(newh/pixel2PR);
vfov=floor(newv/pixel2PR);

newsize=[2*blurbound+(vfov-1)*pixel2PR+1,2*blurbound+(hfov-1)*pixel2PR+1];

% Generate the input imagery for the model to run on
im_ss=zeros(vfov,hfov,part_size);

start_frame=(partnum-1)*part_size+1;
end_frame=partnum*part_size+part_padding;
set(vread,'CurrentTime',(start_frame-1)/framerate);
tic

framenum=1;
while (hasFrame(vread) && framenum <= part_size+part_padding)
    f=readFrame(vread);
    img=imgaussfilt(f,sigma_pixel,'FilterSize',kernel_size);
    img_ss=img( (1+blurbound):pixel2PR:newsize(1)-blurbound,(1+blurbound):pixel2PR:newsize(2)-blurbound);
    im_ss(:,:,framenum)=img_ss;
    framenum=framenum+1;
end
disp(['Imagery generated over ' num2str(toc) ' seconds.'])
im_ss=im_ss/255;
im_ss=im_ss(:,:,1:framenum-1); % If this was the last part then it will have fewer frames
[~,~,frames]=size(im_ss);

frame_times=(0:frames-1)*frame_period;
frame_modelstep=ceil(frame_times/Ts); % Which model step to match against each frame
im_ss=timeseries(im_ss,frame_times);

simtime=(frames-1)*frame_period;
simtime=floor(simtime/Ts)*Ts; % Ensure simtime is a fixed multiple of the step size

assignin('base','TimeStep',Ts);
assignin('base','hdegs',hfov);
assignin('base','vdegs',vfov);

assignin('base','alpha_upslope_DARK_on',Ts/(tcs(1)+Ts));
assignin('base','alpha_downslope_DARK_on',Ts/(tcs(2)+Ts));
assignin('base','alpha_upslope_DARK_off',Ts/(tcs(3)+Ts));
assignin('base','alpha_downslope_DARK_off',Ts/(tcs(4)+Ts));

% Reverse the time constants for the LIGHT targets
assignin('base','alpha_upslope_LIGHT_on',Ts/(tcs(3)+Ts));
assignin('base','alpha_downslope_LIGHT_on',Ts/(tcs(4)+Ts));
assignin('base','alpha_upslope_LIGHT_off',Ts/(tcs(1)+Ts));
assignin('base','alpha_downslope_LIGHT_off',Ts/(tcs(2)+Ts));

% Run for each of the kernel sizes etc
maxlocs_DARK.value=zeros(numel(cs_kernel_size_s),part_size,10);
maxlocs_DARK.row=zeros(numel(cs_kernel_size_s),part_size,10);
maxlocs_DARK.column=zeros(numel(cs_kernel_size_s),part_size,10);

maxlocs_LIGHT=maxlocs_DARK;
maxlocs_COMBO=maxlocs_DARK;

targresp.LIGHT=NaN(numel(cs_kernel_size_s),part_size);
targresp.DARK=targresp.LIGHT;
targresp.COMBO=targresp.LIGHT;

bg_counts.LIGHT=zeros(numel(cs_kernel_size_s),numel(bg_edges)-1);
bg_counts.DARK=bg_counts.LIGHT;
bg_counts.COMBO=bg_counts.LIGHT;
% Load in ground truth for the purpose of figuring out the target responses
ssgt=ConvertGroundTruth18_08_09(gtfile,pixel2PR,blurbound,hfov,vfov);
% Need to extract out the frames that relate to this part


for k=1:numel(cs_kernel_size_s)
    cs_kernel_size=cs_kernel_size_s(k);
    pad_size=floor(cs_kernel_size/2);
    
    assignin('base','fdsr_lp',fdsr_lp_s(k));
    assignin('base','pad_size_param',pad_size);
    
    cs_kernel=zeros(cs_kernel_size);
    
    cs_kernel_val=-16/(4*cs_kernel_size-4);
    
    cs_kernel(:,1)      = cs_kernel_val;
    cs_kernel(1,:)      = cs_kernel_val;
    cs_kernel(:,end)    = cs_kernel_val;
    cs_kernel(end,:)    = cs_kernel_val;
    
    cs_kernel((cs_kernel_size+1)/2,(cs_kernel_size+1)/2)=2;
    
    % Same on and off cs kernel for now
    cs_kernel_on=cs_kernel;
    cs_kernel_off=cs_kernel;

    d=fix(clock);
    disp(['Started k ' num2str(k) ' of ' num2str(numel(cs_kernel_size_s)) ' at ' num2str(d(4)) ':' num2str(d(5)) ':' num2str(d(6)) ' on ' num2str(d(1)) '-' num2str(d(2)) '-' num2str(d(3))]);

    tic
    p=1;

    assignin('base','cs_kernel_on',cs_kernel_on)
    assignin('base','cs_kernel_off',cs_kernel_off)
    assignin('base','im_ss',im_ss)
    
    sout=sim(char(models{p}),'ReturnWorkspaceOutputs','on',...
            'FixedStep',num2str(Ts),...
            'StopTime',num2str(simtime));
    disp(['Sim took ' num2str(toc) ' seconds'])
    
    RTC_DARK=get(get(sout,'logsout'),'RTC_Out_DARK');
    RTC_LIGHT=get(get(sout,'logsout'),'RTC_Out_LIGHT');
	sout=[];
    
    % Crunch the RTC outputs down from 1000Hz timesteps to frames time base
    framesummed_DARK=zeros(vfov,hfov,frames);
    framesummed_LIGHT=zeros(vfov,hfov,frames);
    framesummed_COMBO=framesummed_DARK;
    
    for iix=1:numel(frame_modelstep)-1
        bdark=RTC_DARK.Values.Data(:,:,frame_modelstep(iix)+1:frame_modelstep(iix+1));
        blight=RTC_LIGHT.Values.Data(:,:,frame_modelstep(iix)+1:frame_modelstep(iix+1));
        framesummed_DARK(:,:,iix+1)=sum(bdark,3);
        framesummed_LIGHT(:,:,iix+1)=sum(blight,3);
        framesummed_COMBO(:,:,iix+1)=(sum(bdark,3)+sum(blight,3))/2; % Keep the values in the same range as light and dark
    end
	RTC_DARK = [];
	RTC_LIGHT = [];
    	bdark=[];
	blight=[];


    % For the last part_size frames, save out the locations and values of
    % the top 10 responses
    for iix=part_padding+1:frames
        
        realframenum= iix - part_padding;
        
        bdark=framesummed_DARK(:,:,iix);
        blight=framesummed_LIGHT(:,:,iix);
        bcombo=framesummed_DARK(:,:,iix)+framesummed_LIGHT(:,:,iix);
        [val_DARK,I_DARK]=sort(bdark(:),'descend');
        [val_LIGHT,I_LIGHT]=sort(blight(:),'descend');
        [val_COMBO,I_COMBO]=sort(bcombo(:),'descend');
	bdark=[];
	blight=[];
	bcombo=[];
        
        [ir,ic]=ind2sub(size(bdark),I_DARK(1:10));
        maxlocs_DARK.row(k,realframenum,:)=ir;
        maxlocs_DARK.column(k,realframenum,:)=ic;
        maxlocs_DARK.value(k,realframenum,:)=val_DARK(1:10);
        
        [ir,ic]=ind2sub(size(blight),I_LIGHT(1:10));
        maxlocs_LIGHT.row(k,realframenum,:)=ir;
        maxlocs_LIGHT.column(k,realframenum,:)=ic;
        maxlocs_LIGHT.value(k,realframenum,:)=val_LIGHT(1:10);
        
        [ir,ic]=ind2sub(size(bcombo),I_COMBO(1:10));
        maxlocs_COMBO.row(k,realframenum,:)=ir;
        maxlocs_COMBO.column(k,realframenum,:)=ic;
        maxlocs_COMBO.value(k,realframenum,:)=val_COMBO(1:10);

    end
    
    % Save out the target responses whether they are in the maxima or not
    for iix=part_padding+1:frames
        realframenum= iix - part_padding; % Runs from 1 to part_size
        % Actual absolute frame position relative to the start of the video
        abs_frame_num= iix - part_padding + part_size*(partnum-1);
        if(ssgt.gtbuttonfill(abs_frame_num) == 1)
            
            x_span=max(1,ssgt.gtx_ss(abs_frame_num)-1):min(hfov,ssgt.gtx_ss(abs_frame_num)+1);
            y_span=max(1,ssgt.gty_ss(abs_frame_num)-1):min(vfov,ssgt.gty_ss(abs_frame_num)+1);
            
            targresp.LIGHT(k,realframenum)=max(max(framesummed_LIGHT(y_span,x_span,iix),[],2),[],1);
            targresp.DARK(k,realframenum)=max(max(framesummed_DARK(y_span,x_span,iix),[],2),[],1);
            targresp.COMBO(k,realframenum)=max(max(framesummed_COMBO(y_span,x_span,iix),[],2),[],1);
%             
%             targresp.LIGHT(k,realframenum)=framesummed_LIGHT(ssgt.gty_ss(abs_frame_num),ssgt.gtx_ss(abs_frame_num),iix);
%             targresp.DARK(k,realframenum)=framesummed_DARK(ssgt.gty_ss(abs_frame_num),ssgt.gtx_ss(abs_frame_num),iix);
%             targresp.COMBO(k,realframenum)=framesummed_COMBO(ssgt.gty_ss(abs_frame_num),ssgt.gtx_ss(abs_frame_num),iix);
        end
    end
    
    % Bin the background by value
    bg_mask=true(vfov,hfov,frames);
    for iix = part_padding+1:frames
        realframenum= iix - part_padding;
        abs_frame_num= iix - part_padding + part_size*(partnum-1);
        % remove target responses from the mask
        if(ssgt.gtbuttonfill(abs_frame_num) == 1)
            bg_mask(ssgt.gty_ss(abs_frame_num),ssgt.gtx_ss(abs_frame_num),iix)=false;
        end
    end
    
    bgresp.LIGHT=framesummed_LIGHT(bg_mask);
    bgresp.DARK=framesummed_DARK(bg_mask);
    bgresp.COMBO=framesummed_COMBO(bg_mask);
    
    bg_counts.LIGHT(k,:)=histcounts(bgresp.LIGHT(:),bg_edges);
    bg_counts.DARK(k,:)=histcounts(bgresp.DARK(:),bg_edges);
    bg_counts.COMBO(k,:)=histcounts(bgresp.COMBO(:),bg_edges);

    bgresp=[];
    
    
    
end

save(savename,'maxlocs_DARK','maxlocs_LIGHT','maxlocs_COMBO',...
    'cs_kernel_size_s','fdsr_lp_s',...
    'part_size','part_padding',...
    'tcs','input_video_end','partnum',...
    'numparts','pixel2PR','sigma_hw_pixel',...
    'grp_file','grp_index','targresp',...
    'bg_edges','bg_counts')

% end
