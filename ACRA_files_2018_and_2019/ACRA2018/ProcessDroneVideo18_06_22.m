
function ProcessDroneVideo18_06_22(grp_index,grp_file,groupname,cs_kernel_size_range,fdsr_lp_range)
% grp_index=1;
% grp_file='D:\Johnstuff\Matlab\Data\paramfiles\DroneVidParams18_06_22.txt';
% groupname='DroneVidProcessed';
% cs_kernel_size_range=3:2:13; % Keep it odd
% fdsr_lp_range=[0.005 0.0250    0.0450    0.0650    0.0850];

% End of feed in
part_size=600; % How many frames per part
part_padding=150; % How many frames at the start for warm-up
errorflag=false;

% locroot='D:\Johnstuff\Matlab\Data\';
% vid_dir='D:\Johnstuff\Matlab\Data\GIC_Actual\Pursuit1\fragmented_video\';
locroot='/fast/users/a1119946/simfiles/dronefiles/';
vid_dir='/fast/users/a1119946/dronevids/';

[cs_kernel_size_s,fdsr_lp_s]=meshgrid(cs_kernel_size_range,fdsr_lp_range);

% Index, name, part, numparts, pixel2PR, sigma_hw
file=fopen(grp_file,'r');
grp_info=textscan(file,'%f%s%f%f%f%f','Headerlines',1,'Delimiter','\t');
fclose(file);

input_video_end=char(grp_info{2}(grp_index));
partnum=grp_info{3}(grp_index);
numparts=grp_info{4}(grp_index);
pixel2PR=grp_info{5}(grp_index);
sigma_hw=grp_info{6}(grp_index);

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
tcs=[0.005 0.3 0.005 0.005];

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
sigma_deg = sigma_hw/2.35;
sigma_pixel = sigma_deg*pixel2PR;
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
try
    set(vread,'CurrentTime',(start_frame-1)/framerate);
catch
    disp('Setting a valid end time')
    set(vread,'CurrentTime',(vid_frames-800)/framerate);
    errorflag=true;
end
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

for k=1:numel(cs_kernel_size_s)
    cs_kernel_size=cs_kernel_size_s(k);
    pad_size=floor(cs_kernel_size/2);
    
    assignin('base','fdsr_lp',fdsr_lp_s(k));
    assignin('base','pad_size_param',pad_size);
    
    cs_kernel=zeros(cs_kernel_size);
    
    cs_kernel(:,1)      = -1;
    cs_kernel(1,:)      = -1;
    cs_kernel(:,end)    = -1;
    cs_kernel(end,:)    = -1;
    
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
    
    % Crunch the RTC outputs down from 1000Hz timesteps to frames time base
    framesummed_DARK=zeros(vfov,hfov,frames);
    framesummed_LIGHT=zeros(vfov,hfov,frames);
    
    for iix=1:numel(frame_modelstep)-1
        bdark=RTC_DARK.Values.Data(:,:,frame_modelstep(iix)+1:frame_modelstep(iix+1));
        blight=RTC_LIGHT.Values.Data(:,:,frame_modelstep(iix)+1:frame_modelstep(iix+1));
        framesummed_DARK(:,:,iix+1)=sum(bdark,3);
        framesummed_LIGHT(:,:,iix+1)=sum(blight,3);
    end
    
    % For the last part_size frames, save out the locations and values of
    % the top 10 responses
    for iix=part_padding+1:frames
        
        realframenum= iix - part_padding;
        
        bdark=framesummed_DARK(:,:,iix);
        blight=framesummed_LIGHT(:,:,iix);
        [val_DARK,I_DARK]=sort(bdark(:),'descend');
        [val_LIGHT,I_LIGHT]=sort(blight(:),'descend');
        
        [ir,ic]=ind2sub(size(bdark),I_DARK(1:10));
        maxlocs_DARK.row(k,realframenum,:)=ir;
        maxlocs_DARK.column(k,realframenum,:)=ic;
        maxlocs_DARK.value(k,realframenum,:)=val_DARK(1:10);
        
        [ir,ic]=ind2sub(size(blight),I_LIGHT(1:10));
        maxlocs_LIGHT.row(k,realframenum,:)=ir;
        maxlocs_LIGHT.column(k,realframenum,:)=ic;
        maxlocs_LIGHT.value(k,realframenum,:)=val_LIGHT(1:10);
    end
end

save(savename,'maxlocs_DARK','maxlocs_LIGHT',...
    'cs_kernel_size_s','fdsr_lp_s',...
    'part_size','part_padding',...
    'tcs','input_video_end','partnum',...
    'numparts','pixel2PR','sigma_hw',...
    'grp_file','grp_index','errorflag')

% end
