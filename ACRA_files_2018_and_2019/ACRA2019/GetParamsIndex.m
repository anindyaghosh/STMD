clear all
close all

% Retrieve the index for given parameters of interest

param_settings_file = 'E:\PHD\TEST\TrackerParamFile19_10_03.mat';
load(param_settings_file)

hfov = outer_var.sets.hfov(outer_ix);
image_num = outer_var.sets.image_num(outer_ix);
image_name=outer_var.sets.image_name(outer_ix);
tracker_num = outer_var.sets.tracker_num(outer_ix);
tracker = outer_var.sets.tracker(outer_ix);
draw_distractors = outer_var.sets.draw_distractors(outer_ix);
do_predictive_turn = outer_var.sets.do_predictive_turn(outer_ix);
do_saccades = outer_var.sets.do_saccades(outer_ix);
saccade_duration = inner_var.saccade_duration;
assumed_fwd_vel = inner_var.assumed_fwd_vel;

% Print the tracker names and numbers 
for k=1:numel(unique(tracker))
    ix=find(tracker_num == k,1,'first');
    disp([num2str(k) ' ' tracker{ix}])
end
disp(' ')
% Print the image names and numbers
for k=1:numel(unique(image_num))
    ix=find(image_num == k,1,'first');
    disp([num2str(k) ' ' image_name{ix}])
end

% Desired
des_hfov=60;
des_image_num = 1;
des_tracker_num = 1;
des_draw_distractors = true;
des_do_predictive_turn = false;
des_do_saccades = false;

des_saccade_duration = 0.025;
des_assumed_fwd_vel = 1;

mask =  hfov == des_hfov &...
        image_num == des_image_num &...
        tracker_num == des_tracker_num &...
        draw_distractors == des_draw_distractors &...
        do_predictive_turn == des_do_predictive_turn;
    
if(des_draw_distractors)
    mask = mask & assumed_fwd_vel == des_assumed_fwd_vel;
end
if(des_do_saccades)
    mask = mask & saccade_duration == des_saccade_duration;
end

gind = find(mask,1,'first');
disp(' ')
disp(['Use gind=' num2str(gind)])
return
% no distractors, Hill(6), 60 deg, no saccades, predict: 1817
% 1m/s distractors, Botanic(1), 60 deg, no saccades, predict: 2254
% 3m/s distractors, Park(10), 60 deg, no saccades, predict: 2336
% 6m/s distractors, Creek bed(3), 60 deg, no saccades, predict: 2274

% no distractors, Botanic, 60 deg, no saccades, no predict: 2
% no distractors, Botanic, 60 deg, 25ms saccades, no predict: 3604
% no distractors, Botanic, 60 deg, 50ms saccades, no predict: 3605
% no distractors, Botanic, 60 deg, 100ms saccades, no predict: 3606
% no distractors, Botanic, 60 deg, 25ms saccades, predict: 9004
% no distractors, Botanic, 60 deg, 50ms saccades, predict: 9005
% no distractors, Botanic, 60 deg, 100ms saccades, predict: 9006

%%
TMMakeVideo(17)
TMMakeVideo(1817)
TMMakeVideo(2254)
TMMakeVideo(2336)
TMMakeVideo(2274)
TMMakeVideo(2)
TMMakeVideo(3604)
TMMakeVideo(3605)
TMMakeVideo(3606)
TMMakeVideo(9004)
TMMakeVideo(9005)
TMMakeVideo(9006)
disp('Done')

%%
rootdir='E:\PHD\pres_temp\';
distractor_names={'vid1817-25.mat';...
    'vid2254-25.mat';...
    'vid2336-23.mat'};
%     'vid2274-17.mat'};

saccades_names={...%'vid2-25.mat';...
    'vid3604-25.mat';...
    'vid3605-25.mat';...
    'vid3606-25.mat'};


% draw_range={'saccades','distractors'};
draw_range={'saccades'};

for outer_ix=1:numel(draw_range)
    to_draw=draw_range{outer_ix}

    if(strcmp(to_draw,'saccades'))
        now_drawing_names=saccades_names;
        vid_out_name='E:\PHD\pres_temp\videos\all_saccades2.mp4';
    elseif(strcmp(to_draw,'distractors'))
        now_drawing_names=distractor_names;
        vid_out_name='E:\PHD\pres_temp\videos\all_distractors2.mp4';
    end

    frames_out=zeros(183+210,420*numel(now_drawing_names),2000);
    tic
    for k=1:numel(distractor_names)
        toc
        a=load([rootdir now_drawing_names{k}]);

        subs_top = a.complete_frames(1:2:end,1:2:end,:);
        for t=1:2000
            top_part = [zeros(183,29) subs_top(:,:,t) zeros(183,28)];
            bottom_part = repelem(a.subs_frames(:,:,t),7,7);
            frames_out(:,(k-1)*440+1:k*440-20,t)=[top_part;bottom_part];
        end
    end
    vidObj=VideoWriter(vid_out_name,'MPEG-4');
    vidObj.FrameRate=30;
    vidObj.Quality=90;
    open(vidObj)
    for t=1:2000
        writeVideo(vidObj,frames_out(:,:,t))
    end
    close(vidObj)
end

%%
a=load('E:\PHD\pres_temp\tst_14inf_0fr_1de_0trackres_2254.mat');
trackdemo_out_name = 'E:\PHD\pres_temp\videos\trackdemo.mp4';
vidObj=VideoWriter(trackdemo_out_name,'MPEG-4');
vidObj.FrameRate=30;
vidObj.Quality=90;
open(vidObj)
for t=1:1567
    writeVideo(vidObj,video_frames(:,:,:,t))
end
close(vidObj)

%%
a=load('E:\PHD\pres_temp\basic_stuff.mat');
trackdemo_out_name = 'E:\PHD\pres_temp\videos\basic_stuff.mp4';
vidObj=VideoWriter(trackdemo_out_name,'MPEG-4');
vidObj.FrameRate=10;
vidObj.Quality=90;
open(vidObj)
for t=1:80
    left_part = a.complete_frames(:,:,t);
    right_part = [zeros(3,480);repelem(a.subs_frames(:,:,t),12,12); zeros(2,480)];
    writeVideo(vidObj,[left_part zeros(245,20) right_part])
end
close(vidObj)
