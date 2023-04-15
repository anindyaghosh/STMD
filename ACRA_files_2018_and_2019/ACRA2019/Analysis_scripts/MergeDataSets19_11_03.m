clear all
close all
% This script mashes the data sets together
% TrackerFull19_10_03 
% TrackerFull19_10_30
% TrackerFull19_11_01

% Found from comparing that:
% 1. my fac implementation with high gain seemed to
% do better on hard sets than the others but didn't do as well on some easy
% sets
%
% 2. Constgain did better with a higher gain value
%
% 3. For some reason the ZB_fac implementations did worse, probably has to
% do with the threshold.

which_pc='mecheng';
if(strcmp(which_pc,'mecheng'))
    load('D:\temp\TrackerParamFile19_10_03.mat');
    paramfile2=load('D:\temp\TrackerParamFile19_10_30.mat');
    paramfile3=load('D:\temp\TrackerParamFile19_11_01.mat');
    
    load('D:\temp\lost_it19_11_03_set_1.mat');
    datafile2=load('D:\temp\lost_it19_11_03_set_2.mat');
    datafile3=load('D:\temp\lost_it19_11_03_set_3.mat');
    
    save_dir = 'D:\temp\';
elseif(strcmp(which_pc,'mrblack'))
    load('E:\PHD\TEST\TrackerParamFile19_10_03.mat');
    paramfile2=load('E:\PHD\TEST\TrackerParamFile19_10_30.mat');
    paramfile3=load('E:\PHD\TEST\TrackerParamFile19_11_01.mat');

    load('E:\PHD\TEST\lost_it19_11_03_set_1.mat');
    datafile2=load('E:\PHD\TEST\lost_it19_11_03_set_2.mat');
    datafile3=load('E:\PHD\TEST\lost_it19_11_03_set_3.mat');
    save_dir = 'E:\PHD\TEST\';
end



% map the trackers in 19_10_03 file to trackers in 19_10_30 file
utrack = unique(outer_var.sets.tracker);

% Replace the tracker in col 1 with col 2 from datafile2
replacewith2 = {'constgain','constgain';...
    'constgain_dark','constgain_dark';...
    'facilitation','facilitation';...
    'facilitation_dark','facilitation_dark'};

% replacewith2 = {'constgain','constgain';...
%     'constgain_dark','constgain_dark'};

replacewith2_nums = zeros(6,1);

for k = 1:6
    mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == k;
    Ix = find(mask,1,'first');
    track_string = paramfile2.outer_var.sets.tracker{paramfile2.outer_ix(Ix)};
    
    for t=1:size(replacewith2,1)
        if(strcmp(track_string,replacewith2{t,2}))
            for q=1:10
                % Find the matching tracker in the main datafile
                inner_mask = outer_var.sets.tracker_num(outer_ix) == q;
                inner_Ix = find(inner_mask,1,'first');
                inner_track_string = outer_var.sets.tracker{outer_ix(inner_Ix)};
                if(strcmp(inner_track_string,track_string))
                    replacewith2_nums(k) = q; % Indicates that number q should be replaced with data from k
                    break;
                end
            end
        end
    end
end

% matched_number = zeros(6,1);
% unique(paramfile2.outer_var.sets.tracker_num);
% for k=1:6
%     mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == k;
%     Ix = find(mask,1,'first');
%     track_string = paramfile2.outer_var.sets.tracker{paramfile2.outer_ix(Ix)};
%     % Find the number in the first paramfile which matches
%     for t=1:10
%         inner_mask = outer_var.sets.tracker_num(outer_ix) == t;
%         inner_Ix = find(inner_mask,1,'first');
%         inner_track_string = outer_var.sets.tracker{inner_Ix};
%         if(strcmp(inner_track_string,track_string))
%             matched_number(k)=t;
%             break;
%         end
%     end
% end

tracker_num = outer_var.sets.tracker_num(outer_ix);
hfov = outer_var.sets.hfov(outer_ix);
image_num = outer_var.sets.image_num(outer_ix);
draw_distractors = outer_var.sets.draw_distractors(outer_ix);
do_predictive_turn = outer_var.sets.do_predictive_turn(outer_ix);
do_saccades = outer_var.sets.do_saccades(outer_ix);
saccade_duration = inner_var.saccade_duration;
assumed_fwd_vel = inner_var.assumed_fwd_vel;

for k=1:8640
    this_outer = paramfile2.outer_ix(k);
    this_tracker_num = paramfile2.outer_var.sets.tracker_num(this_outer);
    this_hfov = paramfile2.outer_var.sets.hfov(this_outer);
    this_image_num = paramfile2.outer_var.sets.image_num(this_outer);
    this_draw_distractors = paramfile2.outer_var.sets.draw_distractors(this_outer);
    this_do_predictive_turn = paramfile2.outer_var.sets.do_predictive_turn(this_outer);
    this_do_saccades = paramfile2.outer_var.sets.do_saccades(this_outer);
    
    this_saccade_duration = paramfile2.inner_var.saccade_duration(k);
    this_assumed_fwd_vel = paramfile2.inner_var.assumed_fwd_vel(k);
    
    if(replacewith2_nums(this_tracker_num) ~= 0)
        % In this case, should replace the data in the matching index
        match_mask =    tracker_num == replacewith2_nums(this_tracker_num) &...
                        hfov == this_hfov &...
                        image_num == this_image_num &...
                        draw_distractors == this_draw_distractors &...
                        do_predictive_turn == this_do_predictive_turn &...
                        do_saccades == this_do_saccades;
    
        if(this_do_saccades)
            match_mask = match_mask & saccade_duration == this_saccade_duration;
        end
        
        if(this_draw_distractors)
            match_mask = match_mask & assumed_fwd_vel == this_assumed_fwd_vel;
        end
        
        if(sum(match_mask) ~= 1)
            disp('Something has gone wrong')
            return
        end
        
        match_ix = find(match_mask,1,'first');
        % Replace the match_mask index data with data from datafile2
        logs.num_saccades(match_ix,:) = datafile2.logs.num_saccades(k,:);
        logs.num_good_saccades(match_ix,:) = datafile2.logs.num_good_saccades(k,:);
        logs.num_bad_saccades(match_ix,:) = datafile2.logs.num_bad_saccades(k,:);
        logs.t_saccade_onscreen(match_ix,:) = datafile2.logs.t_saccade_onscreen(k,:);
        
        logs.vel30.first.median(match_ix,:) = datafile2.logs.vel30.first.median(k,:);
        logs.vel30.first.max(match_ix,:) = datafile2.logs.vel30.first.max(k,:);
        logs.vel30.first.mean(match_ix,:) = datafile2.logs.vel30.first.mean(k,:);
        
        logs.vel30.last.median(match_ix,:) = datafile2.logs.vel30.last.median(k,:);
        logs.vel30.last.max(match_ix,:) = datafile2.logs.vel30.last.max(k,:);
        logs.vel30.last.mean(match_ix,:) = datafile2.logs.vel30.last.mean(k,:);
        
        logs.target_30ms.first.median(match_ix,:) = datafile2.logs.target_30ms.first.median(k,:);
        logs.target_30ms.first.max(match_ix,:) = datafile2.logs.target_30ms.first.max(k,:);
        logs.target_30ms.first.mean(match_ix,:) = datafile2.logs.target_30ms.first.mean(k,:);
        
        logs.target_30ms.last.median(match_ix,:) = datafile2.logs.target_30ms.last.median(k,:);
        logs.target_30ms.last.max(match_ix,:) = datafile2.logs.target_30ms.last.max(k,:);
        logs.target_30ms.last.mean(match_ix,:) = datafile2.logs.target_30ms.last.mean(k,:);
        
        logs.distractor_30ms.first.median(match_ix,:) = datafile2.logs.distractor_30ms.first.median(k,:);
        logs.distractor_30ms.first.max(match_ix,:) = datafile2.logs.distractor_30ms.first.max(k,:);
        logs.distractor_30ms.first.mean(match_ix,:) = datafile2.logs.distractor_30ms.first.mean(k,:);
        
        logs.distractor_30ms.last.median(match_ix,:) = datafile2.logs.distractor_30ms.last.median(k,:);
        logs.distractor_30ms.last.max(match_ix,:) = datafile2.logs.distractor_30ms.last.max(k,:);
        logs.distractor_30ms.last.mean(match_ix,:) = datafile2.logs.distractor_30ms.last.mean(k,:);
        
        logs.ispresent(match_ix,:) = datafile2.logs.ispresent(k,:);
        
        logs.t_lost_it.first(match_ix,:) = datafile2.logs.t_lost_it.first(k,:);
        logs.t_lost_it.last(match_ix,:) = datafile2.logs.t_lost_it.last(k,:);
        logs.t_lost_it.target_distractor_ratio(match_ix,:) = datafile2.logs.t_lost_it.target_distractor_ratio(k,:);
        logs.t_lost_it.cause_first(match_ix,:) = datafile2.logs.t_lost_it.cause_first(k,:);
        logs.t_lost_it.cause_last(match_ix,:) = datafile2.logs.t_lost_it.cause_last(k,:);
    end
end

save([save_dir 'merged_data19_11_04.mat'],'logs','gind_to_load','replacewith2')
