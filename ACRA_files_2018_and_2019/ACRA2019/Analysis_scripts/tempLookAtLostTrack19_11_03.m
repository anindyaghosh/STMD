clear all
close all

% Data merged with MergeDataSets19_11_03

which_pc='mrblack';

if(strcmp(which_pc,'mrblack'))
    params = load('E:\PHD\TEST\TrackerParamFile19_10_03.mat');
    load('E:\PHD\TEST\merged_data19_11_03.mat');
%     load('E:\PHD\TEST\lost_it19_10_31_set_ 1.mat');
%     load('E:\PHD\TEST\temp_lost_it_19_10_22.mat');

    figures_out_dir = 'E:\PHD\conf2\autofigures19_11_03\';
    
    logs.ispresent(isnan(logs.ispresent)) = false;
elseif(strcmp(which_pc,'mecheng'))
    params=load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');
    traj_rmtv=load('D:\simfiles\conf2\RunningMedianTargetVelocity19_10_11_gb30.mat');
    load('D:\temp\merged_data19_11_04.mat');
    figures_out_dir = 'D:\simfiles\conf2\autofigures19_11_04\';
    
    % TEMPORARY HACK
    logs.ispresent(isnan(logs.ispresent)) = false;
end

% If the trial is complete and lost_it is NaN, then this means tracking was
% never lost
m1 = isnan(logs.t_lost_it.first);
m2 = logs.ispresent == true;
logs.t_lost_it.first(m1 & m2) = 30251;

m1 = isnan(logs.t_lost_it.last);
m2 = logs.ispresent == true;
logs.t_lost_it.last(m1 & m2) = 30251;

gind_range = 1:numel(params.outer_ix);
numgind = numel(gind_range);
spin_ticks = 250;

[~,num_traj] = size(logs.t_lost_it.first);

% Create masks for the different important settings

% Tracker nums:
% 1 - prob
% 2 - prob_fixed
% 3 - facilitation
% 4 - facilitation_dark
% 5 - kf
% 6 - kf_dark
% 7 - constgain
% 8 - constgain_dark
% 9 - none
% 10 - none_dark

num_trackers=10;

% Get vectors of the settings indexed by gind
hfov=params.outer_var.sets.hfov(params.outer_ix);
image_num=params.outer_var.sets.image_num(params.outer_ix);
tracker_num=params.outer_var.sets.tracker_num(params.outer_ix);

image_names = unique(params.outer_var.sets.image_name);

draw_distractors=params.outer_var.sets.draw_distractors(params.outer_ix);
do_predictive_turn=params.outer_var.sets.do_predictive_turn(params.outer_ix);
do_saccades=params.outer_var.sets.do_saccades(params.outer_ix);

% The inner vars are already indexed by gind but keep a consistent
% interface going
saccade_duration = params.inner_var.saccade_duration;
assumed_fwd_vel = params.inner_var.assumed_fwd_vel;

% Create tracker masks
prob_mask = tracker_num == 1;
% prob_fixed_mask = tracker_num == 2;
% facilitation_mask = tracker_num == 3;
% facilitation_dark_mask = tracker_num == 4;
% kf_mask = tracker_num == 5;
% kf_dark_mask = tracker_num == 6;

tracker_masks = cell(num_trackers,1);
for p=1:num_trackers
    tracker_masks{p} = tracker_num == p;
end

b=median(logs.t_lost_it.first,2,'omitnan');
%%=
% fac_mean = mean(logs.t_lost_it.first(facilitation_mask,:),2,'omitnan');
% prob_mean = mean(logs.t_lost_it.first(prob_mask,:),2,'omitnan');

% Find indices matching every prob_tracker result
close all
I_prob = gind_range(tracker_masks{1});

I_all_traj = NaN(numel(I_prob),1);
I_tracker = NaN(numel(I_prob),num_trackers);

for ix=1:numel(I_prob)
    gind_prob = gind_range(I_prob(ix));
    % Get the settings
    x_hfov = hfov(gind_prob);
    x_do_predictive_turn = do_predictive_turn(gind_prob);
    x_do_saccades = do_saccades(gind_prob);
    x_draw_distractors = draw_distractors(gind_prob);
    x_image_num = image_num(gind_prob);
    x_saccade_duration = saccade_duration(gind_prob);
    x_assumed_fwd_vel = assumed_fwd_vel(gind_prob);
    
    match_mask = x_hfov == hfov &...
        x_do_predictive_turn == do_predictive_turn &...
        x_do_saccades == do_saccades &...
        x_draw_distractors == draw_distractors &...
        x_image_num == image_num &...
        (x_saccade_duration == saccade_duration | (isnan(x_saccade_duration) & isnan(saccade_duration))) &...
        (x_assumed_fwd_vel == assumed_fwd_vel | (isnan(x_assumed_fwd_vel) & isnan(assumed_fwd_vel)));
    
    
    for pinner=1:num_trackers
        % Get the ginds for comparisons
        I_tracker(ix,pinner) = gind_range(match_mask & tracker_masks{pinner});
    end
    
    % Work out how many trajectories are available at minimum for the index
    temp = true(1,num_traj);
    for pinner=1:num_trackers
        temp = temp & ~isnan(logs.t_lost_it.first(I_tracker(ix,pinner),:));
    end
    % The index of the last trajectory that all variants have data for
    temp = find(temp,1,'last');
    if(~isempty(temp))
        I_all_traj(ix) = temp;
    end
end

results.mean=NaN(numel(I_prob),num_trackers);
results.median=results.mean;

for ix=1:numel(I_prob)
    if(~isnan(I_all_traj(ix)))
        for pinner = 1:num_trackers
            results.mean(ix,pinner) = mean(logs.t_lost_it.first(I_tracker(ix,pinner),1:I_all_traj(ix)),2,'omitnan');
            results.median(ix,pinner) = median(logs.t_lost_it.first(I_tracker(ix,pinner),1:I_all_traj(ix)),2,'omitnan');
        end
    end
end

% Get I_prob for a single background image
I_prob_single_image = gind_range(tracker_masks{1} & image_num == 1);
I_all_traj_singleimage = NaN(numel(I_prob_single_image),1);

for ix=1:numel(I_prob_single_image)
    gind_prob = gind_range(I_prob_single_image(ix));
    % Get the settings
    x_hfov = hfov(gind_prob);
    x_do_predictive_turn = do_predictive_turn(gind_prob);
    x_do_saccades = do_saccades(gind_prob);
    x_draw_distractors = draw_distractors(gind_prob);
    x_saccade_duration = saccade_duration(gind_prob);
    x_assumed_fwd_vel = assumed_fwd_vel(gind_prob);
    
    match_mask = x_hfov == hfov &...
        x_do_predictive_turn == do_predictive_turn &...
        x_do_saccades == do_saccades &...
        x_draw_distractors == draw_distractors &...
        (x_saccade_duration == saccade_duration | (isnan(x_saccade_duration) & isnan(saccade_duration))) &...
        (x_assumed_fwd_vel == assumed_fwd_vel | (isnan(x_assumed_fwd_vel) & isnan(assumed_fwd_vel)));
    
    % Want to know the minimum trajectories for all trackers on all images
    % matching these settings
    % Work out how many trajectories are available at minimum for the index
    
    I_to_use = find(match_mask);
    temp = true(1,num_traj);
    for pinner=1:numel(I_to_use)
        temp = temp & logs.ispresent(I_to_use(pinner),:);
%         temp = temp & ~isnan(logs.t_lost_it.first(I_to_use(pinner),:));
%         if(sum(temp) == 0)
%             return
%         end
    end
    
    % The index of the last trajectory that all variants have data for
    temp = find(temp,1,'last');
    if(~isempty(temp))
        I_all_traj_singleimage(ix) = temp;
    end
end

% Results.raw will contain all of the individual trials, however it will
% ensure an even sampling across the different images

results.raw = NaN(numel(I_prob_single_image),num_trackers,numel(unique(image_num)),100);
results.raw_last = results.raw;

results.first.lost_raw = results.raw;
results.first.cause = results.raw;
results.first.vel30.mean = results.raw;
results.first.vel30.median = results.raw;
results.first.vel30.max = results.raw;
results.first.distractor_30ms = results.first.vel30;
results.first.target_30ms = results.first.vel30;

results.last = results.first;

results.num_good_saccades = results.raw;
results.num_bad_saccades  = results.raw;

% Populate results.raw
uimg=unique(image_num);
for ix=1:numel(I_prob_single_image)
    gind_prob = gind_range(I_prob_single_image(ix));
    % Get the settings
    x_hfov = hfov(gind_prob);
    x_do_predictive_turn = do_predictive_turn(gind_prob);
    x_do_saccades = do_saccades(gind_prob);
    x_draw_distractors = draw_distractors(gind_prob);
    x_saccade_duration = saccade_duration(gind_prob);
    x_assumed_fwd_vel = assumed_fwd_vel(gind_prob);
    
    for pinner=1:num_trackers
        for img_ix = 1:numel(uimg)
            match_mask = x_hfov == hfov &...
                x_do_predictive_turn == do_predictive_turn &...
                x_do_saccades == do_saccades &...
                x_draw_distractors == draw_distractors &...
                (x_saccade_duration == saccade_duration | (isnan(x_saccade_duration) & isnan(saccade_duration))) &...
                (x_assumed_fwd_vel == assumed_fwd_vel | (isnan(x_assumed_fwd_vel) & isnan(assumed_fwd_vel))) &...
                tracker_masks{pinner} &...
                image_num == img_ix;
            
            I_get=find(match_mask);
            
            results.raw(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.first(I_get,1:I_all_traj_singleimage(ix));
            
%             m1=isnan(results.raw(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)));
%             m2=logs.ispresent(I_get,1:I_all_traj_singleimage(ix));
%             m = m1(:) &...
%                 m2(:);
%             
%             results.raw(m) = 30251;
            
            results.raw_last(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.last(I_get,1:I_all_traj_singleimage(ix));
            
%             m1=isnan(results.raw_last(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)));
%             m2=logs.ispresent(I_get,1:I_all_traj_singleimage(ix));
%             m = m1(:) &...
%                 m2(:);
%             
%             results.raw_last(m) = 30251;
            
            results.first.cause(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.cause_first(I_get,1:I_all_traj_singleimage(ix));
            
            results.last.cause(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.cause_last(I_get,1:I_all_traj_singleimage(ix));
            
            results.first.vel30.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.first.mean(I_get,1:I_all_traj_singleimage(ix));
            results.first.vel30.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.first.median(I_get,1:I_all_traj_singleimage(ix));
            results.first.vel30.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.first.max(I_get,1:I_all_traj_singleimage(ix));
            
            results.last.vel30.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.last.mean(I_get,1:I_all_traj_singleimage(ix));
            results.last.vel30.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.last.median(I_get,1:I_all_traj_singleimage(ix));
            results.last.vel30.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.vel30.last.max(I_get,1:I_all_traj_singleimage(ix));
            
            results.first.distractor_30ms.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.first.mean(I_get,1:I_all_traj_singleimage(ix));
            results.first.distractor_30ms.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.first.median(I_get,1:I_all_traj_singleimage(ix));
            results.first.distractor_30ms.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.first.max(I_get,1:I_all_traj_singleimage(ix));
            
            results.last.distractor_30ms.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.last.mean(I_get,1:I_all_traj_singleimage(ix));
            results.last.distractor_30ms.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.last.median(I_get,1:I_all_traj_singleimage(ix));
            results.last.distractor_30ms.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.distractor_30ms.last.max(I_get,1:I_all_traj_singleimage(ix));

            results.first.target_30ms.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.first.mean(I_get,1:I_all_traj_singleimage(ix));
            results.first.target_30ms.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.first.median(I_get,1:I_all_traj_singleimage(ix));
            results.first.target_30ms.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.first.max(I_get,1:I_all_traj_singleimage(ix));
            
            results.last.target_30ms.mean(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.last.mean(I_get,1:I_all_traj_singleimage(ix));
            results.last.target_30ms.median(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.last.median(I_get,1:I_all_traj_singleimage(ix));
            results.last.target_30ms.max(ix,pinner,img_ix,1:I_all_traj_singleimage(ix)) =...
                logs.target_30ms.last.max(I_get,1:I_all_traj_singleimage(ix));
        end
    end
end

results.first.all_ticks = results.raw;
results.last.all_ticks = results.raw_last;

% No longer need to remove spin_ticks offset because this is done in
% GetLostTrack
% Removal of spin_ticks offset
% results.raw = results.raw - spin_ticks;
% results.raw_last = results.raw_last - spin_ticks;

% Convert to seconds
results.raw = results.raw / 1000;
results.raw_last = results.raw_last / 1000;

% Collapse results across images
results.first.lost_it = results.raw(:,:,:);
results.last.lost_it = results.raw_last(:,:,:);

results.first.cause = results.first.cause(:,:,:);
results.last.cause = results.last.cause(:,:,:);

results.first.vel30.mean = results.first.vel30.mean(:,:,:);
results.first.vel30.median = results.first.vel30.median(:,:,:);
results.first.vel30.max = results.first.vel30.max(:,:,:);

results.last.vel30.mean = results.last.vel30.mean(:,:,:);
results.last.vel30.median = results.last.vel30.median(:,:,:);
results.last.vel30.max = results.last.vel30.max(:,:,:);

results.first.target_30ms.mean = results.first.target_30ms.mean(:,:,:);
results.first.target_30ms.median = results.first.target_30ms.median(:,:,:);
results.first.target_30ms.max = results.first.target_30ms.max(:,:,:);

results.last.target_30ms.mean = results.last.target_30ms.mean(:,:,:);
results.last.target_30ms.median = results.last.target_30ms.median(:,:,:);
results.last.target_30ms.max = results.last.target_30ms.max(:,:,:);

results.first.distractor_30ms.mean = results.first.distractor_30ms.mean(:,:,:);
results.first.distractor_30ms.median = results.first.distractor_30ms.median(:,:,:);
results.first.distractor_30ms.max = results.first.distractor_30ms.max(:,:,:);

results.last.distractor_30ms.mean = results.last.distractor_30ms.mean(:,:,:);
results.last.distractor_30ms.median = results.last.distractor_30ms.median(:,:,:);
results.last.distractor_30ms.max = results.last.distractor_30ms.max(:,:,:);

micro_assumed_fwd_vel = assumed_fwd_vel(I_prob_single_image);
micro_saccade_duration = saccade_duration(I_prob_single_image);
micro_hfov = hfov(I_prob_single_image);
micro_do_predictive_turn = do_predictive_turn(I_prob_single_image);
micro_do_saccade = do_saccades(I_prob_single_image);
micro_draw_distractors = draw_distractors(I_prob_single_image);

%%
% Results now contains a [1:1440, num_trackers] set of results for each tracker in
% units of seconds with the spin_ticks offset removed.
% [ prob, prob_fixed, fac, fac_dark]

% Apply masks to the results
mini_distract_mask = draw_distractors(I_prob);
mini_predictive_turn_mask = do_predictive_turn(I_prob);
mini_do_saccades_mask = do_saccades(I_prob);
mini_image_num = image_num(I_prob);
mini_hfov = hfov(I_prob);

mini_saccade_duration = saccade_duration(I_prob);
mini_assumed_fwd_vel = assumed_fwd_vel(I_prob);

%%
% Plot 15 boxes, 1 for each hfov, with fwd_vel == 1
close all
track_ix=[10 8 6 3 2 1];  % Dark selective variants
num_trackix=numel(track_ix);
track_labels=['nckfp']; % These must be single chars

% label_length = 0;
% for k=1:numel(track_labels)
%     label_length = max(label_length,numel(track_labels{k}));
% end

% micro_assumed_fwd_vel = assumed_fwd_vel(I_prob_single_image);
% micro_saccade_duration = saccade_duration(I_prob_single_image);
% micro_hfov = hfov(I_prob_single_image);
% micro_do_predictive_turn = do_predictive_turn(I_prob_single_image);
% micro_do_saccade = do_saccades(I_prob_single_image);
% micro_draw_distractors = draw_distractors(I_prob_single_image);

ufwd = unique(micro_assumed_fwd_vel(~isnan(micro_assumed_fwd_vel)));
usaccade = unique(saccade_duration(~isnan(saccade_duration)));
uhfov = unique(hfov);
temp_mask = micro_draw_distractors &...
    micro_do_predictive_turn &...
    ~micro_do_saccade &...
    micro_assumed_fwd_vel == ufwd(1) &...
    micro_hfov == uhfov(1);
numrows = sum(temp_mask);

% % Pad the labels
% label_mat = repmat(' ',numrows*num_trackix,label_length);
% for k=1:numel(track_labels)
%     for p=1:numrows
%         label_mat((k-1)*numrows+p,1:numel(track_labels{k})) = track_labels{k};
%     end
% end

% Setup labels
hfov_mat = repmat([40 60 90],1,num_trackix);
hfov_mat = repmat(hfov_mat,1500,1);

label_mat = repelem(track_labels,1,numel(uhfov));
label_mat = repmat(label_mat,1500,1);

[~,~,num_traj_in_all] = size(results.first.lost_it);
A=NaN(num_traj_in_all,num_trackix*3); % One column for each hfov and tracker
edges=0:31;
figure_ix = 1;
for last_setting = [false true]
    for predictive_setting = [false true]
        for fwd_ix = 1:numel(ufwd)
            f=figure(figure_ix);
            figure_ix = figure_ix+1;
            f.Name = ['boxplots fwd vel = ' num2str(ufwd(fwd_ix)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
            % Populate A, which will be used for the boxplots
            for k=1:num_trackix
                for p=1:3
                    mask = micro_draw_distractors &...
                        micro_do_predictive_turn == predictive_setting &...
                        ~micro_do_saccade &...
                        micro_assumed_fwd_vel == ufwd(fwd_ix) &...
                        micro_hfov == uhfov(p);
                    if(last_setting)
                        A(:,3*(k-1)+p) = results.last.lost_it(mask,track_ix(k),:);
                    else
                        A(:,3*(k-1)+p) = results.first.lost_it(mask,track_ix(k),:);
                    end
                    
                end
            end
            %             G={label_mat(:),hfov_mat(:)};
            %             boxplot(A(:),G,'factorgap',10,'colorgroup',repmat([1 2 3],1,num_trackix),'plotstyle','compact')
            
            %             f=figure(figure_ix);
            %             figure_ix = figure_ix+1;
            %             f.Name = ['histo fwd vel = ' num2str(ufwd(fwd_ix)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
            %             nc=NaN(numel(edges)-1,num_trackix*3);
            %             for k=1:num_trackix*3
            %                 nc(:,k) = histcounts(A(:,k),edges);
            %                 % Need to normalise the counts based on how many non-nan there were
            %                 nc(:,k) = nc(:,k) / sum(~isnan(A(:,k)));
            %             end
            %
            %             track_colours='rgbmk';
            %             fov_lines = {':','--','-'};
            %             for k=1:5
            %                 for p=1:3
            %                     drawchar = [track_colours(k) fov_lines{p}];
            %                     plot(0.5*(edges(1:end-1)+edges(2:end)),nc(:,3*(k-1)+p),drawchar)
            %                     hold on
            %                 end
            %             end
            
            % Plot proportion which failed prior to different times
            
            %             k_shades = [0.8 0.6 0.4 0.2 0];
            
            k_colours = [1 0 0;
                0 1 0;
                0 0 0;
                1 0 0;
                0 0 1;
                0 0 0];
            
            %     k_colours=[1 0 0; %none
            %         0 1 0 %cg
            %         0 0 1;
            %         0.5 0 0.5;
            %         0 0 0];
            %             k_lines = {'-','--','-.',':','-'};
            
            k_lines= {'-','--','-','--','-','--'};
            for p = 1:3
                f=figure(figure_ix);
                figure_ix = figure_ix+1;
                f.Name = ['Cumulative failure fwd vel = ' num2str(ufwd(fwd_ix)) ' hfov=' num2str(uhfov(p)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
                for k=1:6
                    %                     drawchar = [track_colours(k) fov_lines{p}];
                    b=A(:,3*(k-1)+p);
                    b=b(~isnan(b));
                    % Need to exclude results of 30 because that's not
                    % really a failure
                    made_to_the_end = sum(b == 30);
                    perc_failed_by = NaN(numel(b),1);
                    for inner_ix=1:numel(b)
                        perc_failed_by(inner_ix) = sum(b(1:inner_ix) < 30)/numel(b)*100;
                    end
                    %                     h=plot(sort(b),(1:numel(b))/numel(b)*100,k_lines{k},'LineWidth',2);
                    h=plot(sort(b),perc_failed_by,k_lines{k},'LineWidth',2);
                    %             h.Color = k_colours(k,:);
                    %                     h.Color = k_shades(k)*[1 1 1];
                    h.Color = k_colours(k,:);
                    hold on
                end
                plot(0.03,0.03,'w.')
                g=findall(f.Children,'Type','Axes');
                g.XScale = 'log';
                g.XTick = [0.1 0.3 1 3 10 30];
                g.XTickLabels = {'0.1','0.3','1','3','10','30'};
                g.Box='off';
                g.XLim=[0 31];
                xlabel('Time until failure (s)')
                ylabel('% failed')
                g.FontName = 'Times New Roman';
                g.FontSize = 10;
                g.Units = 'centimeters';
                g.Position(3:4) = [7.6 3];
                legend({'none','cg','kf','fac','prob (fixed vel.)','prob'})
            end
        end
    end
end

% Save out figures
if(~exist([figures_out_dir 'Distractors\'],'dir'))
    mkdir([figures_out_dir 'Distractors\'])
end
for k=1:figure_ix-1
    f=figure(k);
    saveas(f,[figures_out_dir 'Distractors\' f.Name '.fig'],'fig');
end
close all
%%
% Effect of saccade duration with fov
% Setup labels
close all
track_ix=[10 8 6 4 1];
num_trackix=numel(track_ix);
hfov_mat = repmat([40 60 90],1,num_trackix);
hfov_mat = repmat(hfov_mat,1500,1);

label_mat = repelem(track_labels,1,numel(uhfov));
label_mat = repmat(label_mat,1500,1);

[~,~,num_traj_in_all] = size(results.first.lost_it);
A=NaN(num_traj_in_all,num_trackix*3); % One column for each hfov and tracker
edges=0:31;
figure_ix = 1;
for last_setting=[false true]
    for predictive_setting=[false true]
        for sacc_ix = 1:numel(usaccade)
            f=figure(figure_ix);
            figure_ix = figure_ix+1;
            f.Name = ['boxplots saccade dur = ' num2str(usaccade(sacc_ix)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
            % Populate A, which will be used for the boxplots
            for k=1:num_trackix
                for p=1:3
                    mask = ~micro_draw_distractors &...
                        micro_do_predictive_turn == predictive_setting &...
                        micro_do_saccade &...
                        micro_saccade_duration == usaccade(sacc_ix) &...
                        micro_hfov == uhfov(p);
                    if(last_setting)
                        A(:,3*(k-1)+p) = results.last.lost_it(mask,track_ix(k),:);
                    else
                        A(:,3*(k-1)+p) = results.first.lost_it(mask,track_ix(k),:);
                    end
                end
            end
            G={label_mat(:),hfov_mat(:)};
            boxplot(A(:),G,'factorgap',10,'colorgroup',repmat([1 2 3],1,num_trackix),'plotstyle','compact')
            
            f=figure(figure_ix);
            figure_ix = figure_ix+1;
            f.Name = ['histo saccade dur = ' num2str(usaccade(sacc_ix)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
            nc=NaN(numel(edges)-1,num_trackix*3);
            for k=1:num_trackix*3
                nc(:,k) = histcounts(A(:,k),edges);
                % Need to normalise the counts based on how many non-nan there were
                nc(:,k) = nc(:,k) / sum(~isnan(A(:,k)));
            end
            
            track_colours='rgbmk';
            fov_lines = {':','--','-'};
            for k=1:5
                for p=1:3
                    drawchar = [track_colours(k) fov_lines{p}];
                    plot(0.5*(edges(1:end-1)+edges(2:end)),nc(:,3*(k-1)+p),drawchar)
                    hold on
                end
            end
            
            % Plot proportion which failed prior to different times
            
            k_shades = [0.8 0.6 0.4 0.2 0];
            k_lines = {'-','--','-.',':','-'};
            for p = 1:3
                f=figure(figure_ix);
                figure_ix = figure_ix+1;
                f.Name = ['Cumulative failure saccade duration = ' num2str(usaccade(sacc_ix)) ' hfov=' num2str(uhfov(p)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
                for k=1:5
                    drawchar = [track_colours(k) fov_lines{p}];
                    b=A(:,3*(k-1)+p);
                    b=b(~isnan(b));
                    h=plot(sort(b),(1:numel(b))/numel(b)*100,k_lines{k},'LineWidth',2);
                    %             h.Color = k_colours(k,:);
                    h.Color = k_shades(k)*[1 1 1];
                    hold on
                end
                plot(0.03,0.03,'w.')
                g=findall(f.Children,'Type','Axes');
                g.XScale = 'log';
                g.XTick = [0.1 0.3 1 3 10 30];
                g.XTickLabels = {'0.1','0.3','1','3','10','30'};
                g.Box='off';
                g.XLim=[0 30];
                xlabel('Time until failure (s)')
                ylabel('% failed')
                g.FontName = 'Times New Roman';
                g.FontSize = 10;
                g.Units = 'centimeters';
                g.Position(3:4) = [7.6 3];
                legend({'none','cg','kf','fac','prob'})
            end
        end
    end
end

% Save out figures
if(~exist([figures_out_dir 'Saccades\'],'dir'))
    mkdir([figures_out_dir 'Saccades\'])
end

for k=1:figure_ix-1
    f=figure(k);
    saveas(f,[figures_out_dir '\Saccades\' f.Name '.fig'],'fig');
end

close all

%%
% Show causes of lost tracking
% Want 3 groups of bars, one for each hfov
close all

overmask = ~micro_draw_distractors &...
    ~micro_do_saccade &...
    ~micro_do_predictive_turn;

track_ix = [10 8 6 4 1];
A=NaN(numel(uhfov),numel(track_ix));

for hfov_ix = 1:numel(uhfov)
    mask = overmask & micro_hfov == uhfov(hfov_ix);
    for pinner=1:numel(track_ix)
        temp = results.first.cause(mask,track_ix(pinner),:);
        A(hfov_ix,pinner) = sum(temp(:),'omitnan')/sum(~isnan(temp(:)));
    end
end

f=figure(1);
f.Name = 'Caused by saccade';
bar(A);

%%
% Median velocity in lead-up to failure

for hfov_setting = [40 60 90]
    for predictive_setting=[false true]
        close all
        % With distractors
        overmask = micro_draw_distractors &...
            micro_hfov == hfov_setting &...
            ~micro_do_saccade &...
            micro_do_predictive_turn == predictive_setting;
        
        track_ix = 1:10;
        
        ufwd = unique(assumed_fwd_vel(~isnan(assumed_fwd_vel)));
        A=NaN(numel(ufwd),numel(track_ix),15*100);
        
        for fwd_ix = 1:numel(ufwd)
            mask = overmask & micro_assumed_fwd_vel == ufwd(fwd_ix);
            for pinner=1:numel(track_ix)
                A(fwd_ix,pinner,:) = results.first.vel30.median(mask,track_ix(pinner),:);
            end
        end
        
        edges=0:10:150;
        nc=NaN(numel(ufwd),numel(track_ix),numel(edges)-1);
        
        k_shades = [0 0.6 0.4 0.2 0.8];
        k_lines = {'-','--','-',':','--'};
        
        track_shades = [0.8 0.6 0.4 0.2 0];
        for fwd_ix = 1:numel(ufwd)
            %     f=figure(fwd_ix);
            for pinner = 1:numel(track_ix)
                nc(fwd_ix,pinner,:)=histcounts(A(fwd_ix,pinner,:),edges);
                nc(fwd_ix,pinner,:) = nc(fwd_ix,pinner,:) / sum(nc(fwd_ix,pinner,:));
                %         b=nc(fwd_ix,pinner,:);
                %         plot(edges(1:end-1),b(:),k_lines{pinner},'Color',k_shades(pinner)*[1 1 1],'LineWidth',2)
                %         hold on
            end
        end
        
        % figure(4);
        % No distractors
        overmask = ~micro_draw_distractors &...
            micro_hfov == hfov_setting &...
            ~micro_do_saccade &...
            ~micro_do_predictive_turn;
        
        A=NaN(numel(track_ix),15*100);
        
        for pinner=1:numel(track_ix)
            A(pinner,:) = results.first.vel30.median(overmask,track_ix(pinner),:);
        end
        
        nc_none=NaN(numel(track_ix),numel(edges)-1);
        for pinner = 1:numel(track_ix)
            nc_none(pinner,:)=histcounts(A(pinner,:),edges);
            nc_none(pinner,:) = nc_none(pinner,:) / sum(nc_none(pinner,:));
            %     b=nc_none(pinner,:);
            %     plot(edges(1:end-1),b(:),k_lines{pinner},'Color',k_shades(pinner)*[1 1 1],'LineWidth',2)
            %     hold on
        end
        
        % Frequency polygon for 1) No distractors, 2-4) Distractors at different
        % velocities
        % All on one plot
        
        for tr_ix = 1:numel(track_ix)
            f=figure(tr_ix);
            f.Name = ['Target vel prior to failure ' num2str(track_ix(tr_ix)) '-predictive=' num2str(predictive_setting) '-hfov=' num2str(hfov_setting)];
            b=reshape(nc(:,tr_ix,:),numel(ufwd),numel(edges)-1);
            h=plot(0.5*(edges(1:end-1)+edges(2:end)),[nc_none(tr_ix,:)' b'],'LineWidth',2);
            k_shades = [0 0.6 0.8 0.4];
            k_linestyle = {'-','--','-',':'};
            for k=1:numel(h)
                h(k).Color = k_shades(k)*[1 1 1];
                h(k).LineStyle = k_linestyle{k};
            end
            xlabel('Median target velocity during 30ms before target lost (°/s)');
            ylabel('Relative frequency')
            legend({'None','1 m/s','3 m/s','6 m/s'},'box','off')
            g=findall(f.Children,'Type','Axes');
            g.FontName = 'Times New Roman';
            g.FontSize = 10;
            g.Units = 'centimeters';
            g.Position(3:4) = [7.6 4];
            g.Box='off';
            g.YLim=[0 1];
        end
        
        % Save out
        if(~exist([figures_out_dir 'Median_vel\'],'dir'))
            mkdir([figures_out_dir 'Median_vel\']);
        end
        
        for k=1:numel(track_ix)
            f=figure(k);
            saveas(f,[figures_out_dir 'Median_vel\' f.Name '.fig'],'fig');
        end
    end
end
close all

%%
% Combined effect of saccade duration and distractors

% fwd_vel on rows, saccade duration on cols
for last_setting = [false true]
    for hfov_setting = [40 60 90];
        for predictive_setting = [false true];
            x_masks=cell(4,4);
            
            overmask = micro_do_predictive_turn == predictive_setting &...
                micro_hfov == hfov_setting;
            
            x_masks{1,1} = overmask &...
                ~micro_draw_distractors &...
                ~micro_do_saccade;
            
            ufwd=unique(assumed_fwd_vel(~isnan(assumed_fwd_vel)));
            usacc = unique(saccade_duration(~isnan(saccade_duration)));
            
            for x=2:4
                % Distractors but no saccades
                x_masks{x,1} = overmask &...
                    micro_draw_distractors &...
                    micro_assumed_fwd_vel == ufwd(x-1) &...
                    ~micro_do_saccade;
                % Saccades but no distractors
                x_masks{1,x} = overmask &...
                    ~micro_draw_distractors &...
                    micro_do_saccade &...
                    micro_saccade_duration == usacc(x-1);
            end
            
            for x=2:4
                for y=2:4
                    % Mixed saccades and distractors
                    x_masks{x,y} = overmask &...
                        micro_draw_distractors &...
                        micro_do_saccade &...
                        micro_saccade_duration == usacc(y-1) &...
                        micro_assumed_fwd_vel == ufwd(x-1);
                end
            end
            
            % track_ix = 1;
            % % Find the median performance for these conditions
            % pz=NaN(size(x_masks));
            %
            % for k=1:numel(x_masks)
            %     pz(k) = median(results.all(x_masks{k},1,:),'omitnan');
            % end
            
            % pcolor(pz)
            
            %
            % % Exclude the no-distractors cases, located in the first row of x_masks
            % f=figure(1);
            % A=NaN(1500,12);
            %
            % g1=repmat(1000*[0 usacc'],3,1);
            % g2=repmat([ufwd],1,4);
            %
            % G={g1(:),g2(:)};
            %
            % for x=2:4
            %     for y=1:4
            %         A(:,4*(x-2)+y) = results.all(x_masks{x,y},1,:);
            %     end
            % end
            % boxplot(A,G)
            %
            % g=findall(f.Children,'Type','Axes');
            % g.Units='centimeters';
            % g.Position(3:4) = [7.6 3];
            
            % Try an errorbar plot instead with one line per saccade duration
            
            close all
            track_ix=1:10;
            for tr_ix = 1:numel(track_ix)
                f=figure(tr_ix);
                f.Name = ['hfov=' num2str(hfov_setting) 'pred=' num2str(predictive_setting) 'last=' num2str(last_setting) '-Saccades and distractors ' num2str(track_ix(tr_ix))];
                eb_med = NaN(3,4);
                eb_upper = eb_med;
                eb_lower = eb_med;
                for x=2:4
                    for y=1:4
                        if(last_setting)
                            temp = results.last.lost_it(x_masks{x,y},track_ix(tr_ix),:);
                        else
                            temp = results.first.lost_it(x_masks{x,y},track_ix(tr_ix),:);
                        end
                        temp = sort(temp(~isnan(temp)));
                        eb_med(x-1,y) = median(temp);
                        eb_upper(x-1,y) = temp(floor(0.75*numel(temp)));
                        eb_lower(x-1,y) = temp(floor(0.25*numel(temp)));
                    end
                end
                x_base = 1000*repmat([0;usacc],1,3);
                spacing = 2.5;
                % x_base = x_base + repmat([-spacing*3/2 -spacing/2 spacing/2 spacing*3/2],1,3);
                x_base = x_base + repmat([-spacing 0 spacing],4,1);
                
                M = eb_med';
                L = (eb_med')-(eb_lower');
                U = eb_upper' - eb_med';
                h=errorbar(x_base,M,L,U,'LineWidth',2);
                g=findall(f.Children,'Type','Axes');
                g.Units='centimeters';
                g.Position(3:4) = [7.6 7];
                g.Box = 'off';
                
                k_shades = [0.6 0.4 0.8 0];
                k_lines = {'-','--','-',':'};
                
                for k=1:3
                    h(k).Color = k_shades(k)*[1 1 1];
                    h(k).LineStyle =k_lines{k};
                end
                
                xlabel('Saccade duration (ms)')
                ylabel('Time to failure (s)')
                g.XTick = 1000*[0;usacc];
                g.XLim = [-5 max(1000*usacc)+10];
                legend({'1 m/s','3 m/s','6 m/s'},'box','off','FontName','Times New Roman','FontSize',10)
                g.FontName = 'Times New Roman';
                g.FontSize = 10;
            end
            
            if(~exist([figures_out_dir 'SaccDistract\'],'dir'))
                mkdir([figures_out_dir 'SaccDistract\'])
            end
            
            for k=1:numel(track_ix)
                f=figure(k);
                saveas(f,[figures_out_dir 'SaccDistract\' f.Name '.fig'],'fig');
            end
        end
    end
end

close all

%%
% Plotting relationship between failures and running median target velocity
% traj_rmtv
close all
overmask = micro_draw_distractors &...
        ~micro_do_saccade &...
        micro_do_predictive_turn;
    
track_ix = [1 2 3 6 8 10];

% Bin the rmtv
edges = 0:10:190;
ufwd = unique(assumed_fwd_vel(~isnan(assumed_fwd_vel)));

figure_ix = 1;
for hfov_setting = [40 60 90]
    for last_setting = [false true]
        for fwd_ix = 1:numel(ufwd)
            mask = overmask &...
                micro_assumed_fwd_vel == ufwd(fwd_ix) &...
                micro_hfov == hfov_setting;

            % Go through the failures and assign each failure to a rmtv bin
            num_fails = zeros(numel(edges)-1,numel(track_ix));
            num_in_bin = zeros(numel(edges)-1,numel(track_ix)); % How often the relevant velocity appeared

            for tr_ix = 1:numel(track_ix)
                curr_track = track_ix(tr_ix);
                if(last_setting)
                    relevant_results = results.last.all_ticks(mask,curr_track,:,:);
                else
                    relevant_results = results.first.all_ticks(mask,curr_track,:,:);
                end

                for over_ix = 1:sum(mask)
                    for traj_ix = 1:num_traj
                        for im_ix = 1:numel(image_names)
                            fail_time = relevant_results(over_ix,1,im_ix,traj_ix);
                            if(~isnan(fail_time))
                                fail_rmtv = traj_rmtv.rmtv(fail_time,traj_ix);
                                place_in_bin = find(fail_rmtv >= edges(1:end-1) & fail_rmtv < edges(2:end));
                                num_fails(place_in_bin,tr_ix) = num_fails(place_in_bin,tr_ix) + 1;
                                num_in_bin(:,tr_ix) = num_in_bin(:,tr_ix) + (histcounts(traj_rmtv.rmtv(1:fail_time,traj_ix),edges))';
                            end
                        end
                    end
                end
            end

            % Probability of a failure occurring in a bin
            prob_fail = num_fails ./ num_in_bin;

            % Normalise prob fail
            prob_fail = prob_fail ./ repmat(sum(prob_fail,1,'omitnan'),numel(edges)-1,1);
            f=figure(figure_ix);
            figure_ix = figure_ix + 1;
            f.Name = ['Assumed fwd_vel = ' num2str(ufwd(fwd_ix)) ' last=' num2str(last_setting) ' hfov=' num2str(hfov_setting)];
            plot(0.5*(edges(1:end-1)+edges(2:end)),prob_fail,'-x','LineWidth',2)
            legend({'prob','prob_fixed','fac','kf','cg','none'},'box','off','Interpreter','none')
        end
    end
end

if(~exist([figures_out_dir 'FailVel\'],'dir'))
    mkdir([figures_out_dir 'FailVel\']);
end

for k=1:figure_ix - 1
    f=figure(k);
    saveas(f,[figures_out_dir 'FailVel\' f.Name '.fig'],'fig');
end

close all

%%



% Distractor strength, by model
% close all
% overmask = micro_draw_distractors &...
%         ~micro_do_saccade &...
%         micro_do_predictive_turn;
%     
% track_ix = [1 2 3 6 8 10];
% 
% % Define distractor strength as distractor / (target + distractor)
% 
% % Bin the distractor strength
% edges = [0:0.05:0.95 1+eps];
% target_edges= linspace(0,2.5,20);
% ufwd = unique(assumed_fwd_vel(~isnan(assumed_fwd_vel)));
% 
% figure_ix = 1;
% for hfov_setting = [40 60 90]
%     for last_setting = [false true]
%         for fwd_ix = 1:numel(ufwd)
%             mask = overmask &...
%                 micro_assumed_fwd_vel == ufwd(fwd_ix) &...
%                 micro_hfov == hfov_setting;
% 
%             % Go through the failures and assign each failure to a rmtv bin
%             num_fails = zeros(numel(edges)-1,numel(track_ix));
%             num_in_bin = zeros(numel(edges)-1,numel(track_ix)); % How often the relevant distractor strength appeared
%             num_target = zeros(numel(target_edges)-1,numel(track_ix));
%             
%             nands = zeros(numel(track_ix),1);
% 
%             for tr_ix = 1:numel(track_ix)
%                 curr_track = track_ix(tr_ix);
%                 
%                 relevant_target = results.target_intersaccade(mask,curr_track,:,:);
%                 relevant_distractor = results.distractor_intersaccade(mask,curr_track,:,:);
% 
%                 for over_ix = 1:sum(mask)
%                     for traj_ix = 1:num_traj
%                         for im_ix = 1:numel(image_names)
%                             ds = relevant_distractor(over_ix,1,im_ix,traj_ix) /...
%                                 (relevant_target(over_ix,1,im_ix,traj_ix) + relevant_distractor(over_ix,1,im_ix,traj_ix) );
%                             if(~isnan(ds))
%                                 place_in_bin = find(ds >= edges(1:end-1) & ds < edges(2:end));
%                                 num_fails(place_in_bin,tr_ix) = num_fails(place_in_bin,tr_ix) + 1;
%                             else
%                                 nands(tr_ix) = nands(tr_ix) + 1;
%                             end
%                             
%                             place_in_target_bin = find(relevant_target(over_ix,1,im_ix,traj_ix) >= edges(1:end-1) &...
%                                     relevant_target(over_ix,1,im_ix,traj_ix) >= edges(2:end));
%                                 
%                             num_target(place_in_target_bin,tr_ix) = num_target(place_in_target_bin,tr_ix) + 1;
%                         end
%                     end
%                 end
%             end
% 
%             return
%             
%             h=plot(0.5*(edges(1:end-1)+edges(2:end)),num_fails,'LineWidth',2);
%             
%             k_lines={'-','--','-','--','-','--'};
%             k_colours = [0 0 0; 1 0 0;...
%                 0.5 0.5 0.5; 0 1 0;...
%                 0.2 0.2 0.2; 0 0 1];
%             for k=1:numel(h)
%                 h(k).LineStyle = k_lines{k};
%                 h(k).Color = k_colours(k,:);
%             end
%             legend({'p','pf','fac','kf','cg','none'},'box','off')
%             return
% 
% %             f=figure(figure_ix);
% %             figure_ix = figure_ix + 1;
% %             f.Name = ['Assumed fwd_vel = ' num2str(ufwd(fwd_ix)) ' last=' num2str(last_setting) ' hfov=' num2str(hfov_setting)];
% %             plot(0.5*(edges(1:end-1)+edges(2:end)),prob_fail,'-x')
% %             legend({'prob','prob_fixed','fac','kf','cg','none'},'box','off','Interpreter','none')
%         end
%     end
% end
