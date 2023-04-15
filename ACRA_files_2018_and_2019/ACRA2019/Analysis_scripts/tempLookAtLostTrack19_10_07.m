clear all
close all

which_pc='mrblack';

if(strcmp(which_pc,'mrblack'))
    params = load('E:\PHD\TEST\TrackerParamFile19_10_03.mat');
%     load('E:\PHD\TEST\temp_lost_it_19_09_24.mat');
    load('E:\PHD\TEST\temp_lost_it_19_10_07.mat');
    figures_out_dir = 'E:\PHD\conf2\autofigures\';
elseif(strcmp(which_pc,'mecheng'))
    params=load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');
    load('D:\temp\temp_lost_it_19_10_06.mat');
end

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
%%
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
        temp = temp & ~isnan(logs.t_lost_it.first(I_to_use(pinner),:));
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
results.cause = results.raw;
results.rawlast = results.raw;
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
            
            results.rawlast(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.last(I_get,1:I_all_traj_singleimage(ix));
            
            results.cause(ix,pinner,img_ix,1:I_all_traj_singleimage(ix))=...
                logs.t_lost_it.caused_by_saccade(I_get,1:I_all_traj_singleimage(ix));
            
            
        end
    end
end

results.mean = results.mean - spin_ticks; % Removal of spin_ticks
results.median = results.median- spin_ticks;
results.raw = results.raw - spin_ticks;
results.rawlast = results.rawlast - spin_ticks;

results.mean=results.mean/1000; % Conversion to seconds
results.median=results.median/1000;
results.raw = results.raw / 1000;
results.rawlast = results.rawlast / 1000;

% Collapse results.raw across images
results.all = results.raw(:,:,:);
results.all_last = results.rawlast(:,:,:);
results.cause = results.cause(:,:,:);

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
track_ix=[10 8 6 4 1];  % Dark selective variants
num_trackix=numel(track_ix);
track_labels=['nckfp']; % These must be single chars

% label_length = 0;
% for k=1:numel(track_labels)
%     label_length = max(label_length,numel(track_labels{k}));
% end

micro_assumed_fwd_vel = assumed_fwd_vel(I_prob_single_image);
micro_saccade_duration = saccade_duration(I_prob_single_image);
micro_hfov = hfov(I_prob_single_image);
micro_do_predictive_turn = do_predictive_turn(I_prob_single_image);
micro_do_saccade = do_saccades(I_prob_single_image);
micro_draw_distractors = draw_distractors(I_prob_single_image);

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

[~,~,num_traj_in_all] = size(results.all);
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
                A(:,3*(k-1)+p) = results.all_last(mask,track_ix(k),:);
            else
                A(:,3*(k-1)+p) = results.all(mask,track_ix(k),:);
            end
            
        end
    end
    G={label_mat(:),hfov_mat(:)};
    boxplot(A(:),G,'factorgap',10,'colorgroup',repmat([1 2 3],1,num_trackix),'plotstyle','compact')
  
    f=figure(figure_ix);
    figure_ix = figure_ix+1;
    f.Name = ['histo fwd vel = ' num2str(ufwd(fwd_ix)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
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
    
%     k_colours=[1 0 0; %none
%         0 1 0 %cg
%         0 0 1;
%         0.5 0 0.5;
%         0 0 0];
    k_lines = {'-','--','-.',':','-'};
    for p = 1:3
        f=figure(figure_ix);
        figure_ix = figure_ix+1;
        f.Name = ['Cumulative failure fwd vel = ' num2str(ufwd(fwd_ix)) ' hfov=' num2str(uhfov(p)) ' predictive=' num2str(predictive_setting) ' last=' num2str(last_setting)];
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
        g.XLim=[0 31];
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
hfov_mat = repmat([40 60 90],1,num_trackix);
hfov_mat = repmat(hfov_mat,1500,1);

label_mat = repelem(track_labels,1,numel(uhfov));
label_mat = repmat(label_mat,1500,1);

[~,~,num_traj_in_all] = size(results.all);
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
                        A(:,3*(k-1)+p) = results.all_last(mask,track_ix(k),:);
                    else
                        A(:,3*(k-1)+p) = results.all(mask,track_ix(k),:);
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
                g.XLim=[0 31];
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
        temp = results.cause(mask,track_ix(pinner),:);
        A(hfov_ix,pinner) = sum(temp(:),'omitnan')/sum(~isnan(temp(:)));
    end
end

f=figure(1);
f.Name = 'Caused by saccade';
bar(A);



return
%%
% Effect of distractors & doing saccades
f=figure(1);
f.Name='Distractors & saccades';
plotmasks = cell(4,1);
plotmasks{1} = mini_distract_mask & mini_do_saccades_mask & mini_predictive_turn_mask & mini_hfov == 90;
plotmasks{2} = ~mini_distract_mask & mini_do_saccades_mask & mini_predictive_turn_mask & mini_hfov == 90;
plotmasks{3} = mini_distract_mask & ~mini_do_saccades_mask & mini_predictive_turn_mask & mini_hfov == 90;
plotmasks{4} = ~mini_distract_mask & ~mini_do_saccades_mask & mini_predictive_turn_mask & mini_hfov == 90;

title_strings={'Distract+saccade','Saccade','Distract','Neither'};
for k=1:4
    g=subplot(2,2,k);
    boxplot(results.mean(plotmasks{k},:))
    g.YLim = [0 max(results.mean(:))];
    title(title_strings{k})
    set(g,'XTickLabels',{'prob','prob_fixed','fac','fac_dark','kf','kf_dark','cg','cg_dark','none','none_dark'},'XTickLabelRotation',90)
    ylabel('Time to failure (s)')
end
g.FontName='Times New Roman';
g.FontSize= 10;

%%
% Boxplots for different field of view

%%
% Plot effect of assumed fwd vel
f=figure(2);
f.Name='Assumed forward vel';
ufwd = unique(assumed_fwd_vel(~isnan(assumed_fwd_vel)));
plotmasks = cell(numel(ufwd),1);
for k=1:numel(ufwd)
    plotmasks{k} = assumed_fwd_vel(I_prob) == ufwd(k);
end

upper_vals = NaN(numel(ufwd),4);
lower_vals = NaN(numel(ufwd),4);
median_vals = NaN(numel(ufwd),4);

for k=1:numel(ufwd)
%     X=NaN(sum(plotmasks{1}),4);
    
    for p=1:4
        sorted_this_res = sort(results.mean(plotmasks{k},p));
        num_notnan = sum(~isnan(sorted_this_res));
        
        median_vals(k,p) = median(sorted_this_res,'omitnan');
        upper_vals(k,p) = sorted_this_res(floor(0.75*num_notnan));
        lower_vals(k,p) = sorted_this_res(floor(0.25*num_notnan));
    end
end

spacing=0.1;
x_base = repmat(ufwd,1,4) + repmat([-1.5*spacing -0.5*spacing 0.5*spacing 1.5*spacing],numel(ufwd),1);
colour_set = [1 0 0; 0 1 0; 0 0 1; 0 0 0];

for k=1:4
    plot(x_base(:,k),median_vals(:,k),'-','LineWidth',2,'Color',colour_set(k,:))
    hold on
end
for k=1:4
    plot(repmat(x_base(:,k)',2,1),[lower_vals(:,k) upper_vals(:,k)]','Color',colour_set(k,:));
end
legend({'prob','prob_fixed','fac','fac_dark'},'box','off','interpreter','none')
hold off

xlabel('Forward velocity (°/s)')
ylabel('Time to failure (s)')
g=findall(f.Children,'Type','Axes');
set(g,'XTick',ufwd)
set(g,'FontName','Times New Roman','FontSize',10)

% Effect of saccade duration
f=figure(3);
f.Name='Saccade duration';
ufwd = unique(saccade_duration(~isnan(saccade_duration)));
plotmasks = cell(numel(ufwd),1);
for k=1:numel(ufwd)
    plotmasks{k} = saccade_duration(I_prob) == ufwd(k);
end

upper_vals = NaN(numel(ufwd),4);
lower_vals = NaN(numel(ufwd),4);
median_vals = NaN(numel(ufwd),4);

for k=1:numel(ufwd)
%     X=NaN(sum(plotmasks{1}),4);
    
    for p=1:4
        sorted_this_res = sort(results.mean(plotmasks{k},p));
        num_notnan = sum(~isnan(sorted_this_res));
        
        median_vals(k,p) = median(sorted_this_res,'omitnan');
        upper_vals(k,p) = sorted_this_res(floor(0.75*num_notnan));
        lower_vals(k,p) = sorted_this_res(floor(0.25*num_notnan));
    end
end

spacing=0.001;
x_base = repmat(ufwd,1,4) + repmat([-1.5*spacing -0.5*spacing 0.5*spacing 1.5*spacing],numel(ufwd),1);
colour_set = [1 0 0; 0 1 0; 0 0 1; 0 0 0];

for k=1:4
    plot(x_base(:,k),median_vals(:,k),'-','LineWidth',2,'Color',colour_set(k,:))
    hold on
end
for k=1:4
    plot(repmat(x_base(:,k)',2,1),[lower_vals(:,k) upper_vals(:,k)]','Color',colour_set(k,:));
end
legend({'prob','prob_fixed','fac','fac_dark'},'box','off','interpreter','none')
hold off

xlabel('Saccade duration (ms)')
ylabel('Time to failure (s)')
g=findall(f.Children,'Type','Axes');
set(g,'XTick',ufwd)
set(g,'FontName','Times New Roman','FontSize',10)

%%
% Effect of image
f=figure(4);
f.Name = 'Image';
ufwd = unique(image_num(~isnan(image_num)));
plotmasks = cell(numel(ufwd),1);
for k=1:numel(ufwd)
    plotmasks{k} = image_num(I_prob) == ufwd(k);
end

upper_vals = NaN(numel(ufwd),4);
lower_vals = NaN(numel(ufwd),4);
median_vals = NaN(numel(ufwd),4);

for k=1:numel(ufwd)
%     X=NaN(sum(plotmasks{1}),4);
    
    for p=1:4
        sorted_this_res = sort(results.mean(plotmasks{k},p));
        num_notnan = sum(~isnan(sorted_this_res));
        
        median_vals(k,p) = median(sorted_this_res,'omitnan');
        upper_vals(k,p) = sorted_this_res(floor(0.75*num_notnan));
        lower_vals(k,p) = sorted_this_res(floor(0.25*num_notnan));
    end
end

spacing=0.001;
x_base = repmat(ufwd,1,4) + repmat([-1.5*spacing -0.5*spacing 0.5*spacing 1.5*spacing],numel(ufwd),1);
colour_set = [1 0 0; 0 1 0; 0 0 1; 0 0 0];

for k=1:4
    plot(x_base(:,k),median_vals(:,k),'-','LineWidth',2,'Color',colour_set(k,:))
    hold on
end
for k=1:4
    plot(repmat(x_base(:,k)',2,1),[lower_vals(:,k) upper_vals(:,k)]','Color',colour_set(k,:));
end
legend({'prob','prob_fixed','fac','fac_dark'},'box','off','interpreter','none')
hold off

ylabel('Time to failure (s)')
g=findall(f.Children,'Type','Axes');
g.XTickLabels = image_names;
set(g,'XTick',ufwd)
set(g,'FontName','Times New Roman','FontSize',10)




return
%%
% Compare the effect of using a different tracker on the same trial
perf=cell(num_trackers,1);
prob_inds = gind_range(prob_mask);

% compare_inds=[1 2
%     1 3
%     1 4
%     1 5
%     1 6
%     2 3
%     2 4
%     2 5
%     2 6
%     3 4 
%     3 5
%     3 6
%     4 5
%     4 6
%     5 6];

compare_inds = [1 4 ; 1 6; 1 8; 1 10; 4 6; 4 8; 4 10; 6 8; 6 10; 8 10];

[num_compare,~]=size(compare_inds);

all_median= NaN(numel(prob_inds),numel(compare_inds));
all_max = all_median;
all_mean = all_median;

for p=1:num_compare
    for k=1:numel(prob_inds)
        lhs_inds = gind_range(tracker_masks{compare_inds(p,1)});
        rhs_inds = gind_range(tracker_masks{compare_inds(p,2)});

        curr_ind = prob_inds(k);

        % Get mask for matching settings
        match_mask = hfov == hfov(curr_ind) &...
            do_saccades == do_saccades(curr_ind) &...
            draw_distractors == draw_distractors(curr_ind) &...
            do_predictive_turn == do_predictive_turn(curr_ind) &...
            image_num == image_num(curr_ind);

        if(draw_distractors(curr_ind))
            match_mask = match_mask & ...
                assumed_fwd_vel == assumed_fwd_vel(curr_ind);
        end

        if(do_saccades(curr_ind))
            match_mask = match_mask & ...
                saccade_duration == saccade_duration(curr_ind);
        end

        % These masks should have only one true entry
        
%         for pinner=1:10
%             this_mask = match_mask & tracker_masks{pinner};
%             this_set = logs.t_lost_it.first(this_mask,:);
%             all_median.raw(k,pinner) = median(this_set,2,'omitnan');
%         end
        
        lhs_mask = match_mask & tracker_masks{compare_inds(p,1)};
        rhs_mask = match_mask & tracker_masks{compare_inds(p,2)};

        if(sum(lhs_mask) > 0 && sum(rhs_mask) > 0)
            lhs_set = logs.t_lost_it.first(lhs_mask,:);
            rhs_set = logs.t_lost_it.first(rhs_mask,:);

            % Work out some metric on the delta
            all_median(k,p) = median(lhs_set - rhs_set,'omitnan');
            all_mean(k,p) = mean(lhs_set - rhs_set,'omitnan');
            all_max(k,p) = max(lhs_set - rhs_set,[],'omitnan');
        end
    end
end

figure(1);
plot(sort(all_median(:,1:4)))
legend({'fac_dark','kf_dark','cg_dark','none_dark'},'Interpreter','none','box','off','FontName','Times New Roman','FontSize',10)

%%
%
res_distract = draw_distractors(prob_inds);
res_saccade = do_saccades(prob_inds);

X=all_median(res_distract,:);
boxplot(X)

return
%%
% For each hfov, compare between distractor/saccade settings
hfov_range=[40 60 90];
for hfix=1:3
    f=figure(hfix);
    f.Name=['hfov= ' num2str(hfov_range(hfix))];
    title_strings={'distract & saccade','distract','saccade','nothing'};

    kmask{1} = draw_distractors & do_saccades & ~do_predictive_turn & hfov == hfov_range(hfix);% & (~isnan(saccade_duration)) & saccade_duration == 0.05;
    kmask{2} = ~draw_distractors & do_saccades & ~do_predictive_turn & hfov == hfov_range(hfix);%  & (~isnan(saccade_duration)) & saccade_duration == 0.05;
    kmask{3} = draw_distractors & ~do_saccades & ~do_predictive_turn & hfov == hfov_range(hfix);
    kmask{4} = ~draw_distractors & ~do_saccades & ~do_predictive_turn & hfov == hfov_range(hfix);

    for k=1:4
        g=subplot(2,2,k);
        % For the mask, work out the number of trajectories which all
        % models have data for
        all_traj_mask = ~isnan(logs.t_lost_it.first(kmask{k},:));
        [v,h]=size(all_traj_mask);
        all_traj_mask = sum(all_traj_mask,1) == v;
        I_last_all_traj = find(all_traj_mask,1,'first');
        
        X=NaN(numgind/num_trackers,num_trackers);
        for p=1:num_trackers
            temp = median(logs.t_lost_it.first(kmask{k} & tracker_masks{p},1:I_last_all_traj),2,'omitnan');
            X(1:numel(temp),p) = temp;
        end
        boxplot((X-spin_ticks)/1000)
        ylim([0 30])
        set(g,'XTickLabels',{'prob','prob_fixed','fac','fac_dark','kf','kf_dark'},'XTickLabelRotation',90)
        title(title_strings{k})
    end
end

%%
% Plot distributions for individual settings
close all
y_hfov = 60;
y_saccade_duration = 0.05;
y_assumed_fwd_vel = 1;
y_do_predictive_turn = true;
y_draw_distractors = false;
y_do_saccades = true;
y_image_num = 1;

match_mask = hfov == y_hfov & ...
    draw_distractors == y_draw_distractors &...
    do_saccades == y_do_saccades &...
    do_predictive_turn == y_do_predictive_turn &...
    image_num == y_image_num;
    
if(y_draw_distractors)
    match_mask = match_mask & assumed_fwd_vel == y_assumed_fwd_vel;
end
if(y_do_saccades)
    match_mask = match_mask & saccade_duration == y_saccade_duration;
end
find(match_mask)

X=NaN(100,6);
for k=1:6
    X(:,k) = logs.t_lost_it.first(match_mask & tracker_masks{k},:);
end

boxplot(X)
return

%%
% Plot effect of saccade duration
close all

usac=unique(params.inner_var.saccade_duration(~isnan(params.inner_var.saccade_duration)));
sac_masks=cell(1+numel(usac),1);
sac_masks{1} = ~params.outer_var.sets.do_saccades(params.outer_ix);
for k=1:numel(usac)
    sac_masks{k+1} = ~isnan(params.inner_var.saccade_duration) & params.inner_var.saccade_duration == usac(k);
end

overmask = do_predictive_turn &...
    ~draw_distractors;

prob_dist = cell(numel(sac_masks),1);
prob_fixed_dist = prob_dist;
fac_dist = prob_dist;
fac_dark_dist = prob_dist;

fac_all=cell(4,1);

% For each saccade duration, work out the distribution of results for each
% tracker
for k=1:numel(sac_masks)
    prob_set = logs.t_lost_it.first(overmask & sac_masks{k} & prob_mask,:);
    prob_fixed_set = logs.t_lost_it.first(overmask & sac_masks{k} & prob_fixed_mask,:);
    fac_set = logs.t_lost_it.first(overmask & sac_masks{k} & facilitation_mask,:);
    fac_dark_set = logs.t_lost_it.first(overmask & sac_masks{k} & facilitation_dark_mask,:);
    
    fac_all{k} = fac_set;
    
    prob_dist{k} = max(prob_set,[],2,'omitnan');
    prob_fixed_dist{k} = max(prob_fixed_set,[],2,'omitnan');
    fac_dist{k} = max(fac_set,[],2,'omitnan');
    fac_dark_dist{k} = max(fac_dark_set,[],2,'omitnan');
end

f=figure(1);
f.Name = 'Saccade duration';
subplot(2,1,1)
X=[prob_dist{1} prob_dist{2} prob_dist{3} prob_dist{4}];
boxplot(X)
set(gca,'XTickLabels',[0;usac])
subplot(2,1,2)
X=[fac_dist{1} fac_dist{2} fac_dist{3} fac_dist{4}];
boxplot(X)
set(gca,'XTickLabels',[0;usac])


