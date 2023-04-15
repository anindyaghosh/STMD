clear all
close all

which_pc='mecheng';

if(strcmp(which_pc,'mrblack'))
    params = load('E:\PHD\conf2\data\trackerinfo\TrackerParamFile19_09_24.mat');
%     load('E:\PHD\TEST\temp_lost_it_19_09_24.mat');
    load('E:\PHD\TEST\temp_lost_it_19_09_24.mat');
elseif(strcmp(which_pc,'mecheng'))
    params=load('D:\temp\TrackerParamFile19_09_26.mat');
    load('D:\temp\temp_lost_it_19_09_26.mat');
end

gind_range = 1:numel(params.outer_ix);
numgind = numel(gind_range);

% Create masks for the different important settings

% Tracker nums:
% 1 - prob
% 2 - prob_fixed
% 3 - facilitation
% 4 - facilitation_dark

% Get vectors of the settings indexed by gind
hfov=params.outer_var.sets.hfov(params.outer_ix);
image_num=params.outer_var.sets.image_num(params.outer_ix);
tracker_num=params.outer_var.sets.tracker_num(params.outer_ix);

draw_distractors=params.outer_var.sets.draw_distractors(params.outer_ix);
do_predictive_turn=params.outer_var.sets.do_predictive_turn(params.outer_ix);
do_saccades=params.outer_var.sets.do_saccades(params.outer_ix);

% The inner vars are already indexed by gind but keep a consistent
% interface going
saccade_duration = params.inner_var.saccade_duration;
assumed_fwd_vel = params.inner_var.assumed_fwd_vel;

% Create tracker masks
prob_mask = tracker_num == 1;
prob_fixed_mask = tracker_num == 2;
facilitation_mask = tracker_num == 3;
facilitation_dark_mask = tracker_num == 4;

tracker_masks = cell(4,1);
tracker_masks{1} = prob_mask;
tracker_masks{2} = prob_fixed_mask;
tracker_masks{3} = facilitation_mask;
tracker_masks{4} = facilitation_dark_mask;

% mm = max(logs.t_lost_it.first,[],2,'omitnan');
mm = mean(logs.t_lost_it.first,2,'omitnan');

%%
completion=NaN(4,1);

for k=1:4
    temp=logs.t_lost_it.first(tracker_masks{k},:);
    completion(k) = sum(~isnan(temp(:)) / numel(temp));
end

%%


% perf=cell(4,1);
% for k=1:4
%     perf{k} = mm(main_mask & tracker_masks{k});
% end
% 
% X = NaN(numel(perf{1}),4);
% for k=1:4
%     X(:,k) = perf{k};
% end
% boxplot(X)

%%
% Compare the effect of using a different tracker on the same trial
perf=cell(4,1);
prob_inds = gind_range(prob_mask);

all_median.d1 = NaN(numel(prob_inds),1);
all_median.d2 = all_median.d1;
all_median.d3 = all_median.d1;

all_max = all_median;
all_mean = all_median;

for k=1:numel(prob_inds)
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
    
    prob_match = match_mask & prob_mask;
    prob_fixed_match = match_mask & prob_fixed_mask;
    fac_match = match_mask & facilitation_mask;
    fac_dark_match = match_mask & facilitation_dark_mask;

    prob_set = logs.t_lost_it.first(prob_match,:);
    prob_fixed_set  = logs.t_lost_it.first(prob_fixed_match,:);
    fac_set = logs.t_lost_it.first(fac_match,:);
    fac_dark_set = logs.t_lost_it.first(fac_dark_match,:);
    
    % Work out some metric on the delta
    all_median.d1(k) = median(prob_set - prob_fixed_set,'omitnan');
    all_median.d2(k) = median(prob_set - fac_set,'omitnan');
    all_median.d3(k) = median(prob_set - fac_dark_set,'omitnan');
    
    all_mean.d1(k) = mean(prob_set - prob_fixed_set,'omitnan');
    all_mean.d2(k) = mean(prob_set - fac_set,'omitnan');
    all_mean.d3(k) = mean(prob_set - fac_dark_set,'omitnan');
    
    all_max.d1(k) = max(prob_set - prob_fixed_set,[],'omitnan');
    all_max.d2(k) = max(prob_set - fac_set,[],'omitnan');
    all_max.d3(k) = max(prob_set - fac_dark_set,[],'omitnan');
    
end

mini_distractor_mask = draw_distractors(prob_inds);
mini_saccade_mask = do_saccades(prob_inds);
mini_predictive_mask = do_predictive_turn(prob_inds);

hfov_range=[40 60 90];
for hfix=1:3
    mini_hfov_mask = hfov(prob_inds) == hfov_range(hfix);

%     gridmasks=cell(4);
%     gridmasks{1} = mini_distractor_mask & mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
%     gridmasks{2} = mini_distractor_mask & ~mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
%     gridmasks{3} = ~mini_distractor_mask & mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
%     gridmasks{4} = ~mini_distractor_mask & ~mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;

    f=figure(hfix);
    f.Name=num2str(['hfov= ' hfov_range(hfix)]);
    title_strings={'distract & saccade','distract','saccade','nothing'};
%     for k=1:4 
%         subplot(2,2,k)
%         h=plot(sort([all_median.d1(gridmasks{k}) all_median.d2(gridmasks{k}) all_median.d3(gridmasks{k})]));
%         h(1).LineStyle='--';
%         h(3).LineStyle=':';
%         h(3).Color=[0 0 0];
%         title(title_strings{k})
%     end
%     legend({'prob_fixed','fac','fac_dark'},'Interpreter','none')

    kmask{1} = draw_distractors & do_saccades & do_predictive_turn & hfov == 40;
    kmask{2} = ~draw_distractors & do_saccades & do_predictive_turn & hfov == 40;
    kmask{3} = draw_distractors & ~do_saccades & do_predictive_turn & hfov == 40;
    kmask{4} = ~draw_distractors & ~do_saccades & do_predictive_turn & hfov == 40;

    for k=1:4
        g=subplot(2,2,k);
        all_traj_mask = ~isnan(logs.t_lost_it.first(kmask{k},:));
        [v,h]=size(all_traj_mask);
        all_traj_mask = sum(all_traj_mask,1) == v;
        I_last_all_traj = find(all_traj_mask,1,'first');
        
        X=NaN(numgind/4,4);
        for p=1:4
            temp = mean(logs.t_lost_it.first(kmask{k} & tracker_masks{p},1:I_last_all_traj),2,'omitnan');
            X(1:numel(temp),p) = temp;
        end
        boxplot(X/1000)
        set(g,'XTickLabels',{'prob','prob_fixed','fac','fac_dark'},'XTickLabelRotation',90)
    end

end

% testmasks=cell(3,1)
% testmask{1} = params.outer_var.sets.hfov(prob_inds) == 40 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% testmask{2} = params.outer_var.sets.hfov(prob_inds) == 60 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% testmask{3} = params.outer_var.sets.hfov(prob_inds) == 90 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% 
% plot(sort([d3(testmask{1}) d3(testmask{2}) d3(testmask{3})]))
return

%%
% Plot effect of saccade duration
usac=unique(params.inner_var.saccade_duration(~isnan(params.inner_var.saccade_duration)));
sac_masks=cell(1+numel(usac),1);
sac_masks{1} = ~params.outer_var.sets.do_saccades(params.outer_ix);
for k=1:numel(usac)
    sac_masks{k+1} = ~isnan(params.inner_var.saccade_duration) & params.inner_var.saccade_duration == usac(k);
end

overmask = mini_predictive_mask &...
    ~mini_distractor_mask;

prob_dist = cell(numel(sac_masks),1);
prob_fixed_dist = prob_dist;
fac_dist = prob_dist;
fac_dark_dist = prob_dist;

fac_all=cell(4,1);

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

f=figure(1)
subplot(2,1,1)
X=[prob_dist{1} prob_dist{2} prob_dist{3} prob_dist{4}];
boxplot(X)
set(gca,'XTickLabels',[0;usac])
subplot(2,1,2)
X=[fac_dist{1} fac_dist{2} fac_dist{3} fac_dist{4}];
boxplot(X)
set(gca,'XTickLabels',[0;usac])

%%
% fac_mean = mean(logs.t_lost_it.first(facilitation_mask,:),2,'omitnan');
% prob_mean = mean(logs.t_lost_it.first(prob_mask,:),2,'omitnan');

% Find indices matching every prob_tracker result
close all
I_prob = gind_range(tracker_masks{1});

I_probfixed = NaN(size(I_prob));
I_fac = I_probfixed;
I_facdark = I_probfixed;
I_all_traj = I_probfixed;

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

    if(sum(match_mask & tracker_masks{2}) == 0)
        return
    end
    I_probfixed(ix) = gind_range(match_mask & tracker_masks{2});
    I_fac(ix) = gind_range(match_mask & tracker_masks{3});
    I_facdark(ix) = gind_range(match_mask & tracker_masks{4});
    
    
    % Work out how many trajectories are available at minimum for the index
    temp = find(~isnan(logs.t_lost_it.first(I_prob(ix),:)) &...
        ~isnan(logs.t_lost_it.first(I_probfixed(ix),:)) &...
        ~isnan(logs.t_lost_it.first(I_fac(ix),:)) &...
        ~isnan(logs.t_lost_it.first(I_facdark(ix),:)),1,'last');
    if(~isempty(temp))
        I_all_traj(ix) = temp;
    end
end

results=NaN(numel(I_prob),4);
for ix=1:numel(I_prob)
    if(~isnan(I_all_traj(ix)))
        results(ix,1) = mean(logs.t_lost_it.first(gind_range(I_prob(ix)),1:I_all_traj(ix)));
        results(ix,2) = mean(logs.t_lost_it.first(gind_range(I_probfixed(ix)),1:I_all_traj(ix)));
        results(ix,3) = mean(logs.t_lost_it.first(gind_range(I_fac(ix)),1:I_all_traj(ix)));
        results(ix,4) = mean(logs.t_lost_it.first(gind_range(I_facdark(ix)),1:I_all_traj(ix)));
    end
end
results=results/1000; % Conversion to seconds

% Apply masks to the results
mini_distract_mask = draw_distractors(I_prob);
mini_predictive_turn_mask = do_predictive_turn(I_prob);
mini_do_saccades_mask = do_saccades(I_prob);
mini_image_num_mask = image_num(I_prob);

% Effect of distractors & doing saccades
f=figure(1);
f.Name='Distractors & saccades';
plotmasks = cell(4,1);
plotmasks{1} = mini_distract_mask & mini_do_saccades_mask & mini_predictive_turn_mask;
plotmasks{2} = ~mini_distract_mask & mini_do_saccades_mask & mini_predictive_turn_mask;
plotmasks{3} = mini_distract_mask & ~mini_do_saccades_mask & mini_predictive_turn_mask;
plotmasks{4} = ~mini_distract_mask & ~mini_do_saccades_mask & mini_predictive_turn_mask;

title_strings={'Distract+saccade','Saccade','Distract','Neither'};
for k=1:4
    g=subplot(2,2,k);
    boxplot(results(plotmasks{k},:))
    g.YLim = [0 max(results(:))];
    title(title_strings{k})
    set(g,'XTickLabels',{'prob','prob_fixed','fac','fac_dark'},'XTickLabelRotation',90)
    ylabel('Time to failure (s)')
end
g.FontName='Times New Roman';
g.FontSize= 10;

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
    X=NaN(sum(plotmasks{1}),4);
    
    for p=1:4
        sorted_this_res = sort(results(plotmasks{k},p));
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
    X=NaN(sum(plotmasks{1}),4);
    
    for p=1:4
        sorted_this_res = sort(results(plotmasks{k},p));
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

