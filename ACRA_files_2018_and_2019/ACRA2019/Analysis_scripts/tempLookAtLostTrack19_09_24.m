clear all
close all

which_pc='mecheng';

if(strcmp(which_pc,'mrblack'))
    params = load('E:\PHD\conf2\data\trackerinfo\TrackerParamFile19_09_24.mat');
%     load('E:\PHD\TEST\temp_lost_it_19_09_24.mat');
    load('E:\PHD\TEST\temp_lost_it_19_09_24.mat');
elseif(strcmp(which_pc,'mecheng'))
    params=load('D:\temp\TrackerParamFile19_09_24.mat');
    load('D:\temp\temp_lost_it_19_09_25.mat');
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

d1=NaN(numel(prob_inds),1);
d2=d1;
d3=d1;
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
    d1(k) = median(prob_set - prob_fixed_set,'omitnan');
    d2(k) = median(prob_set - fac_set,'omitnan');
    d3(k) = median(prob_set - fac_dark_set,'omitnan');
end



mini_distractor_mask = draw_distractors(prob_inds);
mini_saccade_mask = do_saccades(prob_inds);
mini_predictive_mask = do_predictive_turn(prob_inds);

hfov_range=[40 60 90];
for hfix=1:3
    mini_hfov_mask = hfov(prob_inds) == hfov_range(hfix);

    gridmasks=cell(4);
    gridmasks{1} = mini_distractor_mask & mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
    gridmasks{2} = mini_distractor_mask & ~mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
    gridmasks{3} = ~mini_distractor_mask & mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;
    gridmasks{4} = ~mini_distractor_mask & ~mini_saccade_mask & mini_predictive_mask & mini_hfov_mask;

    f=figure(hfix);
    f.Name=num2str(['hfov= ' hfov_range(hfix)]);
    title_strings={'distract & saccade','distract','saccade','nothing'};
    for k=1:4
        subplot(2,2,k)
        h=plot(sort([d1(gridmasks{k}) d2(gridmasks{k}) d3(gridmasks{k})]));
        h(1).LineStyle='--';
        h(3).LineStyle=':';
        h(3).Color=[0 0 0];
        title(title_strings{k})
    end
    legend({'prob_fixed','fac','fac_dark'})
end

% testmasks=cell(3,1)
% testmask{1} = params.outer_var.sets.hfov(prob_inds) == 40 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% testmask{2} = params.outer_var.sets.hfov(prob_inds) == 60 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% testmask{3} = params.outer_var.sets.hfov(prob_inds) == 90 & ~params.outer_var.sets.do_predictive_turn(prob_inds);
% 
% plot(sort([d3(testmask{1}) d3(testmask{2}) d3(testmask{3})]))


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