clear all
close all

which_pc='mecheng';

if(strcmp(which_pc,'mecheng'))
    paramfile = load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_09_22.mat');
    load_dir = 'D:\temp\TrackerFull19_09_22\';
    save_dir = 'D:\temp\';
elseif(strcmp(which_pc,'phoenix'))
    paramfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/TrackerParamFile19_09_22.mat');
end

% paramfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/TrackerParamFile19_09_22.mat');
% Load everything
gind_range = 1:5760;

gind_to_load = gind_range;

trial_length = 30250;

numgind=numel(gind_to_load);
numtraj = 100;
Ts=0.001;

num_saccades=NaN(numgind,numtraj);
t_saccade_onscreen = num_saccades;
mean_vel = num_saccades;
median_vel = num_saccades;
t_lost_it.first = num_saccades;
t_lost_it.last = num_saccades;

% Target distractor ratio pre-loss.
% Determined by looking at the highest distractor response in the 100ms preceding
% a bad saccade and comparing that to the highest target response in that period
t_lost_it.target_distractor_ratio = num_saccades;

target_intersaccade.mean = num_saccades;
target_intersaccade.median = num_saccades;
target_intersaccade.max = num_saccades;

distractor_intersaccade = target_intersaccade;
target_100ms = target_intersaccade;
distractor_100ms = target_intersaccade;

t_lost_it.caused_by_saccade = false(numgind,numtraj);

tic
for g_ix=1:numel(gind_range)
    if(mod(g_ix,10) == 0); disp([num2str(g_ix) '/' num2str(numel(gind_to_load)) ' ' num2str(toc)]); end
    g = gind_to_load(g_ix);
    fname = [load_dir 'trackres_' num2str(g) '.mat'];
    
    if(exist(fname,'file'))
        a=load(fname);
        do_saccades = a.outer_log.settings.do_saccades;
        hfov= a.outer_log.settings.hfov;
        vfov= a.outer_log.settings.vfov;
        for traj_ix = 1:numtraj
            if(a.outer_log.completed(traj_ix))
                % Mask for whether the target is onscreen or not
                test_mask = a.outer_log.data{traj_ix}.t_th_rel_history > hfov/2 |...
                    a.outer_log.data{traj_ix}.t_th_rel_history < -hfov/2;
                
                % Work out when tracking was lost first
                temp = find(test_mask,1,'first');
                if(~isempty(temp))
                    t_lost_it.first(g,traj_ix)=temp;
                else
                    t_lost_it.first(g,traj_ix) = numel(a.outer_log.data{traj_ix}.t_th_rel_history);
                end
                
                % Work out when tracking was lost last
                temp = find(test_mask,1,'last');
                if(~isempty(temp))
                    t_lost_it.last(g,traj_ix) = temp;
                else
                    t_lost_it.last(g,traj_ix) = numel(a.outer_log.data{traj_ix}.t_th_rel_history);
                end
                if(do_saccades)
                    % Work out the number of saccades by locating rising edges
                    % in saccade history
                    temp = find(a.outer_log.data{traj_ix}.saccading_history(2:end) == true &...
                        a.outer_log.data{traj_ix}.saccading_history(1:end-1) == false);
                    if(~isempty(temp))
                        num_saccades(g,traj_ix) = numel(temp);
                    end
                else
                    % Otherwise have to work out when the turns took place
                    % by looking for edges in the observer angle coming
                    % after the initial spin
                    temp = find((a.outer_log.data{traj_ix}.obs_th_pos_history(252:end) ~=...
                        a.outer_log.data{traj_ix}.obs_th_pos_history(251:end-1)) &...
                        ~isnan(a.outer_log.data{traj_ix}.obs_th_pos_history(252:end)) &...
                        ~isnan(a.outer_log.data{traj_ix}.obs_th_pos_history(251:end-1)));
                    if(~isempty(temp))
                        num_saccades(g,traj_ix) = numel(temp);
                    end
                end
               
                % Find out the time of the end of the last saccade that
                % finished with the target being on screen
                t_saccades_initiated=[];
                edgemask=[];
                temp=[];
                t_end_analysis=[];
                analysis_indices=[];
                
                if(do_saccades)
                    % Find end of last successful saccade before the target
                    % was first lost

                    edgemask_lastsuccess = a.outer_log.data{traj_ix}.saccading_history == true &...
                        a.outer_log.data{traj_ix}.t_th_rel_history < hfov/2 &...
                        a.outer_log.data{traj_ix}.t_th_rel_history > -hfov/2;
                    edgemask_lastsuccess(t_lost_it.first(g,traj_ix):end)=false;
                    
                    temp = find(edgemask_lastsuccess,1,'last');
                    if(~isempty(temp))
                        t_saccade_onscreen(g,traj_ix) = temp;
                    end
                    
                    % Find times at which saccades initiated
                    edgemask = a.outer_log.data{traj_ix}.saccading_history(2:end) == 1 &...
                        a.outer_log.data{traj_ix}.saccading_history(1:end-1) == 0;
                    edgemask = [false; edgemask(:)];
                    
                    temp=find(edgemask);
                    if(~isempty(temp))
                        t_saccades_initiated = temp;
                    end
                    
                    % Determine whether the target was first loss due to a
                    % bad saccade or a failure to saccade.
                    if(a.outer_log.data{traj_ix}.saccading_history(t_lost_it.first(g,traj_ix)) == 1)
                        % A saccade was occurring when the target was lost,
                        % so blame it on the saccade
                        t_lost_it.caused_by_saccade(g,traj_ix) = true;
                    else
                        % Otherwise the loss was due to a failure to
                        % turn
                        t_lost_it.caused_by_saccade(g,traj_ix) = false;
                    end
                else
                    % Find end of last successful saccade before the
                    % target was first lost
                    edgemask = a.outer_log.data{traj_ix}.obs_th_pos_history(2:end) ~=...
                        a.outer_log.data{traj_ix}.obs_th_pos_history(1:end-1);
                    edgemask = [false;edgemask(:)]; % Appending one false to account for the offset in defining edgemask
                    
                    edgemask_lastsuccess = edgemask;
                    edgemask_lastsuccess(t_lost_it.first(g,traj_ix)+1:end) = false; % Wipe out saccades after the first loss of the target
                    
                    temp = find(edgemask_lastsuccess &...
                        a.outer_log.data{traj_ix}.t_th_rel_history < hfov/2 &...
                        a.outer_log.data{traj_ix}.t_th_rel_history > -hfov/2,1,'last'); % The last turn that finished with the target being onscreen
                    if(~isempty(temp))
                        t_saccade_onscreen(g,traj_ix)=temp;
                    end
                    
                    % Find times at which saccades initiated
                    temp = find(edgemask);
                    if(~isempty(temp))
                        t_saccades_initiated = temp;
                    end
                    
                    % Determine whether the target was first lost due to a
                    % bad saccade or a failure to saccade.
                    if(edgemask(t_lost_it.first(g,traj_ix)))
                        % The loss of the target coincides with a turn, so
                        % the turn is to blame
                        t_lost_it.caused_by_saccade(g,traj_ix) = true;
                    else
                        t_lost_it.caused_by_saccade(g,traj_ix) = false;
                    end
                end
                
                % Find out the mean and median velocity from the last
                % successful saccade to either the next unsuccessful saccade or
                % the loss of the target
                
                % If there was another (unsuccessful) saccade after the last
                % successful one, do the analysis on the intervening time
                temp = find(t_saccades_initiated > t_saccade_onscreen(g,traj_ix),1,'first'); % Next saccade after last successful one
                if(~isempty(temp))
                    % temp contains the time at which an unsuccessful saccade was
                    % initiated
                    t_end_analysis=t_saccades_initiated(temp);
                else
                    % Find the final non-NaN timestep
                    t_end_analysis = find(~isnan(a.outer_log.data{traj_ix}.t_th_rel_history),1,'last');
                end
                
                % In the analysis interval (following the last successful
                % saccade), calculate mean and median target velocity
%                 disp(['start: ' num2str(t_saccade_onscreen(g,traj_ix)) ' end: ' num2str(t_end_analysis)])
                if(~isnan(t_saccade_onscreen(g,traj_ix)))
                    analysis_indices = t_saccade_onscreen(g,traj_ix):t_end_analysis;
                    if(~isempty(analysis_indices))
                        % Need ensure that the clamping of t_th_rel to
                        % [-180,180] is taken into consideration
                        th_rel_corrected = a.outer_log.data{traj_ix}.t_th_rel_history;
                        
                        for timestep = 2:numel(th_rel_corrected)
                            if(th_rel_corrected(timestep)-th_rel_corrected(timestep-1) > 50) % Indicates a sudden upward swing
                                th_rel_corrected(timestep:end) = th_rel_corrected(timestep:end)-360;
                            elseif((th_rel_corrected(timestep)-th_rel_corrected(timestep-1) < -50)) % Indicates a sudden downward swing
                                th_rel_corrected(timestep:end) = th_rel_corrected(timestep:end)+360;
                            end
                        end

                        dx = th_rel_corrected(analysis_indices(2:end)) -...
                            th_rel_corrected(analysis_indices(1:end-1));
                        dxdt = abs(dx) / Ts;
                        mean_vel(g,traj_ix) = mean(dxdt);
                        median_vel(g,traj_ix) = median(dxdt);
                        
                        % Target and distractor response stats during
                        % intersaccade period
                        target_intersaccade.mean(g,traj_ix) = mean(a.outer_log.data{traj_ix}.max_target_history(analysis_indices),'omitnan');
                        target_intersaccade.median(g,traj_ix) = median(a.outer_log.data{traj_ix}.max_target_history(analysis_indices),'omitnan');
                        target_intersaccade.max(g,traj_ix) = max(a.outer_log.data{traj_ix}.max_target_history(analysis_indices),[],'omitnan');
                        
                        distractor_intersaccade.mean(g,traj_ix) = mean(a.outer_log.data{traj_ix}.max_distractor_history(analysis_indices),'omitnan');
                        distractor_intersaccade.median(g,traj_ix) = median(a.outer_log.data{traj_ix}.max_distractor_history(analysis_indices),'omitnan');
                        distractor_intersaccade.max(g,traj_ix) = max(a.outer_log.data{traj_ix}.max_distractor_history(analysis_indices),[],'omitnan');
                        
                        % Target and distractor response stats during 100ms
                        % prior to unsuccessful saccade
                        target_100ms.mean(g,traj_ix) = mean(a.outer_log.data{traj_ix}.max_target_history(t_end_analysis-99:t_end_analysis),'omitnan');
                        target_100ms.median(g,traj_ix) = median(a.outer_log.data{traj_ix}.max_target_history(t_end_analysis-99:t_end_analysis),'omitnan');
                        target_100ms.max(g,traj_ix) = max(a.outer_log.data{traj_ix}.max_target_history(t_end_analysis-99:t_end_analysis),[],'omitnan');
                        
                        distractor_100ms.mean(g,traj_ix) = mean(a.outer_log.data{traj_ix}.max_distractor_history(t_end_analysis-99:t_end_analysis),'omitnan');
                        distractor_100ms.median(g,traj_ix) = median(a.outer_log.data{traj_ix}.max_distractor_history(t_end_analysis-99:t_end_analysis),'omitnan');
                        distractor_100ms.max(g,traj_ix) = max(a.outer_log.data{traj_ix}.max_distractor_history(t_end_analysis-99:t_end_analysis),[],'omitnan');
                    end
                end
            end
        end
    end
end

logs.num_saccades = num_saccades;
logs.t_saccade_onscreen = t_saccade_onscreen;
logs.mean_vel = mean_vel;
logs.median_vel = median_vel;
logs.target_intersaccade = target_intersaccade;
logs.distractor_intersaccade = distractor_intersaccade;
logs.target_100ms = target_100ms;
logs.distractor_100ms = distractor_100ms;
logs.t_lost_it = t_lost_it;

save([save_dir 'temp_lost19_09_23_all.mat'],'logs','gind_to_load');