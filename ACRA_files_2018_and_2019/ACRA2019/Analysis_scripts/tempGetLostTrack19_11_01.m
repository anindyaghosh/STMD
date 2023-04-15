% clear all
% close all

which_pc='mecheng';

if(strcmp(which_pc,'mecheng'))
    if(set == 1)
        paramfile = load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');
        load_dir = 'D:\temp\TrackerFull19_10_03\';
    elseif(set == 2)
        paramfile = load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_30.mat');
        load_dir = 'D:\temp\TrackerFull19_10_30\';
    elseif(set == 3)
        paramfile = load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_11_01.mat');
        load_dir = 'D:\temp\TrackerFull19_11_01\';
    end
%     paramfile = load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');
%     load_dir = 'D:\temp\TrackerFull19_10_03\';
    trajfile = load('D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat');
    save_dir = 'D:\temp\';
elseif(strcmp(which_pc,'phoenix'))
    if(set == 1)
        paramfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/TrackerParamFile19_10_03.mat');
        load_dir = '/fast/users/a1119946/simfiles/conf2/data/TrackerFull19_10_03/';
    elseif(set == 2)
        paramfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/TrackerParamFile19_10_30.mat');
        load_dir = '/fast/users/a1119946/simfiles/conf2/data/TrackerFull19_10_30/';
    elseif(set == 3)
        paramfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/TrackerParamFile19_11_01.mat');
        load_dir = '/fast/users/a1119946/simfiles/conf2/data/TrackerFull19_11_01/';
    end
    
    trajfile = load('/fast/users/a1119946/simfiles/conf2/data/trackerinfo/trajectories19_09_06.mat');
    
    save_dir = '/fast/users/a1119946/simfiles/conf2/data/processed/';
end

savename=['lost_it19_11_03_set_' num2str(set)];
spin_ticks = 250; % for offsets

% Load everything
if(set == 1)
    gind_range = 1:14400;
elseif( set == 2)
    gind_range = 1:8640;
elseif( set == 3)
    gind_range = 1:2880;
end

gind_to_load = gind_range;

trial_length = 30250;

numgind=numel(gind_to_load);
numtraj = 100;
Ts=0.001;

num_saccades=NaN(numgind,numtraj);
num_good_saccades = num_saccades;
num_bad_saccades = num_saccades;
t_saccade_onscreen = num_saccades;

% Velocity measures for 30ms prior to tracking loss
vel30.first.median = num_saccades;
vel30.first.max = num_saccades;
vel30.first.mean = num_saccades;
vel30.last = vel30.first;

target_30ms = vel30;
distractor_30ms = vel30;

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
% target_100ms = target_intersaccade;
% distractor_100ms = target_intersaccade;

% target_30ms = target_intersaccade;
% distractor_30ms = target_intersaccade;

t_lost_it.cause_first = false(numgind,numtraj);
t_lost_it.cause_last = t_lost_it.cause_first;

ispresent = NaN(numel(gind_range),numtraj);

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
                ispresent(g_ix,traj_ix) = true;
                
                % Eliminate the spin_ticks offset
                use_th_rel = a.outer_log.data{traj_ix}.t_th_rel_history(spin_ticks+1:end);
                use_target = a.outer_log.data{traj_ix}.max_target_history(spin_ticks+1:end);
                use_distractor = a.outer_log.data{traj_ix}.max_distractor_history(spin_ticks+1:end);
                
                % Get the matching velocity trajectory from the trajectory
                % file
                use_th_vel = trajfile.traj{traj_ix}.t_vel_traj;

                % Mask for whether the target is onscreen or not
                onscreen_mask = use_th_rel <= hfov/2 &...
                    use_th_rel >= -hfov/2;
                
                % Create a corrected obs_th_pos (to cancel out the clamping
                % between [-180 and 180];
                corrected_obs_th_pos = a.outer_log.data{traj_ix}.obs_th_pos_history(spin_ticks+1:end);
                
                wrap_mask = abs(corrected_obs_th_pos(2:end) - corrected_obs_th_pos(1:end-1)) > 90;
                
                I=find(wrap_mask)+1; % Find the times where wrapping occurred
                for k=1:numel(I)
                    curr_ix=I(k);
                    if(corrected_obs_th_pos(curr_ix) - corrected_obs_th_pos(curr_ix-1) > 90) % increased
                        corrected_obs_th_pos(curr_ix:end) = corrected_obs_th_pos(curr_ix:end) - 360;
                    else
                        corrected_obs_th_pos(curr_ix:end) = corrected_obs_th_pos(curr_ix:end) + 360;
                    end
                end
                
               % When working out the timing of saccades, need to use a
               % different approach depending on whether do_saccades is
               % true or not.
                if(do_saccades)
                    use_saccading_history = a.outer_log.data{traj_ix}.saccading_history(spin_ticks+1:end);
                    
                    % Work out the number of saccades by locating falling edges
                    % in the saccade history (marking the end of saccades)
                    saccade_end_mask = use_saccading_history(2:end) == false &...
                        use_saccading_history(1:end-1) == true;
                    
                    % Also set up the mask for when saccades commenced
                    saccade_begin_mask = use_saccading_history(2:end) == true &...
                        use_saccading_history(1:end-1) == false;
                    
                    % Both of these masks must be offset by 1
                    saccade_begin_mask = [false; saccade_begin_mask];
                    saccade_end_mask = [false; saccade_end_mask];
                    
                else
                    % Otherwise have to work out when the turns took place
                    % by looking for edges in the observer angle coming
                    % after the initial spin
                    saccade_end_mask = (corrected_obs_th_pos(2:end) ~=...
                        corrected_obs_th_pos(1:end-1)) &...
                        ~isnan(corrected_obs_th_pos(2:end)) &...
                        ~isnan(corrected_obs_th_pos(1:end-1));
                    
                    % The beginning and ends of saccades are offset by 1
                    saccade_begin_mask = [saccade_end_mask ; false];
                    saccade_end_mask = [false ; saccade_end_mask];
                end

                % Find out the time of the end of the last saccade that
                % finished with the target being onscreen
                t_saccades_initiated=[];
                analysis_indices=[];
                
                % Mask for onscreen at end of saccade
                successful_saccade_mask = saccade_end_mask &...
                        onscreen_mask;
                    
                % Offscreen at the end of the saccade
                bad_saccade_mask = saccade_end_mask &...
                    ~onscreen_mask;
                
                num_good_saccades(g,traj_ix) = sum(successful_saccade_mask);
                num_bad_saccades(g,traj_ix) = sum(bad_saccade_mask);
                
                % The first loss is when the onscreen_mask is false for the
                % first time
                temp = find(~onscreen_mask,1,'first');
                if(~isempty(temp))
                    t_lost_it.first(g,traj_ix) = temp;
                end
                
                % The last loss is (500-1) ticks prior to the end of the
                % simulation unless it was never lost.
                % If t_lost_it.last is NaN then that indicates tracking was
                % never lost
                if(sum(~onscreen_mask) > 0)
                    t_lost_it.last(g,traj_ix) = find(~isnan(use_th_rel),1,'last') - 499;
                end
                
                % Calculate vel measures in 30ms prior to the first loss of
                % tracking
                if(~isnan(t_lost_it.first(g,traj_ix)))
                    analysis_window = t_lost_it.first(g,traj_ix) - 29: t_lost_it.first(g,traj_ix);
                    
                    analysis_t_vel = use_th_vel(analysis_window);
                    analysis_target = use_target(analysis_window);
                    analysis_distractor = use_distractor(analysis_window);
                    
                    vel30.first.mean(g,traj_ix) = mean(analysis_t_vel);
                    vel30.first.median(g,traj_ix) = median(analysis_t_vel);
                    vel30.first.max(g,traj_ix) = max(analysis_t_vel);
                    
                    target_30ms.first.mean(g,traj_ix) = mean(analysis_target);
                    target_30ms.first.median(g,traj_ix) = median(analysis_target);
                    target_30ms.first.max(g,traj_ix) = max(analysis_target);
                    
                    distractor_30ms.first.mean(g,traj_ix) = mean(analysis_distractor);
                    distractor_30ms.first.median(g,traj_ix) = median(analysis_distractor);
                    distractor_30ms.first.max(g,traj_ix) = max(analysis_distractor);
                end
                
                % Calculate vel measures in 30ms prior to the last loss of tracking
                if(~isnan(t_lost_it.last(g,traj_ix)))
                   analysis_window = t_lost_it.last(g,traj_ix) - 29 : t_lost_it.last(g,traj_ix);
                   
                   analysis_t_vel = use_th_vel(analysis_window);
                   analysis_target = use_target(analysis_window);
                   analysis_distractor = use_distractor(analysis_window);
                   
                   vel30.last.mean(g,traj_ix) = mean(analysis_t_vel);
                   vel30.last.median(g,traj_ix) = median(analysis_t_vel);
                   vel30.last.max(g,traj_ix) = max(analysis_t_vel);
                   
                   target_30ms.last.mean(g,traj_ix) = mean(analysis_target);
                   target_30ms.last.median(g,traj_ix) = median(analysis_target);
                   target_30ms.last.max(g,traj_ix) = max(analysis_target);
                   
                   distractor_30ms.last.mean(g,traj_ix) = mean(analysis_distractor);
                   distractor_30ms.last.median(g,traj_ix) = median(analysis_distractor);
                   distractor_30ms.last.max(g,traj_ix) = max(analysis_distractor);
                end
                
                % Work out whether the first loss of tracking was caused by
                % a saccade or not
                if(~isnan(t_lost_it.first(g,traj_ix)))
                    if(do_saccades)
                        if(use_saccading_history(t_lost_it.first(g,traj_ix)))
                            % A saccade was in progress when the target went
                            % offscreen so blame it on the saccade
                            t_lost_it.cause_first(g,traj_ix) = true;
                        else
                            % A saccade was in progress when the target went
                            % offscreen so blame it on the saccade
                            t_lost_it.cause_first(g,traj_ix) = false;
                        end
                    else
                        if(saccade_end_mask(t_lost_it.first(g,traj_ix)))
                            t_lost_it.cause_first(g,traj_ix) = true;
                        else
                            t_lost_it.cause_first(g,traj_ix) = false;
                        end
                    end
                end
                
                % Work out whether the last loss of tracking was caused by
                % a saccade or not
                if(~isnan(t_lost_it.last(g,traj_ix)))
                    if(do_saccades)
                        if(use_saccading_history(t_lost_it.last(g,traj_ix)))
                            % A saccade was in progress when the target went
                            % offscreen so blame it on the saccade
                            t_lost_it.cause_last(g,traj_ix) = true;
                        else
                            % A saccade was in progress when the target went
                            % offscreen so blame it on the saccade
                            t_lost_it.cause_last(g,traj_ix) = false;
                        end
                    else
                        if(saccade_end_mask(t_lost_it.last(g,traj_ix)))
                            
                            t_lost_it.cause_last(g,traj_ix) = true;
                        else
                            
                            t_lost_it.cause_last(g,traj_ix) = false;
                        end
                    end
                end
            end
        end
    end
end

logs.num_saccades = num_saccades;
logs.num_good_saccades = num_good_saccades;
logs.num_bad_saccades = num_bad_saccades;
logs.t_saccade_onscreen = t_saccade_onscreen;
logs.vel30 = vel30;
logs.target_30ms = target_30ms;
logs.distractor_30ms = target_30ms;
logs.ispresent = ispresent;
% logs.target_intersaccade = target_intersaccade;
% logs.distractor_intersaccade = distractor_intersaccade;
% logs.target_100ms = target_100ms;
% logs.distractor_100ms = distractor_100ms;
logs.t_lost_it = t_lost_it;

save([save_dir savename '.mat'],'logs','gind_to_load');