clear all
close all

seed_range = 0:100;

this_script = mfilename;

traj=cell(numel(seed_range),1);

traj_length = 30000;
t_vel_initial = 120;
vel_std=80;
vel_mean=0;
vel_period = 500; % How many time steps between intended velocity changes
Ts = 0.001;
acc_value = 80; % deg /s /s

settings.traj_length = traj_length;
settings.t_vel_initial = t_vel_initial;
settings.vel_std = vel_std;
settings.vel_mean = vel_mean;
settings.vel_period = vel_period;
settings.Ts = Ts;
settings.acc_value = acc_value;
settings.seed_range = seed_range;
settings.this_script = this_script;

tic
for seed_ix = 1:numel(seed_range)
    disp([num2str(seed_ix) '/' num2str(numel(seed_range)) ' ' num2str(toc)])
    
    curr_seed = seed_range(seed_ix);
    rng(curr_seed);
    
    % Generate a target movement trajectory.
    % 1) Move in from the left hand side
    % 2) Move with some sensible dynamic model
    %
    % Trajectories generated relative to zero with the intention that in the
    % model script itself this will be offset to -hfov/2

    t_th_traj = NaN(traj_length,1);
    t_th_traj(1)=0;
    t_al_traj = zeros(traj_length,1); % Just keeping it on the midline for now
    
    t_vel_traj = NaN(traj_length,1);
    
    t_vel_traj(1) = t_vel_initial;
    next_vel = t_vel_initial;
    next_vel_history=NaN(traj_length,1);
    
    for t=2:traj_length
        if(mod(t,vel_period) == 0)
            % Update velocity setpoint
            next_vel = norminv(0.05+rand()*0.9,vel_mean,vel_std);
        end
        
        next_vel_history(t) = next_vel;
        
        % Velocity update
        % Change velocity
        if(abs(t_vel_traj(t-1) - next_vel) < acc_value*Ts)
            t_vel_traj(t) = next_vel;
        else
            t_vel_traj(t) = t_vel_traj(t-1) + sign(next_vel - t_vel_traj(t-1)) * acc_value*Ts;
        end
        
        % Update position using average velocity
        t_th_traj(t) = t_th_traj(t-1) + 0.5*(t_vel_traj(t)+t_vel_traj(t-1))*Ts;
    end
%     plot([t_th_traj(:) t_vel_traj(:)])
    
    traj{seed_ix}.t_th_traj = t_th_traj;
    traj{seed_ix}.t_vel_traj = t_vel_traj;
    traj{seed_ix}.next_vel_history = next_vel_history;
    traj{seed_ix}.t_al_traj = t_al_traj;
end

save('D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat','traj','settings');