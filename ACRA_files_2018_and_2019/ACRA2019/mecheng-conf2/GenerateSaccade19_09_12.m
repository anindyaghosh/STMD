function s_all = GenerateSaccade19_09_12(start_angle,end_angle,saccade_duration)
% Plot out a trajectory to move the camera from start_angle to end_angle
% with triangular velocity profile
% clear all
% close all

Ts=0.001;

% The approach for generating a saccade is to assume a triangular velocity
% distribution over time such that
% delta_theta) = 1/2 v_max * delta_t
% i.e. constant acceleration one way or the other, with the magnitude of
% acceleration determined by
% acc_mag = v_max / (0.5*delta_t)

% % acc_required is the accelerations actually achieved the heads of animals
% % in Heuristic Rules Underlying Dragonfly Prey Selection and Interception
% % and therefore represents a realistic head angular acceleration
% phys_peak_vel = 1200;
% phys_saccade_duration = 0.05;
% acc_phys = 2*phys_peak_vel/phys_saccade_duration;

% start_angle=-10;
% end_angle=10;
% saccade_duration = 0.05;

timesteps=(saccade_duration / Ts) + 1;
mid_time = ceil(timesteps/2);

sac_th=NaN(timesteps,1);
sac_th(1) = start_angle;

vmax = 2*(end_angle-start_angle) / saccade_duration;
acc_val = vmax / (0.5*saccade_duration);

vel=NaN(timesteps,1);
vel(ceil(timesteps/2)) = vmax;
vel(1)=0;

% sac_th(end) = end_angle;

% dir = sign(end_angle - start_angle);

for t=2:mid_time
   vel(t) = vel(t-1) + Ts*acc_val;
end
for t=mid_time+1:timesteps
    vel(t) = vel(t-1) - Ts*acc_val;
end

% Calculate displacements for upstroke
up_seconds = ((1:mid_time)-1)*Ts;
up_indexes = 1:mid_time;

s_delta = 0.5*vel(up_indexes(:)) .* up_seconds(:);
s_up = start_angle + s_delta;
s_down = end_angle - s_delta(end-1:-1:1);
s_all=[s_up;s_down];
% plot(s_all)
return