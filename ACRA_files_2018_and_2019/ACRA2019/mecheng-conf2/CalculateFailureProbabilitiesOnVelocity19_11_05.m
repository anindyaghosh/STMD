clear all
close all

% Calculate a running median of the target velocity for each trajectory
% Based on that, calculate the probability that failure occurs if the
% median velocity is at a certain value.

trajfile='E:\PHD\TEST\trajectories19_09_06.mat';
load(trajfile);

% trajfile= 'D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat';
% load('D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat');

go_back = 30;

% savename= ['D:\simfiles\conf2\RunningMedianTargetVelocity19_10_11_gb' num2str(go_back) '.mat'];
savename= ['E:\PHD\TEST\RunningMedianTargetVelocity19_10_11_gb' num2str(go_back) '.mat'];

if(~exist(savename,'file'))
    % For each trajectory
    num_traj = numel(traj);
    traj_length = numel(traj{1}.t_th_traj);

    rmtv = NaN(traj_length,num_traj);

    tic
    for traj_ix = 1:numel(traj)
        disp(['traj_ix=' num2str(traj_ix) ' ' num2str(toc)])
        for k=100:traj_length
            use_range = k-go_back+1:k;
            rmtv(k,traj_ix) = median(abs(traj{traj_ix}.t_vel_traj(use_range)));
        end
    end

    % plot(rmtv(:,1:10),'-')

    this_script = mfilename;
    description = 'rmtv(k,traj_ix) contains the median absolute target velocity during k-goback+1:k for the trajectory specified by traj_ix. The absolute has been used because direction does not affect ESTMD output strength.';

    save(savename,'rmtv','trajfile','description','this_script','go_back')
else
    load(savename);
end

%%
% Need to bin the median target 
f=figure(1);
b=sort(rmtv(:));
plot(b)

f=figure(2);
edges=0:10:140;

nc = histcounts(b,edges);

plot(0.5*edges(1:end-1)+edges(2:end),nc);
xlabel('Target speed (°/s)')
ylabel('Frequency')

%%
% Take an example



