clear all
close all

% Also plot the trajectories

load('D:\temp\TrackerFull19_10_03\trackres_1.mat')
load('D:\simfiles\conf2\trackerinfo\TrackerParamFile19_10_03.mat');
trajfile = load('D:\simfiles\conf2\trackerinfo\trajectories19_09_06.mat');

vel_pts = outer_log.settings.trackerparams.vel_pts;

vel_belief = belief_vel_fixed;

f=figure(1);
f.Name = 'Fixed velocity belief';
f.Units = 'centimeters';
f.Position(3:4) = [7.6 4];

plot_relative = true;

if(plot_relative)
    % Divide by the maximum probability such that the values are scaled
    % between 0 and 1
    plot(vel_pts,vel_belief/max(vel_belief(:)),'k-','LineWidth',2)
else
    % Just plot the raw probability values
    plot(vel_pts,vel_belief,'k-','LineWidth',2)
end

g=findall(f.Children,'Type','Axes');

% g.Units='centimeters';
% g.Position(3:4) = [7.6 3];
xlabel('Velocity magnitude (°/s)')
ylabel('Relative probability')
g.FontName = 'Times New Roman';
g.FontSize = 10;
g.Box = 'off';
g.Units='centimeters';
g.Position(4) = g.Position(4);

f.PaperPosition(3:4)=f.Position(3:4);
saveas(f,'D:\simfiles\conf2\autofigures\Fixed_Velocity_Belief.fig','fig')
saveas(f,'D:\simfiles\conf2\autofigures\Fixed_Velocity_Belief.emf','emf')

%%
f=figure(2);
f.Units='centimeters';
f.Position(3:4)=[7.6 5];
f.PaperPosition(3:4) = f.Position(3:4);

all_traj = NaN(30000,100);
selection = 31:40;
for k=1:100
    all_traj(:,k) = trajfile.traj{k}.t_th_traj;
end

plot((1:30000)/1000,all_traj(:,selection),'LineWidth',2)
xlabel('Time (s)')
ylabel('Azimuthal angle (°)')
g=findall(f.Children,'Type','Axes');

g.FontName = 'Times New Roman';
g.FontSize = 10;
g.Box = 'off';

saveas(f,'D:\simfiles\conf2\autofigures\Trajectory_examples.fig','fig')
saveas(f,'D:\simfiles\conf2\autofigures\Trajectory_examples.emf','emf')
