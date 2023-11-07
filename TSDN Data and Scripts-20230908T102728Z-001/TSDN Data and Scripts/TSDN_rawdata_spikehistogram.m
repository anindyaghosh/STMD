% Last Update 230901 Sarah Nicholas
% This script is used to plot raw data traces and spike histograms of a TSDN 
% in response to a small target with different backgrounds

clear all

%% CHANGE THIS NUMBER TO LOAD DIFFERENT BACKGROUND CONDITIONS!!!
filename_toload= 7; 

%%  Open Folder containing dataset to analyse

selpath=uigetdir; % Select folder containing example data 
cd(selpath)
search_term='*TargetRight*';

binsize= 0.02; %bin size 20ms
Number_of_Layers=2;
unit_number=1;  

%% Load Data in Response to Target Trajectory

[indata,filenames] = Dataload_RawData(search_term,Number_of_Layers,filename_toload); %Loads all files in the current folder containing the text in quotes and that many layers
filenames_target=filenames';

%% Check Sampling Frequency and Stimulus Timing

frame_rate=indata{1,1}.debugData.screenData.hz;
pre_stim=sum(indata{1,1}.Layer_2_Parameters(1).PreStimTime/frame_rate);
post_stim=sum(indata{1,1}.Layer_2_Parameters(1).PostStimTime/frame_rate);
scan_duration=sum(indata{1,1}.Layer_2_Parameters(1).Time/frame_rate);
total_time=pre_stim + scan_duration + post_stim;
total_frames=sum(indata{1,1}.Layer_1_Parameters(1).PreStimTime + indata{1,1}.Layer_1_Parameters(1).Time + indata{1,1}.Layer_1_Parameters(1).PostStimTime);
total_samples=length(indata{1,1}.Units);
sf_check=total_samples/total_time
sf=round((total_samples/total_time),-4);
time_axis_loom=linspace(0,total_time,round(total_frames/frame_rate * sf));
INDX_lower=sum((indata{1,1}.Layer_1_Parameters(1).PreStimTime/frame_rate *sf));
INDX_upper= sum((indata{1,1}.Layer_1_Parameters(1).PreStimTime/frame_rate *sf) + (indata{1,1}.Layer_1_Parameters(1).Time/frame_rate *sf));
loom_start=INDX_lower/sf;
loom_end=INDX_upper/sf;
screen_dim=indata{1, 1}.debugData.screenData.partial;
fly_distance=indata{1, 1}.debugData.screenData.flyDistance;
monitor_height=indata{1, 1}.debugData.screenData.monitorHeight;
ifi=indata{1, 1}.debugData.screenData.ifi;

%% Background Condition Used

Background_On=indata{1,1}.Layer_1_Parameters(1).Time;
Temporal_Freq=indata{1,1}.Layer_1_Parameters(1).Temporal_Freq;
Bg_Direction=indata{1,1}.Layer_1_Parameters(1).Direction;
Target_Direction=indata{1,1}.Layer_2_Parameters(1).Direction;

if Background_On==0
Background_Condition='Target Only'
elseif Temporal_Freq==0
Background_Condition='Stationary Background'
elseif Bg_Direction==Target_Direction
Background_Condition='Background Moving in Same Direction as Target'
else
Background_Condition='Background Moving in Opposite Direction as Target'
end

%% Stimulus Timing for Background

pre_stim=indata{1,1}.Layer_1_Parameters(1).PreStimTime/frame_rate;
post_stim=indata{1,1}.Layer_1_Parameters(1).PostStimTime/frame_rate;
scan_duration=indata{1,1}.Layer_1_Parameters(1).Time/frame_rate;
total_time=pre_stim + scan_duration + post_stim;
INDX_lower=indata{1,1}.Layer_1_Parameters(1).PreStimTime/frame_rate *sf;
INDX_upper=indata{1,1}.Layer_1_Parameters(1).PreStimTime/frame_rate *sf + indata{1,1}.Layer_1_Parameters(1).Time/frame_rate *sf;
background_start=INDX_lower/sf;
background_end=INDX_upper/sf;

%% Stimulus Timing for Target Trajectory

pre_stim=indata{1,1}.Layer_2_Parameters(1).PreStimTime/frame_rate;
post_stim=indata{1,1}.Layer_2_Parameters(1).PostStimTime/frame_rate;
scan_duration=indata{1,1}.Layer_2_Parameters(1).Time/frame_rate;
total_time=pre_stim + scan_duration + post_stim;
total_frames=indata{1,1}.Layer_2_Parameters(1).PreStimTime + indata{1,1}.Layer_2_Parameters(1).Time + indata{1,1}.Layer_2_Parameters(1).PostStimTime;
time_axis_target=linspace(0,total_time,round(total_frames/frame_rate * sf));
INDX_lower=indata{1,1}.Layer_2_Parameters(1).PreStimTime/frame_rate *sf;
INDX_upper= indata{1,1}.Layer_2_Parameters(1).PreStimTime/frame_rate *sf + indata{1,1}.Layer_2_Parameters(1).Time/frame_rate *sf;
target_start=INDX_lower/sf;
target_end=INDX_upper/sf;

%% Response to Target - Plot Raw data (Figure 4), Discrimated unit (Figure 5) and Spike Histogram in 20ms bins (Figure 6) 

units_temp=indata{1,1}.Units(unit_number,:);
rawdata_temp=indata{1,1}.DataBlock(1,:);    
Target_DiscriminatedUnit=(units_temp(1:round(total_frames/frame_rate * sf))/5);
Target_RawData=rawdata_temp(1:round(total_frames/frame_rate * sf));

% Entire Trial - Raw Data - FIGURE 4       
figure(4),clf
% set(gcf, 'Position', [0 650 400 300]);
plot(time_axis_target, Target_RawData,'k')
ylabel('Spike Amplitude (µV)')
xlabel('Time (sec)')
title({(Background_Condition) ; ['Raw Data']}) 
hold on
plot([target_start target_end], [-400 -400], 'r')
plot([background_start background_end], [-500 -500], 'b')
legend('Raw Data','Target On', 'Background On', 'location', 'northwest')
xlim ([time_axis_target(1) time_axis_target(end)]);

% Entire Trial - Discrimated Unit - FIGURE 5
figure(5),clf
% set(gcf, 'Position', [400 650 400 300]);
plot(time_axis_target,Target_DiscriminatedUnit,'color', 'k')
xlabel('Time (sec)')
title({(Background_Condition) ; ['Discriminated Unit']}) 
hold on
plot([target_start target_end], [-0.1 -0.1], '-r')
plot([background_start background_end], [-0.2 -0.2], 'b')
legend('Discriminated Spikes','Target On', 'Background On', 'location', 'northwest')
ylim([-0.3 1.2])
xlim ([time_axis_target(1) time_axis_target(end)]);
       
% Entire Trial - Spike Histogram 20ms bins - FIGURE 6
bin=binsize*sf;
no_bins= length(Target_DiscriminatedUnit)/bin;  
a=1;
for b=1:no_bins;
Target_Total_Spikes_perbin(1, b)= sum(Target_DiscriminatedUnit( 1, (a: (a+(bin-1)))));
a=a+bin;
end
Target_Total_Spikes_perbin_spikerate=Target_Total_Spikes_perbin/binsize;
target_time_axis_bins=linspace(time_axis_target(1),time_axis_target(end),no_bins);
      
figure(6),clf
% set(gcf, 'Position', [800 650 400 300]);
bar(target_time_axis_bins, Target_Total_Spikes_perbin_spikerate, 'k')
ylabel('Response (spikes.s^{-1})')
xlabel('Time (sec)')
title({(Background_Condition) ; ['Spike Histogram -20ms bins']})
hold on      
plot([target_start target_end], [-20 -20], '-r')
plot([background_start background_end], [-30 -30], 'b')
legend('Spike Histogram','Target On', 'Background On', 'location', 'northwest')
ylim ([-50 max(Target_Total_Spikes_perbin_spikerate)*1.1]);
xlim ([time_axis_target(1) time_axis_target(end)]);
