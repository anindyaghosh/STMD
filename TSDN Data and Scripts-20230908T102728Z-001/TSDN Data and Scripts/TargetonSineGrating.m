% Last Update 230901 Sarah Nicholas
% This script is used to analyse the effect of sine grating image moving in 
% the same or opposite of a target has on TSDN responses to a small target. 
% The mean and median response are calculated for the entre period the 
% target appears on the screen
% NOTE - This script does not seperate responses based on the contrast of the sine
% grating

clearvars -except experimentnames indata nfiles selpath search_term repeatnumber_all...
PRISM_NormalizedMean PRISM_NormalizedMedian PRISM_ExampleSpikeHistogram...
PRSIM_TimeAxisHistogram example_histogramtouse ExampleExptDate ExampleNP

search_term='*-Target_SineGrating-*';
[indata,filenames] = Dataload_Extracellular(search_term,2); %Loads all files in the current folder containing the text in quotes and that many layers
unit_number =  1;

%% Check Sampling Frequency

screen_width=indata{2,1}.debugData.screenData.rect(3);
frame_rate=indata{2,1}.debugData.screenData.hz;
pre_stim=indata{2,1}.Layer_2_Parameters(1).PreStimTime/frame_rate;
post_stim=indata{2,1}.Layer_2_Parameters(1).PostStimTime/frame_rate;
scan_duration=indata{2,1}.Layer_2_Parameters(1).Time/frame_rate;
total_time=pre_stim + scan_duration + post_stim;
total_samples=length(indata{2,1}.Units);
sf_check=total_samples/total_time
sf=round((total_samples/total_time),-4)

total_frames=indata{1,1}.Layer_2_Parameters(1).PreStimTime + indata{1,1}.Layer_2_Parameters(1).Time + indata{1,1}.Layer_2_Parameters(1).PostStimTime;
time_axis=linspace(0,total_time,round(total_time * sf)); 

INDX_lower=indata{2,1}.Layer_2_Parameters(1).PreStimTime/frame_rate *sf;
INDX_upper=INDX_lower + indata{2,1}.Layer_2_Parameters(1).Time/frame_rate *sf;

INDX_lower_OpticFlow=indata{2,1}.Layer_1_Parameters(1).PreStimTime/frame_rate *sf;
INDX_upper_OpticFlow=indata{2,1}.Layer_2_Parameters(1).PreStimTime/frame_rate *sf;

%% Extract Responses and Parameter info  

for i=1:length(filenames)
data_temp=indata{i,1}.Units(unit_number,:);    
Temporal_Freq(i)=indata{i,1}.Layer_1_Parameters.Temporal_Freq;
Direction(i)=indata{i,1}.Layer_1_Parameters.Direction;
Bg_On(i)=indata{i,1}.Layer_1_Parameters.Time;
Bg_Contrast(i)=indata{i,1}.Layer_1_Parameters.Contrast;         
Target_ON(i)=indata{i,1}.Layer_2_Parameters.Time;
Target_Direction(i)=indata{i,1}.Layer_2_Parameters.Direction;
name(i)=filenames(i);
           
figure(1),clf  
data_temp=data_temp(1:round(total_frames/frame_rate * sf));
data_matrix(i,:)=data_temp(1:end);
plot(time_axis,data_matrix(i,:),'k')          
pause(0.01)        
end

%% Output

output=[Bg_On' Temporal_Freq' Direction' data_matrix/5];

%% Search Criteria

% Target Alone 
temp=find(output(:,1)==0);
TargetAlone_repeats=length(temp)
for a=1:length(temp)
searchcriteria(temp(a))=1;
end

% Target Stationary
temp=find((output(:,1)>159) & (output(:,2)==0));
TargetStationary_repeats=length(temp)
for a=1:length(temp)
searchcriteria(temp(a))=2;
end

% Sine Grating moving Same Direction as Target
if Target_Direction(1)==0 
temp=find((output(:,1)>159) & (output(:,2)==5) & (output(:,3)==0));
else
temp=find((output(:,1)>159) & (output(:,2)==5) & (output(:,3)==180));
end
SineGratingSameDirection_repeats=length(temp)
for a=1:length(temp)
searchcriteria(temp(a))=3;
end

% Sine Grating moving Opposite Direction to Target
if Target_Direction(1)==0 
temp=find((output(:,1)>159) & (output(:,2)==5) & (output(:,3)==180));
else
temp=find((output(:,1)>159) & (output(:,2)==5) & (output(:,3)==0));
end
SineGratingOppositeDirection_repeats=length(temp)
for a=1:length(temp)
searchcriteria(temp(a))=4;
end

output2=[searchcriteria' data_matrix/5];
unique_search=unique(searchcriteria);
Search_Titles={'Target Alone', 'Stationary', 'Same Direction', 'Opposite Direction'};

%% Average Spike Histogram using Moving Mean 50ms Bins for each Unique Background
    
Bin_Size_ms = 50;
slidingwindowlength= (Bin_Size_ms/1000)*sf;

for k=1:length(unique_search);
temp=find((output2(:, 1) == unique_search(k)));
for d=1:length(temp)
Temp_movsum(d, :)  = (movsum(output2(temp(d), 3:end),slidingwindowlength,2))/(Bin_Size_ms/1000);
end
Mean_movsum(k, :) = mean(Temp_movsum);
clear Temp_movsum %effected by number of repeats hence needs to be cleared
end

time_axis_bins=linspace(0,time_axis(end),length(Mean_movsum));   

figure(5),clf
t1=tiledlayout(1,3);
load('screeninfo.mat')
title(t1,({['TSDN Response to Target on Sine Grating - Contrast ' num2str((Bg_Contrast(1)*100)) '%'] ; [ExptDate NP]})) 
nexttile
set(gcf, 'Position', [50 1000 1200 400]);
plot(time_axis_bins,Mean_movsum);
hold on
plot([INDX_lower/sf INDX_upper/sf], [-15 -15], '-r');  %target on timing
plot([INDX_lower_OpticFlow/sf INDX_upper/sf], [-25 -25], '-b');  %Background timing
legend(Search_Titles, 'Location','northwest');
title('\fontsize{14} \bf Spike Histograms')
text(INDX_upper/sf+0.01,-15,'Target On','Color','red')
text(INDX_upper/sf+0.01,-25,'Background On','Color','blue')
xlabel('Time (s)')
ylabel('Response (spikes.s^{-1})')
axis([0 2 -50 inf])

%% Plot Average Response over Entire Time Target on Screen for each Unique Background
    
Max_repeats=max([TargetAlone_repeats TargetStationary_repeats SineGratingSameDirection_repeats SineGratingOppositeDirection_repeats]);
SumSpikes=NaN(Max_repeats,length(unique_search));

for k=1:length(unique_search)
temp=find((output2(:, 1) == unique_search(k)));
for d=1:length(temp)
SumSpikes(d,k)=sum(output2(temp(d),(round(INDX_lower):round(INDX_upper))));
end
end
SumSpikes_sec=SumSpikes/scan_duration;
Mean_Response=mean(SumSpikes_sec,'omitnan');
Median_Response=median(SumSpikes_sec,'omitnan');

nexttile
plot([0.95 1.95 2.95 3.95],[Median_Response(1) Median_Response(2) Median_Response(3) Median_Response(4)],'ok','MarkerFaceColor', 'b')
hold on
plot([1.05 2.05 3.05 4.05],[Mean_Response(1) Mean_Response(2) Mean_Response(3) Mean_Response(4)],'ok','MarkerFaceColor', 'r')
plot(SumSpikes_sec','.k')
hold on
ax = gca;
ax.XTick = [1 2 3 4];
ax.XTickMode = 'manual';
ax.XTickLabel = Search_Titles(1:4);
ax.XTickLabelRotation = 45;
xlabel('Direction of background motion')
title('\fontsize{14} \bf Mean response during target trajectory')
ylabel('Response (spikes.s^{-1})')
legend({'Median','Mean', 'Repeats'}, 'Location','northwest');
axis([0 5 0 inf])

%% Normalised Response for each Unique Background

NormalizedMean=Mean_Response(2:4)/Mean_Response(1);
NormalizedMedian=Median_Response(2:4)/Median_Response(1);

nexttile
plot([0.95 1.95 2.95],[NormalizedMedian(1) NormalizedMedian(2) NormalizedMedian(3)],'ok', 'MarkerFaceColor', 'b')
hold on
plot([1.05 2.05 3.05],[NormalizedMean(1) NormalizedMean(2) NormalizedMean(3)],'ok', 'MarkerFaceColor', 'r')
ax = gca;
ax.XTick = [1 2 3];
ax.XTickMode = 'manual';
ax.XTickLabel = Search_Titles(2:4);
ax.XTickLabelRotation = 45;
xlabel('Direction of background motion')
title('\fontsize{14} \bf Normalised respose')
ylabel('Normalised response to target only')
legend({'Median','Mean'}, 'Location','northwest');
axis([0 4 0 inf])
