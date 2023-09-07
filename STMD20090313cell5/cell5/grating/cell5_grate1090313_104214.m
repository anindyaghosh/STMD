% This is an automatically generated parameter file
% Do not edit
% Parameter file saved on Friday, 13 March 2009 at 10:42:17 AM;
timing_error = 0;
% Vision Egg parameters
stim_type = 'grate1';
finished_time = 20090313104216;
azimuth = 0.0;
contrast = 1.0;
elevation = 0.0;
mouse_mode = 0;
orient = 180.0;
post_stim_sec = 1.0;
pre_stim_sec = 1.0;
qdir_adapt_dur_sec = 3.0;
qdir_num_orients = 16;
qdir_rotate_cw = 1;
qdir_step_dur_sec = 0.5;
ramp_linearity = 3;
ramp_order = 10;
sf = 0.10000000000000001;
stim_sec = 0.5;
sub_type = 'Contrast Step';
tf = 5.0;
window_func = 'circle';
window_radius = 100.0;
frames_dropped = 0; % boolean;
go_loop_start_time_abs_sec = 1267.6888571033469;
% Data Aquasition Settings
amplifier_gain = 10; % Gain of recording amplifier
daq_board_gain = 1/2.048; % Dummy gain for old analysis programs
daq_delta_t_sec = 0.000200;
my_length = 13428; % Length of recording
fid = fopen('cell5_grate1090313_104214.dat','r','b'); % Open binary data as IEEE-32bit big-endian
Header = fread(fid, 12, 'float'); % Read the first 12 floating point numbers into a header variable
channels = Header(2); % Find the number of data channels recorded
if exist('limit_data')==1; % See if the variable limit_data has been defined
my_length = limit_data; % If the data is to be limited in length then only use the length specified
end; % Exit if statement
All_data = fread(fid, [channels,my_length], 'float'); % Read the specified length of data in as a floating point array
All_data = All_data'; % Transposes the All_data array so the different channels are in columns rather than rows
data = All_data(:,2); % Define data as 2nd channel of All_data
fclose(fid); % Close data file
clear fid; % Remove file data from memory space
time = (0:my_length-1)*daq_delta_t_sec; % Define time array for data
if exist('pre_stim_sec')==0; % See if prestimulus time has been defined
pre_stim_sec = 0; % If no prestimulus defined then use 0
end; % Exit if statement
time = time-pre_stim_sec; % Account for pre-stimulus time
data = data*1000/amplifier_gain; % Rescale data to be in mV
if exist('plot_data')==1; % See if the variable plot_data has been defined
if plot_data==1; % If plot_data is defined see if it is equal to 1
plot(time,data); % If plot_data is 1 then plot the data channel
xlabel('time (s)');
ylabel('Recorded Data (mV)');
end; % Exit if statement
end; % Exit if statement