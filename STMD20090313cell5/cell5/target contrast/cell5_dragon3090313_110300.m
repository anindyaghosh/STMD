% This is an automatically generated parameter file
% Do not edit
% Parameter file saved on Friday, 13 March 2009 at 11:03:03 AM;
timing_error = 1;
% Vision Egg parameters
stim_type = 'dragon3';
finished_time = 20090313110303;
BG_angular_velocity = 0.0;
BG_colorB = 0.5;
BG_colorG = 0.5;
BG_colorR = 0.5;
BG_contrast = 0.0;
BG_filename = 'bg_texture_tile.tif';
BG_motion_blur_on = 1;
BG_on = 0;
BG_on_stim = 1;
BG_orientation = 0.0;
BG_src = '';
BG_update_pic = 0;
TG1_centre_x0 = 320.0;
TG1_centre_y0 = 240.0;
TG1_colorA = 1.0;
TG1_colorB = 1.0;
TG1_colorG = 0.0;
TG1_colorR = 0.0;
TG1_corner_x0 = 270.0;
TG1_corner_y0 = 190.0;
TG1_direction_A = 0.0;
TG1_height = 0.0;
TG1_omega_Wx = 5.0;
TG1_omega_Wy = 5.0;
TG1_on = 0;
TG1_on_stim = 0;
TG1_orbit_on = 0;
TG1_orientation = 0.0;
TG1_radius_Ax = 25.0;
TG1_radius_By = 25.0;
TG1_velocity_A = 0.0;
TG1_width = 640.0;
TG2_centre_x0 = 400.0;
TG2_centre_y0 = 300.0;
TG2_colorA = 1.0;
TG2_colorB = 1.0;
TG2_colorG = 1.0;
TG2_colorR = 1.0;
TG2_corner_x0 = 320.0;
TG2_corner_y0 = 240.0;
TG2_delay_t1 = 0.0;
TG2_delay_t2 = 1.0;
TG2_direction_A = 180.0;
TG2_height = 5.0;
TG2_nscan = 7.0;
TG2_on = 0;
TG2_on_stim = 1;
TG2_orientation = 0.0;
TG2_scan_dur = 5.0;
TG2_scan_line = 0.0;
TG2_scan_on = 0;
TG2_scan_v1 = 320.0;
TG2_scan_v2 = 320.0;
TG2_velocity_A = 320.0;
TG2_width = 5.0;
mouse_mode = 0;
post_stim_sec = 1.0;
pre_stim_sec = 1.0;
stim_sec = 0.5;
frames_dropped = 0; % boolean;
go_loop_start_time_abs_sec = 2514.3094454742154;
% Data Aquasition Settings
amplifier_gain = 10; % Gain of recording amplifier
daq_board_gain = 1/2.048; % Dummy gain for old analysis programs
daq_delta_t_sec = 0.000200;
my_length = 13419; % Length of recording
fid = fopen('cell5_dragon3090313_110300.dat','r','b'); % Open binary data as IEEE-32bit big-endian
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