clear all
close all

input_video_list={'Pursuit1.mj2';...
    'Pursuit2.mj2';...
    '2018_05_31_11_54_11.mj2';...
    '2018_05_31_13_48_46.mj2';...
    'video14.mj2';...
    'video15.mj2';...
    'Stationary1.mj2';...
    'Stationary2.mj2';...
    'Stationary3.mj2'};

reference_vids={'F:\Drone footage\Pursuits\DJI_0001.MP4';...
    'F:\Drone footage\Pursuits\DJI_0002.MP4';...
    'F:\Drone footage\720 stream\2018_05_31_11_54_11.mp4';...
    'F:\Drone footage\720 stream\2018_05_31_13_48_46.mp4';...
    'F:\MrBlack\Data\GC_ CIT\video14.mkv';...
    'F:\MrBlack\Data\GC_ CIT\video15.mkv';...
    'F:\Drone footage\Stationary\DJI_0001.MP4';...
    'F:\Drone footage\Stationary\DJI_0002.MP4';...
    'F:\Drone footage\Stationary\DJI_0003.MP4'};

part_size=600;
part_padding=150;

video_parts=zeros(numel(reference_vids),1);
for k=1:numel(reference_vids)
    vread=VideoReader(reference_vids{k});
    frames=vread.get('FrameRate')*vread.get('Duration');
    frames=floor(frames);

    video_parts(k)=ceil((frames-part_padding)/part_size);
end

vidin_params1=cell(sum(video_parts),1);
vidin_params2=zeros(sum(video_parts),1);
vidin_params3=vidin_params2;
curr_line=1;
for k=1:numel(input_video_list)
    for ix=1:video_parts(k)
        vidin_params1{curr_line}=input_video_list{k};
        vidin_params2(curr_line)=ix;
        vidin_params3(curr_line)=video_parts(k);
        curr_line=curr_line+1;
    end
end

[vixnum,~]=size(vidin_params1);
vix_range=1:vixnum;

pixel2PR_range=18:18:100;
sigma_hw_pixel_range=18:18:300; % In units of pixels now

[vix_s,pixel2PR_s, sigma_hw_pixel_s]=ndgrid(vix_range,pixel2PR_range,sigma_hw_pixel_range);

vidin_full=cell(numel(vix_s),4);

for k=1:numel(vix_s)
    vidin_full{k,1}=vidin_params1{vix_s(k)};
    vidin_full{k,2}=vidin_params2(vix_s(k));
    vidin_full{k,3}=vidin_params3(vix_s(k));
    vidin_full{k,4}=pixel2PR_s(k);
    vidin_full{k,5}=sigma_hw_pixel_s(k);
end

xlswrite('C:\Users\John\Desktop\PhD\tempout.xlsx',vidin_full)

