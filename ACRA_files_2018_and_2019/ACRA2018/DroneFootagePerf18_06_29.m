function DroneFootagePerf18_06_29(grp_file,vid_dir,vid_file,gtfile,load_dir,target_vidnum,vardo)
% % grp_file='C:\Users\John\Desktop\PhD\paramfiles\DroneVidParams18_06_22.txt';
% grp_file='/fast/users/a1119946/simfiles/dronefiles/DroneVidParams18_06_22.txt';
% % vid_dir='D:\Drone footage\MJ2 Format\';
% vid_dir='/fast/users/a1119946/dronevids/';
% vid_file='2018_05_31_11_54_11.mj2';
% % gtfile='D:\Drone footage\MJ2 Format\2018_05_31_11_54_11_gt.mat';
% gtfile='/fast/users/a1119946/simfiles/dronefiles/gtruth/2018_05_31_11_54_11_gt_partial.mat';
% % load_dir='D:\Drone footage\Results\';
% load_dir='/fast/users/a1119946/simfiles/dronefiles/Maxlocs/data/';

% savename=['C:\Users\John\Desktop\PhD\droneprocess\out.mat'];
savename=['/fast/users/a1119946/simfiles/dronefiles/processed/' vid_file(1:end-4) '-' num2str(vardo) '.mat'];

% target_vidnum=1;
part_size=600;
part_padding=150;
% skip_frames=6;

vread=VideoReader([vid_dir vid_file]);
framerate=get(vread,'FrameRate');
vraw=get(vread,'Height');
hraw=get(vread,'Width');

frameratio=1000/framerate;

file=fopen(grp_file,'r');
grp_info=textscan(file,'%f%s%f%f%f%f','Headerlines',1,'Delimiter','\t');
fclose(file);

input_video_end=grp_info{2};
partnum=grp_info{3};
numparts=grp_info{4};
pixel2PR_s=grp_info{5};
sigma_hw_s=grp_info{6};

uvidnames=unique(input_video_end);
vidnum=zeros(numel(input_video_end),1);
for t=1:numel(input_video_end)
    for k=1:numel(uvidnames)
        if(strcmp(input_video_end{t},uvidnames{k}))
            vidnum(t)=k;
            break;
        end
    end
end

pixel2PR_range=18:6:60;
sigma_hw_range=0.8:0.2:5.6;

[pixel2PR_top_s,sigma_hw_top_s]=meshgrid(pixel2PR_range , sigma_hw_range);

cs_kernel_size_range=3:2:13; % Keep it odd
fdsr_lp_range=[0.005 0.0250    0.0450    0.0650    0.0850];
[cs_kernel_size_s,fdsr_lp_s]=meshgrid(cs_kernel_size_range,fdsr_lp_range);

vidmask= vidnum == target_vidnum;
vidmask=numparts(vidmask);
maxpart=vidmask(1); % Only go as far as this part

all_inds=1:numel(input_video_end);

ispresent=false(size(input_video_end));
tic
for p=1:numel(pixel2PR_top_s)
    disp(['Commencing p= ' num2str(p) ' of ' num2str(numel(pixel2PR_top_s)),...
        '. ' num2str(toc)])
    [p_row,p_col]=ind2sub(size(pixel2PR_top_s),p);
    pixel2PR=pixel2PR_top_s(p);
    sigma_hw=sigma_hw_top_s(p);
    
    % Identify the results files to load by combining masks
    pixel2PR_mask=pixel2PR_s == pixel2PR;
    sigma_hw_mask=sigma_hw_s == round(sigma_hw,1);
    partnum_mask = partnum <= maxpart;
    vidname_mask = vidnum == target_vidnum;
    
    target_mask = pixel2PR_mask & ...
        sigma_hw_mask & ...
        partnum_mask & ...
        vidname_mask;
    
    % Load results from the relevant files
    load_indices = all_inds(target_mask);
    
    % format is (cs kernel / fdsr_lp setting, frame, rank in top 10)
    maxlocs_DARK.row=uint16(zeros(30,maxpart*part_size,10));
    maxlocs_DARK.column=maxlocs_DARK.row;
    maxlocs_DARK.value=maxlocs_DARK.row;
    
    maxlocs_LIGHT.row=maxlocs_DARK.row;
    maxlocs_LIGHT.column=maxlocs_DARK.row;
    maxlocs_LIGHT.value=maxlocs_DARK.row;
    
    for t=1:numel(load_indices)
        filename=[load_dir 'gind' num2str(load_indices(t)) '.mat'];
        if(exist(filename,'file'))
            a=load(filename);
            maxlocs_DARK.row(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.row;
            maxlocs_DARK.column(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.column;
            maxlocs_DARK.value(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.value;
            
            maxlocs_LIGHT.row(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.row;
            maxlocs_LIGHT.column(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.column;
            maxlocs_LIGHT.value(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.value;
            
            ispresent(load_indices(t))=true;
        end
    end
    if(sum(ispresent(load_indices)) ~= numel(load_indices))
        disp(['Could not load for p=' num2str(p)])
    else
        
        % Work out the subsampling scheme
        
        sigma_deg = sigma_hw/2.35;
        sigma_pixel = sigma_deg*pixel2PR;
        kernel_size = 2*(ceil(sigma_pixel));
        if(~mod(kernel_size,2))
            kernel_size=kernel_size+1;
        end
        blurbound=floor(kernel_size/2);
        
        newv=vraw-2*blurbound;
        newh=hraw-2*blurbound;
        
        hfov=floor(newh/pixel2PR);
        vfov=floor(newv/pixel2PR);
        
        % Create ground truth for subsampled imagery. Start by interpolating
        % between frames
        a=load(gtfile);
        [~,savelen,~]=size(maxlocs_DARK.row);
        gtbuttonfill=zeros(1,a.record_frames(end));
        gtxfill=zeros(size(gtbuttonfill));
        gtyfill=gtxfill;
        
        if(vardo == 1)
            if(~exist('successes_DARK','var'))
                successes_DARK=uint8(zeros([10 (part_padding+savelen) size(pixel2PR_top_s) size(cs_kernel_size_s)]));
            end
        elseif(vardo == 2)
            if(~exist('successes_DARK_delayed','var'))
                successes_DARK_delayed=uint8(zeros([10 (part_padding+savelen) size(pixel2PR_top_s) size(cs_kernel_size_s)]));
            end
        elseif(vardo == 3)
            if(~exist('successes_LIGHT','var'))
                successes_LIGHT=uint8(zeros([10 (part_padding+savelen) size(pixel2PR_top_s) size(cs_kernel_size_s)]));
            end
        elseif(vardo == 4)
            if(~exist('successes_LIGHT_delayed','var'))
                successes_LIGHT_delayed=uint8(zeros([10 (part_padding+savelen) size(pixel2PR_top_s) size(cs_kernel_size_s)]));
            end
        end
        
        for gtix=2:numel(a.record_frames)
            if(a.gt_button(gtix-1) == 1 && a.gt_button(gtix) == 1)
                gtbuttonfill(a.record_frames(gtix-1):a.record_frames(gtix))=1;
                
                step=(a.gt_x(gtix)-a.gt_x(gtix-1))/(a.record_frames(gtix)-a.record_frames(gtix-1));
                step_basis=0:a.record_frames(gtix)-a.record_frames(gtix-1);
                gtxfill(a.record_frames(gtix-1):a.record_frames(gtix))=a.gt_x(gtix-1)+step_basis*step;
                
                step=(a.gt_y(gtix)-a.gt_y(gtix-1))/(a.record_frames(gtix)-a.record_frames(gtix-1));
                step_basis=0:a.record_frames(gtix)-a.record_frames(gtix-1);
                gtyfill(a.record_frames(gtix-1):a.record_frames(gtix))=a.gt_y(gtix-1)+step_basis*step;
            end
        end
        
        % Convert gt frames to subsampled space
        lbounds=blurbound-pixel2PR/2+pixel2PR*(0:hfov);
        ubounds=blurbound-pixel2PR/2+pixel2PR*(0:vfov);
        
        gtx_ss=zeros(size(gtxfill));
        gty_ss=gtx_ss;
        for gtix=1:a.record_frames(end)
            for h=1:hfov
                for v=1:vfov
                    if(gtxfill(gtix) >= lbounds(h) &&...
                            gtxfill(gtix) < lbounds(h+1) &&...
                            gtyfill(gtix) >= ubounds(v) &&...
                            gtyfill(gtix) < ubounds(v+1))
                        gtx_ss(gtix)=h;
                        gty_ss(gtix)=v;
                    end
                end
            end
        end
        
        % Test agreement between any of the top 10 responses per frame and the
        % ground truth, bearing in mind that the first part_padding frames should
        % be ignored
        
        success_threshold=1.5;
        
        if(vardo == 1)
            for t=1:(numel(gtx_ss)-part_padding);
                framenum=part_padding + t;
                for csix=1:30
                    [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                    for k=1:10
                        x=double(maxlocs_DARK.column(csix,t,k));
                        y=double(maxlocs_DARK.row(csix,t,k));
                        if(((gtx_ss(framenum)-x)^2+...
                                (gty_ss(framenum)-y)^2)^0.5 < success_threshold && gtbuttonfill(framenum) == 1)
                            for kex=k:10
                                successes_DARK(kex,framenum,p_row,p_col,cs_row,cs_col)=...
                                    successes_DARK(kex,framenum,p_row,p_col,cs_row,cs_col)+1;
                            end
                            break;
                        end
                    end
                end
            end
        end
        
        if(vardo == 2)
            for t=1:(numel(gtx_ss)-part_padding);
                framenum=part_padding + t;
                for csix=1:30
                    [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                    for k=1:10
                        x=double(maxlocs_DARK.column(csix,t,k));
                        y=double(maxlocs_DARK.row(csix,t,k));
                        if(((gtx_ss(framenum-1)-x)^2+...
                                (gty_ss(framenum-1)-y)^2)^0.5 < success_threshold  && gtbuttonfill(framenum) == 1)
                            for kex=k:10
                                successes_DARK_delayed(kex,framenum,p_row,p_col,cs_row,cs_col)=...
                                    successes_DARK_delayed(kex,framenum,p_row,p_col,cs_row,cs_col)+1;
                            end
                            break;
                        end
                    end
                end
            end
        end
        
        if(vardo == 3)
            for t=1:(numel(gtx_ss)-part_padding);
                framenum=part_padding + t;
                for csix=1:30
                    [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                    for k=1:10
                        x=double(maxlocs_LIGHT.column(csix,t,k));
                        y=double(maxlocs_LIGHT.row(csix,t,k));
                        if(((gtx_ss(framenum)-x)^2+...
                                (gty_ss(framenum)-y)^2)^0.5 < success_threshold  && gtbuttonfill(framenum) == 1)
                            for kex=k:10
                                successes_LIGHT(kex,framenum,p_row,p_col,cs_row,cs_col)=...
                                    successes_LIGHT(kex,framenum,p_row,p_col,cs_row,cs_col)+1;
                            end
                            break;
                        end
                    end
                end
            end
        end
        
        if(vardo == 4)
            for t=1:(numel(gtx_ss)-part_padding);
                framenum=part_padding + t;
                for csix=1:30
                    [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                    for k=1:10
                        x=double(maxlocs_LIGHT.column(csix,t,k));
                        y=double(maxlocs_LIGHT.row(csix,t,k));
                        if(((gtx_ss(framenum-1)-x)^2+...
                                (gty_ss(framenum-1)-y)^2)^0.5 < success_threshold  && gtbuttonfill(framenum) == 1)
                            for kex=k:10
                                successes_LIGHT_delayed(kex,framenum,p_row,p_col,cs_row,cs_col)=...
                                    successes_LIGHT_delayed(kex,framenum,p_row,p_col,cs_row,cs_col)+1;
                            end
                            break;
                        end
                    end
                end
            end
        end
    end
end

description='rank in top 10, sigma_hw value, pixel2_PR value, cs value, fdsr_lp value';

if(vardo == 1)
    save(savename,'vid_file','grp_file','description','ispresent',...
    'cs_kernel_size_s','sigma_hw_s','pixel2PR_s','fdsr_lp_s',...
    'successes_DARK','-v7.3');
elseif(vardo == 2)
    save(savename,'vid_file','grp_file','description','ispresent',...
    'cs_kernel_size_s','sigma_hw_s','pixel2PR_s','fdsr_lp_s',...
    'successes_DARK_delayed','-v7.3');
elseif(vardo == 3)
    save(savename,'vid_file','grp_file','description','ispresent',...
    'cs_kernel_size_s','sigma_hw_s','pixel2PR_s','fdsr_lp_s',...
    'successes_LIGHT','-v7.3');
elseif(vardo == 4)
    save(savename,'vid_file','grp_file','description','ispresent',...
    'cs_kernel_size_s','sigma_hw_s','pixel2PR_s','fdsr_lp_s',...
    'successes_LIGHT_delayed','-v7.3');
end
