

function CONF_DronePerf18_08_13(grp_file,vid_dir,vid_file,gtfile,load_dir,vardo,groupname)
% clear all
% close all

% grp_file='C:\Users\John\Desktop\PhD\ConfDronefiles\CONF_Droneparams18_08_09.txt';
% % grp_file='/fast/users/a1119946/simfiles/dronefiles/DroneVidParams18_06_22.txt';
% vid_dir='F:\Drone footage\MJ2 Format\';
% % vid_dir='/fast/users/a1119946/dronevids/';
% vid_file='2018_05_31_11_54_11.mj2';
% gtfile='F:\Drone footage\gtruth\2018_05_31_11_54_11_gt.mat';
% % gtfile='/fast/users/a1119946/simfiles/dronefiles/gtruth/2018_05_31_11_54_11_gt_partial.mat';
% load_dir='F:\Drone footage\CONF_Results\';
% % load_dir='/fast/users/a1119946/simfiles/dronefiles/Maxlocs/data/';
% vardo=1;
% groupname='TestDrone';
% locroot='C:\Users\John\Desktop\PhD\';
locroot='/fast/users/a1119946/simfiles/dronefiles/';
savedir=[locroot groupname '/'];
if(~exist(savedir,'dir'))
    mkdir(savedir)
end

% savename=['/fast/users/a1119946/simfiles/dronefiles/processed/' groupname '/'];
% savename=['/fast/users/a1119946/simfiles/dronefiles/processed/' vid_file(1:end-4) '-' num2str(vardo) '.mat'];

% target_vidnum=1;
part_size=600;
part_padding=150;
bg_edges=[0 logspace(-12,0,100)];
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
uive=unique(input_video_end);
for t=1:numel(uive)
    if(strcmp(uive{t},vid_file))
        target_vidnum=t;
    end
end

% pixel2PR_range=18:6:60;
% sigma_hw_range=0.8:0.2:5.6;

pixel2PR_range=unique(pixel2PR_s);
sigma_hw_pixel_range=unique(sigma_hw_s); % In units of pixels now

[pixel2PR_top_s,sigma_hw_top_s]=meshgrid(pixel2PR_range , sigma_hw_pixel_range);

cs_kernel_size_range=3:2:13;
fdsr_lp_range=[0.005 0.0250    0.0450    0.0650    0.0850 0.170];

[cs_kernel_size_s,fdsr_lp_s]=meshgrid(cs_kernel_size_range,fdsr_lp_range);

vidmask= vidnum == target_vidnum;
vidmask=numparts(vidmask);
maxpart=vidmask(1); % Only go as far as this part

all_inds=1:numel(input_video_end);

tic

% For
for p=1:numel(pixel2PR_top_s)
    ispresent=false(maxpart,1);
    
    save_end=['v' num2str(target_vidnum) '-p' num2str(p) '-v' num2str(vardo) '.mat'];
    savename=[savedir save_end];
    
    if(exist(savename,'file'))
        disp(['Skipping ' num2str(p)])
    else
        % Save out at the end of this
        disp(['Commencing p= ' num2str(p) ' of ' num2str(numel(pixel2PR_top_s)),...
            '. ' num2str(toc)])
        [p_row,p_col]=ind2sub(size(pixel2PR_top_s),p);
        pixel2PR=pixel2PR_top_s(p);
        sigma_hw_pixel=sigma_hw_top_s(p);
        
        % Identify the results files to load by combining masks
        pixel2PR_mask=pixel2PR_s == pixel2PR;
        sigma_hw_mask=sigma_hw_s == round(sigma_hw_pixel,1);
        partnum_mask = partnum <= maxpart;
        vidname_mask = vidnum == target_vidnum;
        
        target_mask = pixel2PR_mask & ...
            sigma_hw_mask & ...
            partnum_mask & ...
            vidname_mask;
        
        % Load results from the relevant files
        load_indices = all_inds(target_mask);
        
        % format is (cs kernel / fdsr_lp setting, frame, rank in top 10)
        maxlocs_DARK.row=uint16(zeros(numel(cs_kernel_size_s),maxpart*part_size,10));
        maxlocs_DARK.column=maxlocs_DARK.row;
        maxlocs_DARK.value=maxlocs_DARK.row;
        
        maxlocs_LIGHT.row=maxlocs_DARK.row;
        maxlocs_LIGHT.column=maxlocs_DARK.row;
        maxlocs_LIGHT.value=maxlocs_DARK.row;
        
        maxlocs_COMBO.row=maxlocs_DARK.row;
        maxlocs_COMBO.column=maxlocs_DARK.row;
        maxlocs_COMBO.value=maxlocs_DARK.row;
        
        bg_above=zeros(maxpart,numel(cs_kernel_size_s),numel(bg_edges));
        targ_above=bg_above;
        targ_total=bg_above;
        
        [~,savelen,~]=size(maxlocs_DARK.row);
        
        if(vardo == 1)
            successes_DARK=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            bg_counts.DARK=zeros([maxpart size(cs_kernel_size_s) numel(bg_edges)]);
        elseif(vardo == 2)
            successes_DARK_delayed=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            
        elseif(vardo == 3)
            
            successes_LIGHT=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            bg_counts.LIGHT=zeros([maxpart size(pixel2PR_top_s) numel(bg_edges)]);
            
        elseif(vardo == 4)
            
            successes_LIGHT_delayed=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            
        elseif(vardo == 5)
            
            successes_COMBO=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            bg_counts.COMBO=zeros([maxpart size(pixel2PR_top_s) numel(bg_edges)]);
            
        elseif(vardo == 6)
            
            successes_COMBO_delayed=uint8(zeros([10 (part_padding+savelen) size(cs_kernel_size_s)]));
            
        end

        for t=1:numel(load_indices) % group indices, ticks through the parts
            filename=[load_dir 'gind' num2str(load_indices(t)) '.mat'];
            
            this_part=partnum(load_indices(t));
            
            if(exist(filename,'file'))
                a=load(filename);
                maxlocs_DARK.row(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.row;
                maxlocs_DARK.column(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.column;
                maxlocs_DARK.value(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_DARK.value;
                
                maxlocs_LIGHT.row(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.row;
                maxlocs_LIGHT.column(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.column;
                maxlocs_LIGHT.value(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_LIGHT.value;
                
                maxlocs_COMBO.row(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_COMBO.row;
                maxlocs_COMBO.column(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_COMBO.column;
                maxlocs_COMBO.value(:,(t-1)*part_size+1:t*part_size,:)=a.maxlocs_COMBO.value;
                
                ispresent(t)=true;

                % For the part, determine how many target responses and bg
                % responses over different thresholds in bg_edges
                % Determine specificity and sensitivity for the part
                for edge_ix=1:numel(bg_edges)
                    if(vardo == 1)
                        % Here summing across the bins
                        bg_above(t,:,edge_ix)=sum(a.bg_counts.DARK(:,edge_ix:end),2); % [36,1] % [cs_kernel_s,100]
                        % Here summing across frames
                        targ_above(t,:,edge_ix)=sum(a.targresp.DARK >= bg_edges(edge_ix),2); % [36 1]
                        targ_total(t,:,edge_ix)=sum(~isnan(a.targresp.DARK),2);
                    elseif(vardo == 3)
                        bg_above(t,:,edge_ix)=sum(a.bg_counts.LIGHT(:,edge_ix:end),2);
                        targ_above(t,:,edge_ix)=sum(a.targresp.LIGHT >= bg_edges(edge_ix),2);
                        targ_total(t,:,edge_ix)=sum(~isnan(a.targresp.LIGHT),2);
                    elseif(vardo == 5)
                        bg_above(t,:,edge_ix)=a.bg_counts.COMBO(:,edge_ix);
                        targ_above(t,:,edge_ix)=sum(a.targresp.COMBO >= bg_edges(edge_ix),2);
                        targ_total(t,:,edge_ix)=sum(~isnan(a.targresp.COMBO),2);
                    end
                end
            end
        end
        
        if(sum(ispresent) < numel(ispresent))
            disp(['Could not load for p=' num2str(p)])
        else
            
            % Work out the subsampling scheme
            
            sigma_pixel = sigma_hw_pixel / 2.35;
            kernel_size = 2*(ceil(sigma_pixel));
            if(~mod(kernel_size,2))
                kernel_size=kernel_size+1;
            end
            blurbound=floor(kernel_size/2);
            
            newv=vraw-2*blurbound;
            newh=hraw-2*blurbound;
            
            hfov=floor(newh/pixel2PR);
            vfov=floor(newv/pixel2PR);
            
            % Create ground truth for subsampled imagery
            ssgt=ConvertGroundTruth18_08_09(gtfile,pixel2PR,blurbound,hfov,vfov);
            
            % Test agreement between any of the top 10 responses per frame and the
            % ground truth, bearing in mind that the first part_padding frames should
            % be ignored
            
            success_threshold=1.5;
            
            if(vardo == 1)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_DARK.column(csix,t,k));
                            y=double(maxlocs_DARK.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum)-x)^2+...
                                    (ssgt.gty_ss(framenum)-y)^2)^0.5 < success_threshold && ssgt.gtbuttonfill(framenum) == 1)
                                for kex=k:10
                                    successes_DARK(kex,framenum,cs_row,cs_col)=...
                                        successes_DARK(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            if(vardo == 2)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_DARK.column(csix,t,k));
                            y=double(maxlocs_DARK.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum-1)-x)^2+...
                                    (ssgt.gty_ss(framenum-1)-y)^2)^0.5 < success_threshold  && ssgt.gtbuttonfill(framenum-1) == 1)
                                for kex=k:10
                                    successes_DARK_delayed(kex,framenum,cs_row,cs_col)=...
                                        successes_DARK_delayed(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            if(vardo == 3)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_LIGHT.column(csix,t,k));
                            y=double(maxlocs_LIGHT.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum)-x)^2+...
                                    (ssgt.gty_ss(framenum)-y)^2)^0.5 < success_threshold  && ssgt.gtbuttonfill(framenum) == 1)
                                for kex=k:10
                                    successes_LIGHT(kex,framenum,cs_row,cs_col)=...
                                        successes_LIGHT(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            if(vardo == 4)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_LIGHT.column(csix,t,k));
                            y=double(maxlocs_LIGHT.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum-1)-x)^2+...
                                    (ssgt.gty_ss(framenum-1)-y)^2)^0.5 < success_threshold  && ssgt.gtbuttonfill(framenum-1) == 1)
                                for kex=k:10
                                    successes_LIGHT_delayed(kex,framenum,cs_row,cs_col)=...
                                        successes_LIGHT_delayed(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            if(vardo == 5)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_COMBO.column(csix,t,k));
                            y=double(maxlocs_COMBO.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum)-x)^2+...
                                    (ssgt.gty_ss(framenum)-y)^2)^0.5 < success_threshold  && ssgt.gtbuttonfill(framenum) == 1)
                                for kex=k:10
                                    successes_COMBO(kex,framenum,cs_row,cs_col)=...
                                        successes_COMBO(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            if(vardo == 6)
                for t=1:(numel(ssgt.gtx_ss)-part_padding);
                    framenum=part_padding + t;
                    for csix=1:numel(cs_kernel_size_s)
                        [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),csix);
                        for k=1:10
                            x=double(maxlocs_COMBO.column(csix,t,k));
                            y=double(maxlocs_COMBO.row(csix,t,k));
                            if(((ssgt.gtx_ss(framenum-1)-x)^2+...
                                    (ssgt.gty_ss(framenum-1)-y)^2)^0.5 < success_threshold  && ssgt.gtbuttonfill(framenum-1) == 1)
                                for kex=k:10
                                    successes_COMBO_delayed(kex,framenum,cs_row,cs_col)=...
                                        successes_COMBO_delayed(kex,framenum,cs_row,cs_col)+1;
                                end
                                break;
                            end
                        end
                    end
                end
            end
            
            description='rank in top 10, sigma_hw value, pixel2_PR value, cs value, fdsr_lp value';
            
            if(vardo == 1)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_DARK','bg_above','targ_above','targ_total','ssgt','-v7.3');
            elseif(vardo == 2)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_DARK_delayed','ssgt','-v7.3');
            elseif(vardo == 3)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_LIGHT','bg_above','targ_above','targ_total','ssgt','-v7.3');
            elseif(vardo == 4)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_LIGHT_delayed','ssgt','-v7.3');
            elseif(vardo == 5)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_COMBO','bg_above','targ_above','targ_total','ssgt','-v7.3');
            elseif(vardo == 6)
                save(savename,'vid_file','grp_file','description','ispresent',...
                    'cs_kernel_size_s','sigma_hw_top_s','pixel2PR_top_s','fdsr_lp_s',...
                    'successes_COMBO_delayed','ssgt','-v7.3');
            end
        end
    end
end



