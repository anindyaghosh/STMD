clear all
close all

% Need to load in individual photoreceptor sizes etc
% Calculate segment success ratio for individual photoreceptor/blur
% Then recombine into high-level success structured in terms of videos,
% segments and cs_kernels

doingkalman=false;
if(~doingkalman)
    input_dir='F:\MrBlack\CONFdronefiles\CONF_Analysis18_08_14\';
    output_dir='C:\Users\John\Desktop\PhD\ConfDronefiles\Collapse_temp_18_09_10\';
    figures_dir='F:\MrBlack\CONFdronefiles\autofigures\';
    rix_todo=1:10;
else
    input_dir='F:\MrBlack\CONFdronefiles\CONF_AnalysisKalman18_08_19\';
    output_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_19_kalman\';
    figures_dir='F:\MrBlack\CONFdronefiles\autofigures_kalman\';
    rix_todo=1;
end

if(~exist(output_dir,'dir'))
    mkdir(output_dir)
end
if(~exist(figures_dir,'dir'))
    mkdir(figures_dir)
end

vid_dir='F:\Drone footage\MJ2 Format\';
% figures_dir='E:\PHD\CONFdronefiles\autofigures\';


vidname_list={'2018_05_31_13_48_46';...
    'Pursuit1';...
    'Pursuit2';...
    'Stationary1';...
    'Stationary2'};

vidnum_list=[2;3;4;5;6];

drawfigs=false;

perf_record=cell(numel(vidname_list),4,2);
seg_cu=cell(numel(vidname_list),4);

% The photoreceptor and blur values

for vix=1:numel(vidname_list)
    vidname=vidname_list{vix};
    vidnum=vidnum_list(vix);
    
    a=load([input_dir 'v' num2str(vidnum) '-p1-v1.mat']);
    pixel2PR_s=a.pixel2PR_top_s;
    sigma_hw_s=a.sigma_hw_top_s;
    cs_kernel_size_s=a.cs_kernel_size_s;
    fdsr_lp_s=a.fdsr_lp_s;
    
    if(strcmp(vidname,'2018_05_31_13_48_46'))
        segment_time_list=[...
            15 24;...
            25 29;...
            30 34;...
            35 41;...
            50 53;...
            55 70;...
            94 103];
    elseif(strcmp(vidname,'Pursuit1'))
        segment_time_list=[...
            35 37;...
            40 42;...
            47 47;...
            49 50;...
            53 54;...
            61 61;...
            73 77;...
            85 89;...
            97 109;...
            4*60+22 4*60+24;...
            4*60+32 4*60+34;...
            4*60+35 4*60+42;...
            4*60+42 5*60+17];
    elseif(strcmp(vidname,'Pursuit2'))
        segment_time_list=[...
            3 5;...
            6 8;...
            14 20;...
            23 25;...
            26 38;...
            40 44;...
            49 51;...
            55 70;...
            73 77;...
            81 85;...
            89 99;...
            104 116;...
            122 139;...
            140 144;...
            149 154;...
            159 167;...
            172 180;...
            183 185;...
            196 202;...
            210 213;...
            266 268];
    elseif(strcmp(vidname,'Stationary1'))
        segment_time_list=[...
            0 29;...
            36 43;...
            44 47;...
            50 54;...
            56 58;...
            70 84;...
            89 100;...
            103 113;...
            116 120;...
            139 150;...
            151 165;...
            204 217;...
            180+58 4*60+8;...
            4*60+34 4*60+39];
    elseif(strcmp(vidname,'Stationary2'))
        segment_time_list=[
            3 10;...
            63 67;...
            78 84;...
            91 97;...
            102 105;...
            109 112;...
            119 137;...
            139 157;...
            160 163;...
            165 171;...
            175 178];
    end
    
    segment_time_list(:,2)=segment_time_list(:,2)+0.999;
    
    vread=VideoReader([vid_dir vidname '.mj2']);
    framerate=get(vread,'Framerate');
    vraw=get(vread,'Height');
    hraw=get(vread,'Width');
    
    segment_frame_list=1+floor(segment_time_list*framerate);
    [numsegments,~]=size(segment_time_list);
    segperf=cell(6,1); % One cell for each vardo
    segperf_adjusted=segperf;
    totaldrone=segperf;
    bestperf=segperf;
    best_seg=segperf;
    best_perc=segperf;
    best_overall=segperf;
    best_forpr=segperf;
    best_forpr_forseg=segperf;
    
    for k=1:6
        segperf{k}=zeros([numel(rix_todo) numsegments size(pixel2PR_s) size(cs_kernel_size_s)]);
    end

    savename=[output_dir 'vid' num2str(vidnum) '.mat'];
    if(exist(savename,'file'))
        disp(['Skipping ' savename])
    else
        for vardo=[1 3 5]%1:6%[1 3 5]%1:4
            
            close all
            % The number of frames that the drone appears on varies based
            % on the photoreceptor spacing. So, work out a mask indicating
            % when the drone is on the screen for every variant
            a=load([input_dir 'v' num2str(vidnum) '-p1' '-v' num2str(vardo) '.mat']);
            mingt=true(size(a.ssgt.gtbuttonfill));
            for prnum=1:numel(pixel2PR_s)
                a=load([input_dir 'v' num2str(vidnum) '-p' num2str(prnum) '-v' num2str(1) '.mat']);
                mingt=a.ssgt.gtbuttonfill == 1 & mingt;
            end
            % Pad out mingt to be the same length as successes
            [~,suclen,~,~]=size(a.successes_DARK);
            mingt=[mingt false(1,suclen-numel(mingt))];
            
            % The data structure to store the performance for each segment
            
%             figures_root=[figures_dir vidname];
            field_size=zeros(numel(pixel2PR_s),2);
            
            base_probability=zeros(numel(pixel2PR_s),1);
            field_pixels=base_probability;

            % Chance at each rank and prnum for the target to be guessed randomly
            rix_probability=zeros(numel(rix_todo),numel(pixel2PR_s));
            
            % Expected # of correct guesses per rank, prnum and segment
            base_expected=zeros(numel(rix_todo),numel(pixel2PR_s),numsegments);
            
            numframes_inseg=zeros(numel(pixel2PR_s),k);
            
            for prnum=1:numel(pixel2PR_s)
                % For each photoreceptor/blur, work out the performance for each
                % segment
                a=load([input_dir 'v' num2str(vidnum) '-p' num2str(prnum) '-v' num2str(vardo) '.mat']);
                if(vardo == 1)
                    inquestion=a.successes_DARK;
                    suffix='_DARK';
                elseif(vardo == 2)
                    inquestion=a.successes_DARK_delayed;
                    suffix='_DARK_DELAYED';
                elseif(vardo == 3)
                    inquestion=a.successes_LIGHT;
                    suffix='_LIGHT';
                elseif(vardo == 4)
                    inquestion=a.successes_LIGHT_delayed;
                    suffix='_LIGHT_DELAYED';
                elseif(vardo == 5)
                    inquestion=a.successes_COMBO;
                    suffix='_COMBO';
                elseif(vardo == 6)
                    inquestion=a.successes_COMBO_delayed;
                    suffix='_COMBO_DELAYED';
                end
                [~,savelen,~,~]=size(inquestion);
%                 figures_root=[figures_root suffix '\'];
                
                % Where mingt is false, set the successes to zero
                inquestion(:,~mingt,:,:)=0;
                
                % Calculate expected correct guesses if choosing randomly
                pixel2PR=pixel2PR_s(prnum);
                sigma_hw_pixel=sigma_hw_s(prnum);
               
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
                
                field_size(prnum,:)=[vfov,hfov];
                field_pixels(prnum)=field_size(prnum,1)*field_size(prnum,2);
                base_probability(prnum)=9./field_pixels(prnum);
                
                for rix=rix_todo
                    rix_probability(rix,prnum)=1-(1-base_probability(prnum))^rix;
                end
                
                [ix1,ix2]=ind2sub(size(pixel2PR_s),prnum);
                % Sum up performance for the prnum for each segment
                for k=1:numsegments
                    % sv has form [rank, 1, [cs_kernel_s]] and stores the
                    % number of successful frames in the segment
                    sv=sum(inquestion(:,segment_frame_list(k,1):segment_frame_list(k,2),:,:),2);
                    numframes_inseg(prnum,k)=sum(mingt(segment_frame_list(k,1):segment_frame_list(k,2)));
                    base_expected(:,prnum,k)=rix_probability(:,prnum)*numframes_inseg(prnum,k);
                    % Subtract number of expected correct guesses
                    sv_adjust=sv-repmat(base_expected(:,prnum,k),[1 1 size(cs_kernel_size_s)]);
                    
                    segperf{vardo}(:,k,ix1,ix2,:,:)=sv;
                    segperf_adjusted{vardo}(:,k,ix1,ix2,:,:)=sv_adjust;
                end
            end
            % Find best overall performer for each rank
            
            best_overall{vardo}.pixel2PR=zeros(numel(rix_todo),1);
            best_overall{vardo}.sigma_hw=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.cs_kernel_size=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.fdsr_lp=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.refnum=best_overall{vardo}.pixel2PR;
            
            sv=sum(segperf{vardo},2);
            sv_adjust=sum(segperf_adjusted{vardo},2);
            for rix=rix_todo
                % Make decision about which is best based on successes above
                % the expectation for random guessing. However, record the
                % actual total number of successes.
                [~,I]=max(sv_adjust(rix,:));
                [~,~,ix1,ix2,ix3,ix4]=ind2sub(size(sv(rix,:,:,:,:,:)),I);
                best_overall{vardo}.pixel2PR(rix)=pixel2PR_s(ix1,ix2);
                best_overall{vardo}.sigma_hw(rix)=sigma_hw_s(ix1,ix2);
                best_overall{vardo}.cs_kernel_size(rix)=cs_kernel_size_s(ix3,ix4);
                best_overall{vardo}.fdsr_lp(rix)=fdsr_lp_s(ix3,ix4);
                best_overall{vardo}.perf(rix)=sv(rix,I);
                best_overall{vardo}.refnum(rix)=I;
                best_overall{vardo}.prnum(rix)=sub2ind(size(pixel2PR_s),ix1,ix2);
                best_overall{vardo}.csnum(rix)=sub2ind(size(cs_kernel_size_s),ix3,ix4);
            end
            
            % Find best performing cs_kernel and fdsr_lp for each
            % pixel2PR_s across the entire video
            best_forpr{vardo}.cs_kernel_size=zeros([numel(rix_todo) size(pixel2PR_s)]);
            best_forpr{vardo}.fdsr_lp=best_forpr{vardo}.cs_kernel_size;
            best_forpr{vardo}.perf=best_forpr{vardo}.cs_kernel_size;
            best_forpr{vardo}.refnum=best_forpr{vardo}.cs_kernel_size;
            best_forpr{vardo}.segperf=zeros([numel(rix_todo) numsegments size(pixel2PR_s)]);
            
            for pix=1:numel(pixel2PR_s)
                [ir,ic]=ind2sub(size(pixel2PR_s),pix);
                sv=sum(segperf{vardo}(:,:,ir,ic,:,:),2);
                sv_adjust=sum(segperf_adjusted{vardo}(:,:,ir,ic,:,:),2);
                for rix=rix_todo
                    [~,I]=max(sv_adjust(rix,:));
                    [~,~,~,~,ix3,ix4]=ind2sub(size(sv(rix,:,:,:,:,:)),I);
                    best_forpr{vardo}.cs_kernel_size(rix,pix)=cs_kernel_size_s(ix3,ix4);
                    best_forpr{vardo}.fdsr_lp(rix,pix)=fdsr_lp_s(ix3,ix4);
                    best_forpr{vardo}.perf(rix,pix)=sv(rix,I);
                    best_forpr{vardo}.refnum(rix,pix)=I;
                    best_forpr{vardo}.segperf=segperf{vardo}(rix,:,ir,ic,ix3,ix4);
                end
            end
            
            % pixel2PR_s for each segment
            best_forpr_forseg{vardo}.cs_kernel_size=zeros([numel(rix_todo) numsegments size(pixel2PR_s)]);
            best_forpr_forseg{vardo}.fdsr_lp=best_forpr{vardo}.cs_kernel_size;
            best_forpr_forseg{vardo}.perf=best_forpr{vardo}.cs_kernel_size;
            best_forpr_forseg{vardo}.refnum=best_forpr{vardo}.cs_kernel_size;
            best_forpr_forseg{vardo}.segperf=zeros([numel(rix_todo) numsegments size(pixel2PR_s)]);
            
            for k=1:numsegments
                for pix=1:numel(pixel2PR_s)
                    [ir,ic]=ind2sub(size(pixel2PR_s),pix);
                    sv=segperf{vardo}(:,k,ir,ic,:,:);
                    sv_adjust=segperf_adjusted{vardo}(:,k,ir,ic,:,:);
                    for rix=rix_todo
                        [~,I]=max(sv_adjust(rix,:));
                        [~,~,~,~,ix3,ix4]=ind2sub(size(sv(rix,:,:,:,:,:)),I);
                        best_forpr_forseg{vardo}.cs_kernel_size(rix,k,pix)=cs_kernel_size_s(ix3,ix4);
                        best_forpr_forseg{vardo}.fdsr_lp(rix,k,pix)=fdsr_lp_s(ix3,ix4);
                        best_forpr_forseg{vardo}.perf(rix,k,pix)=sv(rix,I);
                        best_forpr_forseg{vardo}.refnum(rix,k,pix)=I;
                        best_forpr_forseg{vardo}.segperf(rix,k,pix)=segperf{vardo}(rix,k,ir,ic,ix3,ix4);
                    end
                end
            end
            
            % Find best performer within each segment for each rank
            best_seg{vardo}.pixel2PR=zeros(numel(rix_todo),numsegments);
            best_seg{vardo}.sigma_hw=best_seg{vardo}.pixel2PR;
            best_seg{vardo}.cs_kernel_size=best_seg{vardo}.pixel2PR;
            best_seg{vardo}.fdsr_lp=best_seg{vardo}.pixel2PR;
            best_seg{vardo}.perf=best_seg{vardo}.pixel2PR;
            for k=1:numsegments
                for rix=rix_todo
                    sv=segperf{vardo}(rix,k,:,:,:,:);
                    sv_adjust=segperf_adjusted{vardo}(rix,k,:,:,:,:);
                    [~,I]=max(sv_adjust(:));
                    [~,~,all_ix1,all_ix2,all_ix3,all_ix4]=ind2sub(size(sv),I);
                    best_seg{vardo}.pixel2PR(rix,k)=pixel2PR_s(all_ix1,all_ix2);
                    best_seg{vardo}.sigma_hw(rix,k)=sigma_hw_s(all_ix1,all_ix2);
                    best_seg{vardo}.cs_kernel_size(rix,k)=cs_kernel_size_s(all_ix3,all_ix4);
                    best_seg{vardo}.fdsr_lp(rix,k)=fdsr_lp_s(all_ix3,all_ix4);
                    best_seg{vardo}.perf(rix,k)=sv(I);
                    best_overall{vardo}.segperf(rix,k)=segperf{vardo}(rix,k,best_overall{vardo}.refnum(rix));
                    
                end
                % Successes frame nums relative to start of video
                % Ground truth frame nums relative to start of video
                totaldrone{vardo}(k)=sum(mingt(segment_frame_list(k,1):segment_frame_list(k,2)) == 1);
            end
            
            best_seg{vardo}.percent=best_seg{vardo}.perf./repmat(totaldrone{vardo},numel(rix_todo),1);
            best_overall{vardo}.percent=best_overall{vardo}.segperf./repmat(totaldrone{vardo},numel(rix_todo),1);
            best_forpr{vardo}.percent=best_forpr{vardo}.perf./repmat(sum(totaldrone{vardo}),[numel(rix_todo) size(pixel2PR_s)]);
            
            % For best performer overall, find relationship between
            % top1 successes and target apparent velocity
            
            % Start by loading the successes again
            prnum=best_overall{vardo}.prnum(1);
            a=load([input_dir 'v' num2str(vidnum) '-p' num2str(prnum) '-v' num2str(vardo) '.mat']);
            if(vardo == 1)
                inquestion=a.successes_DARK;
                suffix='_DARK';
            elseif(vardo == 2)
                inquestion=a.successes_DARK_delayed;
                suffix='_DARK_DELAYED';
            elseif(vardo == 3)
                inquestion=a.successes_LIGHT;
                suffix='_LIGHT';
            elseif(vardo == 4)
                inquestion=a.successes_LIGHT_delayed;
                suffix='_LIGHT_DELAYED';
            elseif(vardo == 5)
                inquestion=a.successes_COMBO;
                suffix='_COMBO';
            elseif(vardo == 6)
                inquestion=a.successes_COMBO_delayed;
                suffix='_COMBO_DELAYED';
            end
            

            rgt=load(['F:\MrBlack\CONFdronefiles\ext_rect_gtruth\' vidname '_rectfill_gt.mat']);
            
            dt=1/framerate; % Delta time between frames
            gtx=a.ssgt.gtxfill;
            gty=a.ssgt.gtyfill;
            app_speed_x=[0 (gtx(2:end)-gtx(1:end-1))/dt];
            app_speed_y=[0 (gtx(2:end)-gtx(1:end-1))/dt];
            % Exclude any frame where the target is not present or just appeared
            app_speed_excmask= ~[false a.ssgt.gtbuttonfill(2:end)];
            % Exclude any frame where the size is not filled in
            app_size_excmask= (~rgt.gt_fill(1:end))';
            veldist_excmask = app_speed_excmask | app_size_excmask;
            
            app_speed_x(veldist_excmask)=NaN;
            app_speed_y(veldist_excmask)=NaN;
            app_speed_mag=(app_speed_x.^2 + app_speed_y.^2).^0.5;

            % Adjust into units of screen width / s
            app_speed_mag = app_speed_mag / hraw;
            
            % Get diag size
            diag_fill=rgt.diag_fill';
            diag_fill(veldist_excmask)=NaN;
   
            % Pad out to same size as successes minus 1
            app_speed_mag=[app_speed_mag NaN(1,savelen-numel(app_speed_mag))];
            diag_fill=[diag_fill NaN(1,savelen-numel(diag_fill))];

            edges_vel=linspace(0,1,20);
            edges_size=[linspace(0,150,4) 300];
            
            velsize_suc_count=zeros(numel(rix_todo),numsegments,numel(edges_vel)-1,numel(edges_size)-1);
            
            num_in_bin=zeros(numel(rix_todo),numsegments,numel(edges_vel)-1,numel(edges_size)-1);

            for rix=rix_todo
                [cs_row,cs_col]=ind2sub(size(cs_kernel_size_s),best_overall{vardo}.csnum(rix));
                best_suc=inquestion(rix,2:end,cs_row,cs_col);
                best_suc=best_suc(:);
                
                for k=1:numsegments
                    segment_suc=best_suc(segment_frame_list(k,1):segment_frame_list(k,2));
                    
                    for vel_ix=1:numel(edges_vel)-1
                        for size_ix=1:numel(edges_size)-1
                            bin_speed=app_speed_mag(segment_frame_list(k,1):segment_frame_list(k,2));
                            bin_size=diag_fill(segment_frame_list(k,1):segment_frame_list(k,2));
                            bin_gt=a.ssgt.gtbuttonfill(segment_frame_list(k,1):segment_frame_list(k,2));
                            bin_mask = bin_speed >= edges_vel(vel_ix) &...
                                bin_speed < edges_vel(vel_ix+1) &...
                                bin_size >= edges_size(size_ix) &...
                                bin_size < edges_size(size_ix+1);
%                             num_in_bin(rix,k,vel_ix,size_ix)=sum(bin_mask(:));
                            num_in_bin(rix,k,vel_ix,size_ix)=sum(bin_gt == 1 & bin_mask);
                            velsize_suc_count(rix,k,vel_ix,size_ix)=sum(segment_suc(bin_mask(:)));
                        end
                    end
                end
            end
            velsize_suc_rate=velsize_suc_count./num_in_bin;
            velsize_suc_weighted=sum(velsize_suc_count,2)./sum(num_in_bin,2);
            velsize_suc_weighted=reshape(velsize_suc_weighted,numel(rix_todo),numel(edges_vel)-1,numel(edges_size)-1);
            
            best_overall{vardo}.velsize_suc_count = velsize_suc_count;
            best_overall{vardo}.num_in_bin=num_in_bin;
            best_overall{vardo}.edges_vel=edges_vel;
            best_overall{vardo}.edges_size=edges_size;
            best_overall{vardo}.app_speed_mag=app_speed_mag;
            best_overall{vardo}.diag_fill=diag_fill;
            best_overall{vardo}.velsize_suc_weighted=velsize_suc_weighted;
            best_overall{vardo}.velsize_suc_rate=velsize_suc_rate;
%             
%             plot(edges(1:end-1),reshape(app_speed_suc_rate(1,:),14,19),'x-');ylim([0 1])
%             xlabel('Screen width / s')
%             ylabel('Proportion successful')
        end
        
        save(savename,'best_overall','best_seg','best_forpr','best_forpr_forseg','vidname','mingt','segment_time_list','segment_frame_list','base_probability','field_pixels','numframes_inseg')
    end
end