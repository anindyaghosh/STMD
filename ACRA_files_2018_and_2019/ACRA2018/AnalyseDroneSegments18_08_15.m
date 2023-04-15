clear all
close all

% Need to load in individual photoreceptor sizes etc
% Calculate segment success ratio for individual photoreceptor/blur
% Then recombine into high-level success structured in terms of videos,
% segments and cs_kernels

doingkalman=false;
if(~doingkalman)
    input_dir='F:\MrBlack\CONFdronefiles\CONF_Analysis18_08_14\';
    output_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_17\';
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
    segment_frame_list=1+floor(segment_time_list*framerate);
    [numsegments,~]=size(segment_time_list);
    segperf=cell(6,1); % One cell for each vardo
    totaldrone=segperf;
    bestperf=segperf;
    best_seg=segperf;
    best_perc=segperf;
    best_overall=segperf;
    best_forpr=segperf;
    
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
            
            figures_root=[figures_dir vidname];
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
                figures_root=[figures_root suffix '\'];
                
                % Where mingt is false, set the successes to zero
                inquestion(:,~mingt,:,:)=0;
                
                [ix1,ix2]=ind2sub(size(pixel2PR_s),prnum);
                % Sum up performance for the prnum for each segment
                for k=1:numsegments
                    % sv has form [rank, 1, [cs_kernel_s]] and stores the
                    % number of successful frames in the segment
                    sv=sum(inquestion(:,segment_frame_list(k,1):segment_frame_list(k,2),:,:),2);
                    segperf{vardo}(:,k,ix1,ix2,:,:)=sv;
                end
                
            end
            % Find best overall performer for each rank
            
            best_overall{vardo}.pixel2PR=zeros(numel(rix_todo),1);
            best_overall{vardo}.sigma_hw=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.cs_kernel_size=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.fdsr_lp=best_overall{vardo}.pixel2PR;
            best_overall{vardo}.refnum=best_overall{vardo}.pixel2PR;
            
            sv=sum(segperf{vardo},2);
            for rix=rix_todo
                [v,I]=max(sv(rix,:));
                [~,~,ix1,ix2,ix3,ix4]=ind2sub(size(sv(rix,:,:,:,:,:)),I);
                best_overall{vardo}.pixel2PR(rix)=pixel2PR_s(ix1,ix2);
                best_overall{vardo}.sigma_hw(rix)=sigma_hw_s(ix1,ix2);
                best_overall{vardo}.cs_kernel_size(rix)=cs_kernel_size_s(ix3,ix4);
                best_overall{vardo}.fdsr_lp(rix)=fdsr_lp_s(ix3,ix4);
                best_overall{vardo}.perf(rix)=v;
                best_overall{vardo}.refnum(rix)=I;
            end
            
            % Find best performing cs_kernel and fdsr_lp for each
            % pixel2PR_s across the entire video
            best_forpr{vardo}.cs_kernel_size=zeros([numel(rix_todo) size(pixel2PR_s)]);
            best_forpr{vardo}.fdsr_lp=best_forpr{vardo}.cs_kernel_size;
            best_forpr{vardo}.perf=best_forpr{vardo}.cs_kernel_size;
            best_forpr{vardo}.refnum=best_forpr{vardo}.cs_kernel_size;
            
            for pix=1:numel(pixel2PR_s)
                [ir,ic]=ind2sub(size(pixel2PR_s),pix);
                sv=sum(segperf{vardo}(:,:,ir,ic,:,:),2);
                for rix=rix_todo
                    [v,I]=max(sv(rix,:));
                    [~,~,~,~,ix3,ix4]=ind2sub(size(sv(rix,:,:,:,:,:)),I);
                    best_forpr{vardo}.cs_kernel_size(rix,pix)=cs_kernel_size_s(ix3,ix4);
                    best_forpr{vardo}.fdsr_lp(rix,pix)=fdsr_lp_s(ix3,ix4);
                    best_forpr{vardo}.perf(rix,pix)=v;
                    best_forpr{vardo}.refnum(rix,pix)=I;
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
                    [v,I]=max(sv(:));
                    [~,~,all_ix1,all_ix2,all_ix3,all_ix4]=ind2sub(size(sv),I);
                    best_seg{vardo}.pixel2PR(rix,k)=pixel2PR_s(all_ix1,all_ix2);
                    best_seg{vardo}.sigma_hw(rix,k)=sigma_hw_s(all_ix1,all_ix2);
                    best_seg{vardo}.cs_kernel_size(rix,k)=cs_kernel_size_s(all_ix3,all_ix4);
                    best_seg{vardo}.fdsr_lp(rix,k)=fdsr_lp_s(all_ix3,all_ix4);
                    best_seg{vardo}.perf(rix,k)=v;
                    best_overall{vardo}.segperf(rix,k)=segperf{vardo}(rix,k,best_overall{vardo}.refnum(rix));
                end
                % Successes frame nums relative to start of video
                % Ground truth frame nums relative to start of video
                totaldrone{vardo}(k)=sum(mingt(segment_frame_list(k,1):segment_frame_list(k,2)) == 1);
            end
            
            best_seg{vardo}.percent=best_seg{vardo}.perf./repmat(totaldrone{vardo},numel(rix_todo),1);
            best_overall{vardo}.percent=best_overall{vardo}.segperf./repmat(totaldrone{vardo},numel(rix_todo),1);
            best_forpr{vardo}.percent=best_forpr{vardo}.perf./repmat(sum(totaldrone{vardo}),[numel(rix_todo) size(pixel2PR_s)]);
            
            %         % Find best overall performer
            %         sv=sum(inquestion,2);
            %         sv=sv(1,:,:,:,:,:);
            %         [v,I]=max(sv(:));
            %         [~,~,all_ix1,all_ix2,all_ix3,all_ix4]=ind2sub(size(sv),I);
            %
            %         vraw=get(vread,'Height');
            %         hraw=get(vread,'Width');
            %         [pixel2PR_s,sigma_hw_s]=meshgrid(unique(a.pixel2PR_s),unique(a.sigma_hw_s));
            %
            %         gtbuttonfill=zeros(1,gt.record_frames(end));
            %         gtxfill=gtbuttonfill;
            %         gtyfill=gtbuttonfill;
            %         for gtix=2:numel(gt.record_frames)
            %             if(gt.gt_button(gtix-1) == 1 && gt.gt_button(gtix) == 1)
            %                 gtbuttonfill(gt.record_frames(gtix-1):gt.record_frames(gtix))=1;
            %
            %                 step=(gt.gt_x(gtix)-gt.gt_x(gtix-1))/(gt.record_frames(gtix)-gt.record_frames(gtix-1));
            %                 step_basis=0:gt.record_frames(gtix)-gt.record_frames(gtix-1);
            %                 gtxfill(gt.record_frames(gtix-1):gt.record_frames(gtix))=gt.gt_x(gtix-1)+step_basis*step;
            %
            %                 step=(gt.gt_y(gtix)-gt.gt_y(gtix-1))/(gt.record_frames(gtix)-gt.record_frames(gtix-1));
            %                 step_basis=0:gt.record_frames(gtix)-gt.record_frames(gtix-1);
            %                 gtyfill(gt.record_frames(gtix-1):gt.record_frames(gtix))=gt.gt_y(gtix-1)+step_basis*step;
            %             end
            %         end
            
            %         % Calculate target velocities
            %         gt_appspeedx=(gtxfill(2:end)-gtxfill(1:end-1))*framerate;
            %         gt_appspeedy=(gtyfill(2:end)-gtyfill(1:end-1))*framerate;
            %         % only consider cases where the target was present on the preceding frame
            %
            %         segment_all_suc=cell(numsegments,1);
            %         segment_spec_suc=segment_all_suc;
            %         getspeeds_x=segment_all_suc;
            %         getspeeds_y=segment_all_suc;
            %         getspeeds_mag=segment_all_suc;
            %         pz=segment_all_suc;
            %
            %         all_perf=zeros(numsegments,1);
            %         spec_perf=all_perf;
            %
            %         pixel2PR=zeros(numsegments,1);
            %         sigma_hw=pixel2PR;
            %         totalpossible=zeros(numsegments,1);
            %
            %         for k=1:numsegments
            %
            %             %     set(vread,'CurrentTime',segment_time_list(k,1))
            %
            %             segment_all_suc{k}=inquestion(1,segment_frame_list(k,1):segment_frame_list(k,2),...
            %                 all_ix1,all_ix2,all_ix3,all_ix4);
            %
            %             all_val=sum(segment_all_suc{k});
            %
            %             % Find best performer for the segment
            %             sv=sum(inquestion(1,segment_frame_list(k,1):segment_frame_list(k,2),:,:,:,:),2);
            %             [v,I]=max(sv(:));
            %             [~,~,ix1,ix2,ix3,ix4]=ind2sub(size(sv),I);
            %
            %             segment_spec_suc{k}=inquestion(1,segment_frame_list(k,1):segment_frame_list(k,2),ix1,ix2,ix3,ix4);
            %             spec_val=sum(segment_spec_suc{k});
            %
            %             totaldrone=sum(gtbuttonfill(segment_frame_list(k,1):segment_frame_list(k,2))==1);
            %
            %             all_perf(k)=all_val/totaldrone;
            %             spec_perf(k)=spec_val/totaldrone;
            %
            %             frame_mask=false(1,gt.record_frames(end));
            %             frame_mask(segment_frame_list(k,1):segment_frame_list(k,2))=true;
            %             use_mask=frame_mask(2:end) & gtbuttonfill(1:end-1) == 1 & gtbuttonfill(2:end) == 1;
            %
            %             getspeeds_x{k}=gt_appspeedx(use_mask);
            %             getspeeds_y{k}=gt_appspeedy(use_mask);
            %             getspeeds_mag{k}=(getspeeds_x{k}.^2+getspeeds_y{k}.^2).^0.5;
            %
            %             pixel2PR(k)=pixel2PR_s(ix1,ix2);
            %             sigma_hw(k)=sigma_hw_s(ix1,ix2);
            %
            %             tempz=inquestion(1,segment_frame_list(k,1):segment_frame_list(k,2),:,:,:,:);
            %             tempz=sum(tempz,2);
            %             tempz=max(tempz,[],5);
            %             tempz=max(tempz,[],6);
            %             %     [~,tempframes,~,~]=size(tempz);
            %             pz{k}=reshape(tempz/totaldrone,size(pixel2PR_s));
            %
            %             totalpossible(k)=totaldrone;
            %
            %         end
            
            %         table(all_perf,spec_perf)
            % Plot performance vs target velocity for each segment
            %
            % div_suc=zeros(4,1);
            % div_total=div_suc;
            % figure(1)
            % for k=1:numsegments
            %     divisions=linspace(min(getspeeds_mag{k}),1+max(getspeeds_mag{k}),5);
            %     for t=1:numel(divisions)-1
            %         div_mask=getspeeds_mag{k} >= divisions(t) & getspeeds_mag{k} < divisions(t+1);
            %         div_suc(t)=sum(segment_spec_suc{k}(div_mask));
            %         div_total(t)=sum(div_mask);
            %     end
            %     plot(0.5*(divisions(1:end-1)+divisions(2:end)),div_suc./div_total,'-')
            %     hold on
            % end
            % For each segment, plot success as function of pixel2PR / sigma_hw
            
            %         % Separate out the top 10. Slot 11 is for missed frames
            %         outrank_spec=cell(10,1);
            %         outrank_all=cell(10,1);
            %         for k=2:10
            %             outrank_spec{k}=inquestion(k,:,ix1,ix2,ix3,ix4)-inquestion(k-1,:,ix1,ix2,ix3,ix4);
            %             outrank_all{k}=inquestion(k,:,all_ix1,all_ix2,all_ix3,all_ix4)-inquestion(k-1,:,all_ix1,all_ix2,all_ix3,all_ix4);
            %         end
            %         outrank_spec{1}=inquestion(1,:,ix1,ix2,ix3,ix4);
            %         outrank_all{1}=inquestion(1,:,all_ix1,all_ix2,all_ix3,all_ix4);
            %
            %         totalmax=0;
            %         pixel_per_degree=get(vread,'width')/84;
            %         degree2PR_s=pixel2PR_s/pixel_per_degree;
            %         degree_hw_s=sigma_hw_s.*pixel2PR_s/pixel_per_degree;
            %         for k=1:numsegments
            %             totalmax=max(totalmax,max(pz{k}(:)));
            %         end
            %
            %         if(drawfigs)
            %             for k=1:numsegments
            %                 figure(k)
            %                 contourf(pixel2PR_s,sigma_hw_s,pz{k})
            %                 caxis([0 totalmax])
            %                 colormap('gray')
            %                 colorbar
            %                 title(['Section ' num2str(k)])
            %                 xlabel('Pixel spacing between photoreceptors')
            %                 ylabel('Blur kernel FWHM in photoreceptor widths')
            %                 saveas(gcf,[figures_root '-segment' num2str(k) '.pdf'],'pdf')
            %
            %                 contourf(degree2PR_s,degree_hw_s,pz{k})
            %                 caxis([0 totalmax])
            %                 colormap('gray')
            %                 colorbar
            %                 title(['Section ' num2str(k)])
            %                 xlabel('Degree spacing between photoreceptors')
            %                 ylabel('Blur kernel FWHM in degrees')
            %                 saveas(gcf,[figures_root '-degrees_segment' num2str(k) '.pdf'],'pdf')
            %
            %             end
            %
            %             figure(numsegments+1)
            %             bar([all_perf spec_perf])
            %             xlabel('Segment number')
            %             ylabel('Successful detection rate')
            %             legend({'Best overall','Best for segment'},'box','off','Location','NorthWest')
            %             saveas(gcf,[figures_root '-bar.pdf'],'pdf')
            %
            %             %         figure(numsegments+2)
            %             %         drawchars={'ro','bo','go','mo','ko','rx','bx','gx','mx','kx','rp','bp','gp','mp','kp'};
            %             %         segnames=cell(numsegments,1);
            %             %         figure
            %             %         for k=1:numsegments
            %             %             p=plot(pixel2PR(k),sigma_hw(k),drawchars{k})
            %             %             set(p,'MarkerSize',6+k*2)
            %             %             hold on
            %             %             segnames{k}=num2str(k);
            %             %         end
            %             %
            %             %         xlim([min(pixel2PR_s(:)) max(pixel2PR_s(:))])
            %             %         ylim([min(sigma_hw_s(:)) max(sigma_hw_s(:))])
            %             %         legend(segnames,'box','off')
            %             %         xlabel('Pixels between photoreceptors')
            %             %         ylabel('Blur kernel FWHM in photoreceptor widths')
            %             %         saveas(gcf,[figures_root '-bestlocations.pdf'],'pdf')
            %             %         hold off
            %
            %             % Parameters
            %             figure(numsegments+3)
            %             subplot(2,1,1)
            %             bar(pixel2PR)
            %             ylim([0 max(pixel2PR_s(:))])
            %             ylabel('Photoreceptor spacing')
            %             subplot(2,1,2)
            %             bar(sigma_hw)
            %             ylim([0 max(sigma_hw_s(:))])
            %             xlabel('Segment number')
            %             ylabel('blur fwhm')
            %             saveas(gcf,[figures_root '-paramsettings.pdf'],'pdf')
            %
            %             % Parameters in degrees
            %             figure(numsegments+3)
            %             subplot(2,1,1)
            %             bar(pixel2PR/pixel_per_degree)
            %             ylim([0 max(pixel2PR_s(:)/pixel_per_degree)])
            %             ylabel('Photoreceptor spacing (degrees)')
            %             subplot(2,1,2)
            %             bar(sigma_hw.*pixel2PR/pixel_per_degree)
            %             ylim([0 max(sigma_hw_s(:).*pixel2PR_s(:)/pixel_per_degree)])
            %             xlabel('Segment number')
            %             ylabel('blur fwhm (degrees)')
            %             saveas(gcf,[figures_root '-degree_paramsettings.pdf'],'pdf')
            %         end
            %         %Per segment histograms of rank
            %         seg_cu{vix,vardo}=zeros(numsegments,11);
            %         for k=1:numsegments
            %             hy_all=zeros(11,1);
            %             %             leg_entry=cell(10,1);
            %             cu_suc=zeros(11,1);
            %             for t=1:10
            %                 hy_all(t)=sum(outrank_all{t}(segment_frame_list(k,1):segment_frame_list(k,2))); % the number of successes at this rank
            %                 cu_suc(t)=100*sum(inquestion(t,segment_frame_list(k,1):segment_frame_list(k,2),all_ix1,all_ix2,all_ix3,all_ix4))/totalpossible(k);
            %                 %                 leg_entry{t}=num2str(cu_suc(t));
            %             end
            %             hy_all(11)=totalpossible(k) - sum(inquestion(10,segment_frame_list(k,1):segment_frame_list(k,2),all_ix1,all_ix2,all_ix3,all_ix4),2);
            %
            %
            %             hy_all=100*hy_all/totalpossible(k);
            %             cu_suc(11)=sum(hy_all);
            %             seg_cu{vix,vardo}(k,:)=cu_suc;
            %
            %             % Normalise to a percentage
            %             %             hy_all = hy_all/sum(hy_all)*100;
            %             if(true)
            %                 figure
            %                 bar([hy_all])
            %                 hold on
            %                 plot(cu_suc,'x--')
            %                 hold off
            %                 xl=get(gca,'XTickLabel');
            %                 xl{11}='>10';
            %                 set(gca,'XTickLabel',xl)
            %                 xlabel('Rank')
            %                 ylabel('Frames %')
            %                 ylim([0 100])
            %                 legend({'Successes at rank','Cumulative successes'},'Box','off')
            %                 %             saveas(gcf,['D:\Johnstuff\Matlab\Data\Drone Footage\Histograms\' vidname '-v' num2str(vardo) '-seg' num2str(k) '-all.pdf'],'pdf')
            %                 saveas(gcf,[figures_root '-histogram_cu-seg' num2str(k) '-all.pdf'],'pdf')
            %
            %                 bar([hy_all(1:10)])
            %                 hold on
            %                 plot(cu_suc(1:10),'x--')
            %                 hold off
            %                 ylim([0 100])
            %                 xlabel('Rank')
            %                 ylabel('Frames %')
            %                 legend({'Successes at rank','Cumulative successes'},'Box','off','Location','Northwest')
            %                 saveas(gcf,[figures_root '-histogram_cu_noremainder-seg' num2str(k) '-all.pdf'],'pdf')
            %             end
            %
            %         end
            %         perf_record{vix,vardo,1}=all_perf;
            %         perf_record{vix,vardo,2}=spec_perf;
            
        end
        save(savename,'best_overall','best_seg','best_forpr','vidname','mingt','segment_time_list','segment_frame_list')
    end
end
return
%%
%Script for generating single plot with all segments at all ranks
close all

%Collisions
figure(1)
px_LIGHT=[seg_cu{5,3}(12:14,:);seg_cu{2,3}([5 7 8 9],:)];
plot(px_LIGHT')
ylim([0 100])
xlabel('Rank')
ylabel('Frames %')
title('Collisions')
[rows,~]=size(px_LIGHT);
legend({num2str((1:rows)')},'box','off','Location','Northwest')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Collisions_ranks.pdf','pdf')

% Flyby6m
figure(2)
px_LIGHT=seg_cu{5,3}(1:11,:);
[rows,~]=size(px_LIGHT);
for t=1:rows
    if(t<8)
        drawchar='-';
    else
        drawchar='--';
    end
    plot(px_LIGHT(t,:),drawchar)
    hold on
end
ylim([0 100])
xlabel('Rank')
ylabel('Frames %')
title('Flyby6m')
[rows,~]=size(px_LIGHT);
legend({num2str((1:rows)')},'box','off','Location','Northwest')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m_ranks.pdf','pdf')

% Flyby12m
figure(3)
px_LIGHT=seg_cu{4,3}(1:11,:);
[rows,~]=size(px_LIGHT);
for t=1:rows
    if(t<8)
        drawchar='-';
    else
        drawchar='--';
    end
    plot(px_LIGHT(t,:),drawchar)
    hold on
end
hold off
ylim([0 100])
xlabel('Rank')
ylabel('Frames %')
title('Flyby12m')
legend({num2str((1:rows)')},'box','off','Location','Northwest')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby12m_ranks.pdf','pdf')

%Dogfight
figure(4)
px_LIGHT=[seg_cu{1,3}(1:7,:);seg_cu{3,3}(1:21,:)];
plot(px_LIGHT')
[rows,~]=size(px_LIGHT);
for t=1:rows
    if(t<8)
        drawchar='-';
    elseif(t<16)
        drawchar='--';
    elseif(t<24)
        drawchar=':';
    else
        drawchar='-.';
    end
    plot(px_LIGHT(t,:),drawchar)
    hold on
end
ylim([0 100])
xlabel('Rank')
ylabel('Frames %')
title('Dogfight')
[rows,~]=size(px_LIGHT);
legend({num2str((1:rows)')},'box','off','Location','Northwest')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Dogfight_ranks.pdf','pdf')

%%
% script for generating bar graphs of different categories
close all

% Collisions
px_all_DARK=100*[perf_record{5,1,1}(12:14);perf_record{2,1,1}([5 7 8 9])];
px_best_DARK=100*[perf_record{5,1,2}(12:14);perf_record{2,1,2}([5 7 8 9])];
px_all_LIGHT=100*[perf_record{5,3,1}(12:14);perf_record{2,3,1}([5 7 8 9])];
px_best_LIGHT=100*[perf_record{5,3,2}(12:14);perf_record{2,3,2}([5 7 8 9])];
figure(1)
bar([px_all_DARK px_best_DARK px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Collisions')
legend({'Dark selective best overall','Dark selective best for segment','Light selective best overall','Light selective best for segment'},'Box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Collisions bar - COMBINED.pdf','pdf')

figure(2)
bar([px_all_DARK px_best_DARK])
ylim([0 100])
title('Collisions - Dark selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Collisions bar - DARK.pdf','pdf')

figure(3)
bar([px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Collisions - Light selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Collisions bar - LIGHT.pdf','pdf')

%%

close all
% Fly-bys at 6m
% These are taken from stationary 1, vix =5
px_all_DARK=100*[perf_record{5,1,1}(1:11)];
px_best_DARK=100*[perf_record{5,1,2}(1:11)];
px_all_LIGHT=100*[perf_record{5,3,1}(1:11)];
px_best_LIGHT=100*[perf_record{5,3,2}(1:11)];
figure(1)
bar([px_all_DARK px_best_DARK px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Fly-bys at 6m ')
legend({'Dark selective best overall','Dark selective best for segment','Light selective best overall','Light selective best for segment'},'Box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m bar - COMBINED.pdf','pdf')

figure(2)
bar([px_all_DARK px_best_DARK])
ylim([0 100])
title('Fly-bys at 6m  - Dark selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m bar - DARK.pdf','pdf')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m bar - DARK.fig','fig')

figure(3)
bar([px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Fly-bys at 6m - Light selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m bar - LIGHT.pdf','pdf')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby6m bar - LIGHT.fig','fig')

%%
% Fly-bys at 12 m
% These are from Stationary 2, vix=4
close all
px_all_DARK=100*[perf_record{4,1,1}(1:11)];
px_best_DARK=100*[perf_record{4,1,2}(1:11)];
px_all_LIGHT=100*[perf_record{4,3,1}(1:11)];
px_best_LIGHT=100*[perf_record{4,3,2}(1:11)];
ylim([0 100])
figure(1)
bar([px_all_DARK px_best_DARK px_all_LIGHT px_best_LIGHT])
title('Fly-bys at 12m ')
legend({'Dark selective best overall','Dark selective best for segment','Light selective best overall','Light selective best for segment'},'Box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby12m bar - COMBINED.pdf','pdf')

figure(2)
bar([px_all_DARK px_best_DARK])
ylim([0 100])
title('Fly-bys at 12m  - Dark selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby12m bar - DARK.pdf','pdf')

figure(3)
bar([px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Fly-bys at 12m - Light selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Flyby12m bar - LIGHT.pdf','pdf')

%%
% Dogfight
% Take all from 2018____ (vix=1) and then all from Pursuit2 (vix=3)
close all
px_all_DARK=100*[perf_record{1,1,1}(1:7);perf_record{3,1,1}(1:21)];
px_best_DARK=100*[perf_record{1,1,2}(1:7);perf_record{3,1,2}(1:21)];
px_all_LIGHT=100*[perf_record{1,3,1}(1:7);perf_record{3,3,1}(1:21)];
px_best_LIGHT=100*[perf_record{1,3,2}(1:7);perf_record{3,3,2}(1:21)];
figure(1)
bar([px_all_DARK px_best_DARK px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Dogfights')
legend({'Dark selective best overall','Dark selective best for segment','Light selective best overall','Light selective best for segment'},'Box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Dogfights bar - COMBINED.pdf','pdf')
set(gca,'XTick',1:numel(px_all_DARK))

figure(2)
bar([px_all_DARK px_best_DARK])
ylim([0 100])
title('Dogfights - Dark selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Dogfights bar - DARK.pdf','pdf')
set(gca,'XTick',1:numel(px_all_DARK))

figure(3)
bar([px_all_LIGHT px_best_LIGHT])
ylim([0 100])
title('Dogfights - Light selective')
legend({'Best overall','Best for segment'},'box','off')
xlabel('Segment number')
ylabel('Proportion of frames successful')
saveas(gcf,'D:\Johnstuff\Matlab\Data\Drone Footage\Dogfights bar - LIGHT.pdf','pdf')
set(gca,'XTick',1:numel(px_all_DARK))