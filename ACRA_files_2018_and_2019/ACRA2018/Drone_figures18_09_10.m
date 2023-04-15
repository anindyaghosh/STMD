clear all
close all

drawsegs=false;

doingkalman=false;
if(~doingkalman)
    load_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_09_14\';
    out_dir='C:\Users\John\Desktop\PhD\ConfDronefiles\dump\';
else
    load_dir='E:\PHD\CONFdronefiles\CollapsedAnalysis18_08_19_kalman\';
    out_dir='E:\PHD\CONFdronefiles\autofigures_kalman\';
end

detect_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_09_14\';
track_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_19_kalman\';

% This file contains masks identifying the low, medium and high performing segments
load('F:\MrBlack\CONFdronefiles\perf_masks18_08_23.mat');

vidname_list={'2018_05_31_13_48_46';...
    'Pursuit1';...
    'Pursuit2';...
    'Stationary1';...
    'Stationary2'};

vidnum_list=[2;3;4;5;6];

a=load('F:\MrBlack\CONFdronefiles\CONF_Analysis18_08_14\v1-p1-v1.mat');
pixel2PR_s=a.pixel2PR_top_s;
sigma_hw_s=a.sigma_hw_top_s;

% Numbers for the different types
sortbytype='bg_vel';
if(strcmp(sortbytype,'categories'))
    collisions=cat(1,[repmat(5,3,1) (12:14)'],[repmat(3,4,1) [5 7 8 9]']);
    dogfights=cat(1,[repmat(5,7,1) (1:7)'],[repmat(4,21,1) (1:21)']);
    flyby6m=[repmat(5,11,1) (1:11)'];
    flyby12m=[repmat(6,11,1) (1:11)'];
    unused=[repmat(3,9,1) [(1:4)';(9:13)']];
    
    cat_ix={collisions;dogfights;flyby6m;flyby12m;unused};
    cat_labels={'Collisions','Dogfights','Fly-by at 6m','Fly-by at 12m','unused'};
elseif(strcmp(sortbytype,'bg_vel'))
    %     low_speed_vnum=[5	5	5	5	5	5	5	5	5	3	4	3	4	4	5	5	5	5	5	5	5	5	5	5	5	5	3	2	4	5	5	5	5	4	4	4	4];
    %     low_speed_segnum=[1	12	13	1	2	3	4	7	8	12	20	9	6	16	2	4	6	7	8	10	11	14	5	6	9	10	5	5	15	3	5	9	11	10	18	19	21];
    %low_speed_vnum=[5	6	6	6	3	6	5	5	6	6	4	4	4	3	5	5	6	6	3	2	5	5	5	6	6	4	5	5	5	5	5	4	4	5	6	4	4];
    %low_speed_segnum=[1	1	3	4	12	2	12	13	7	8	20	6	16	9	2	4	5	6	5	5	6	7	8	9	10	15	14	10	11	3	5	18	21	9	11	10	19];
    
    low_speed_vnum=[5	6	6	6	3	5	5	6	6	3	5	5	6	5	5	4	4	4	4	3	2	6	6	4	5	5	5	6	6	4	5	6	4	4	5	5	5];
    low_speed_segnum=[1	1	3	4	12	2	4	5	6	5	3	5	2	12	13	6	16	18	21	9	5	7	8	20	6	7	8	9	10	15	9	11	10	19	10	11	14];
    
    low_speed=[low_speed_vnum' low_speed_segnum'];
    
    % Within low_speed, the distance categories
    low_speed_cats={1:12,13:21,21:37};
    med_speed_cats={38:41,42:53,54:56};
    high_speed_cats={[57 58],59:63,64};
    all_subcats={low_speed_cats;med_speed_cats;high_speed_cats};
    % Scratched 7
    %mod_speed_vnum=[4	4	4	2	4	2	4	4	3	2	2	2	3	3	3	4	2	4	3];
    %mod_speed_segnum=[9	13	12	7	7	1	5	17	2	4	2	3	13	7	8	8	6	14	3];
    
    mod_speed_vnum=[2	2	2	3	4	4	4	3	2	3	4	4	3	3	2	2	4	4];
    mod_speed_segnum=[1	2	3	3	9	13	17	2	4	13	8	12	7	8	6	7	7	14];
    
    mod_speed=[mod_speed_vnum' mod_speed_segnum'];
    
    % Scratched 3
    %     high_speed_vnum=[4	3	3	3	3	4	4	4];
    %     high_speed_segnum=[11	4	10	1	11	2	3	4];
    
    high_speed_vnum=[4	3		3	3	3	4	4	4];
    high_speed_segnum=[4	1		4	10	11	2	3	11];
    
    
    high_speed=[high_speed_vnum' high_speed_segnum'];
    
    cat_ix={low_speed;mod_speed;high_speed};
    cat_labels={'low speed','mod speed','high speed'};
elseif(strcmp(sortbytype,'distance'))
    low_dist_vnum=[5	6	6	6	3	2	5	5	6	6	3	2	2	4	5	5	3	3];
    low_dist_segnum=[1	1	3	4	12	1	2	4	5	6	5	2	3	4	3	5	3	1];

    mod_dist_vnum=[3	6	4	3	3	4	4	4	4	4	4	3	3	4	4	4	4	4	3	4	2	3	3	2];
    mod_dist_segnum=[6	2	1	4	10	6	16	9	13	5	17	13	11	2	3	8	18	21	9	12	5	7	8	6];
    
    high_dist_vnum=[4	6	6	4	2	4	5	5	5	6	6	4	5	6	4	4	4	3	2	5	5	5	5	5];
    high_dist_segnum=[11	7	8	20	7	7	6	7	8	9	10	15	9	11	10	19	14	2	4	10	11	12	13	14];
    
    low_dist=[low_dist_vnum' low_dist_segnum'];
    mod_dist=[mod_dist_vnum' mod_dist_segnum'];
    high_dist=[high_dist_vnum' high_dist_segnum'];
    
    %distnums=[1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	2	2	3	3	2	2	2];
    %targnums=[1	1	1	1	1	1	2	2	2	2	2	2	2	2	3	3	3	2	3	1	1	1	1	1	1	1	1	1	1	2	2	2	2	2	3	3	2	2	2	2	2	2	3	1	1	1	1	1	2	2	2	2	2	2	3	3	3	3	3	2	2	2	2	1	1	2];
    %bgnums=[1	1	1	1	1	2	1	1	1	1	1	2	2	3	1	1	2	3	3	1	2	3	3	1	1	1	1	2	2	2	3	3	3	2	1	1	1	1	1	2	2	2	3	1	1	1	2	2	1	1	1	1	1	1	1	1	1	1	2	2	2	1	1	1	1	1];
    
    cat_ix={low_dist;mod_dist;high_dist};
    cat_labels={'low dist','mod dist','high dist'};
elseif(strcmp(sortbytype,'everything'))
    vnum=[repmat(2,1,7) repmat(3,1,13) repmat(4,1,21) repmat(5,1,14) repmat(6,1,11)];
    segnum=[1:7 1:13 1:21 1:14 1:11];
    
    everything=[vnum' segnum'];
    cat_ix={everything};
    cat_labels={'everything'};
end

%%
if(drawsegs)
    % Figure showing Light, Dark, and Combo best per seg (at different ranks)
    % TODO: Figure 2(A)-(C)
    %       Low medium and high categories for distance
    %       Make legend boxes smaller
    %       Consecutive numbering of segments
    %       Knock tops off the y axis (after 100)
    close all
    if(~doingkalman)
        rix_range=repmat([1 5],1,2);
    else
        rix_range=[1 1];
    end
    bestperseg=[false(1,numel(rix_range)/2) true(1,numel(rix_range)/2)];
    av_perf=zeros(numel(rix_range),3);
    for rix=1:numel(rix_range)
        rankix=rix_range(rix);
        for k=1:numel(cat_ix)
            [catrows,~]=size(cat_ix{k});
            catperf=zeros(catrows,3);
            catperf_raw=catperf;
            segframes=zeros(catrows,1);

            for t=1:catrows
                vidnum=cat_ix{k}(t,1);
                segnum=cat_ix{k}(t,2);
                a=load([load_dir 'vid' num2str(vidnum) '.mat']);

                compare_inds=1:numel(a.mingt);
                % indices which are part of mingt
                compare_inds=compare_inds(a.mingt);
                % Incidices which are part of both mingt and the segment frame
                % list
                compare_inds=compare_inds(ismember(compare_inds,a.segment_frame_list(segnum,1):a.segment_frame_list(segnum,2)));
                segframes(t)=numel(compare_inds);

                if(bestperseg(rix))
                    catperf(t,1)=a.best_seg{1}.percent(rankix,segnum);
                    catperf(t,2)=a.best_seg{3}.percent(rankix,segnum);
                    catperf(t,3)=a.best_seg{5}.percent(rankix,segnum);

                    catperf_raw(t,1)=a.best_seg{1}.perf(rankix,segnum);
                    catperf_raw(t,2)=a.best_seg{3}.perf(rankix,segnum);
                    catperf_raw(t,3)=a.best_seg{5}.perf(rankix,segnum);
                else
                    catperf(t,1)=a.best_overall{1}.percent(rankix,segnum);
                    catperf(t,2)=a.best_overall{3}.percent(rankix,segnum);
                    catperf(t,3)=a.best_overall{5}.percent(rankix,segnum);

                    catperf_raw(t,1)=a.best_overall{1}.segperf(rankix,segnum);
                    catperf_raw(t,2)=a.best_overall{3}.segperf(rankix,segnum);
                    catperf_raw(t,3)=a.best_overall{5}.segperf(rankix,segnum);
                end
            end

            % Average performance across cat_ix (weighted by frames in
            % segments)
            av_perf(rix,:)=sum(catperf_raw,1) ./ repmat(sum(segframes),1,3);

            [catperf_rows,~]=size(catperf);
            figure(k+(rix-1)*numel(cat_ix))
            midpos=1+[0.5*(min(all_subcats{k}{1}(:))+max(all_subcats{k}{1}(:)));...
                0.5*(min(all_subcats{k}{2}(:))+max(all_subcats{k}{2}(:)));...
                0.5*(min(all_subcats{k}{3}(:))+max(all_subcats{k}{3}(:)))] -...
                min(all_subcats{k}{1}(:));
            maxpos=1+0.5+[max(all_subcats{k}{1}(:)) max(all_subcats{k}{2}(:)) max(all_subcats{k}{3}(:))] -...
                min(all_subcats{k}{1}(:));

            bar(100*catperf)
            hold on
            h=fill([1 catperf_rows catperf_rows 1],[0 0 100 100],'w')
            set(h,'LineStyle','none')
            h=fill([maxpos(1:2) maxpos(2:-1:1)],[0 0 100 100],'k');
            set(h,'FaceColor',[0.9 0.9 0.9],'LineStyle','none')
            hold on
            bar(100*catperf)
            legend({'Dark','Light','Combined'},'box','off','Location','northwest')
            ylim([0 100])
            set(gca,'XTick',midpos)
            set(gca,'XTickLabels',{'Low','Moderate','High'})
            set(gca,'TickLength',[0 0])
            set(gca,'YTick',0:20:100)
            hold on
    %         plot(repelem(maxpos(1),1,2),[0 100],'k-')
    %         plot(repelem(maxpos(2),1,2),[0 100],'k-')
            if(bestperseg(rix))
                set(gcf,'Name',['Seg R' num2str(rankix) ' ' cat_labels{k}])
            else
                set(gcf,'Name',['All R' num2str(rankix) ' ' cat_labels{k}])
            end
            set(gca,'box','off','FontSize',16,'FontName','Times New Roman')
            colormap([0 0 0; 1 1 1; 0.5 0.5 0.5])
            xlabel('Distance')
            ylabel('%frames detected')
            saveas(gcf,[out_dir get(gcf,'Name') '.fig'],'fig')

            %         saveas(gcf,[out_dir get(gcf,'Name') '.pdf'],'pdf')
        end
    end

    %%
    % Comparison of kalman filtered vs not kalman filtered, rank 1 only
    close all
    for bestperseg=[false true]
        for k=1:numel(cat_ix)
            [catrows,~]=size(cat_ix{k});
            catperf_detect=zeros(catrows,3);
            catperf_kalman=catperf_detect;
            for t=1:catrows
                vidnum=cat_ix{k}(t,1);
                segnum=cat_ix{k}(t,2);
                rankix=1;
                a=load([detect_dir 'vid' num2str(vidnum) '.mat']);
                b=load([track_dir 'vid' num2str(vidnum) '.mat']);
                if(bestperseg)
                    catperf_detect(t,1)=a.best_seg{1}.percent(rankix,segnum);
                    catperf_detect(t,2)=a.best_seg{3}.percent(rankix,segnum);
                    catperf_detect(t,3)=a.best_seg{5}.percent(rankix,segnum);
                    catperf_kalman(t,1)=b.best_seg{1}.percent(rankix,segnum);
                    catperf_kalman(t,2)=b.best_seg{3}.percent(rankix,segnum);
                    catperf_kalman(t,3)=b.best_seg{5}.percent(rankix,segnum);
                else
                    catperf_detect(t,1)=a.best_overall{1}.percent(rankix,segnum);
                    catperf_detect(t,2)=a.best_overall{3}.percent(rankix,segnum);
                    catperf_detect(t,3)=a.best_overall{5}.percent(rankix,segnum);
                    catperf_kalman(t,1)=b.best_overall{1}.percent(rankix,segnum);
                    catperf_kalman(t,2)=b.best_overall{3}.percent(rankix,segnum);
                    catperf_kalman(t,3)=b.best_overall{5}.percent(rankix,segnum);
                end
            end
            % Interleave detect and kalman
            catperf_interleave=zeros(catrows,6);
            catperf_interleave(:,[1 3 5]) = catperf_detect;
            catperf_interleave(:,[2 4 6]) = catperf_kalman;
            figure(k+(1*bestperseg)*numel(cat_ix))
            bar(100*catperf_interleave)
            legend({'Detect:DARK','Track:DARK','Detect:LIGHT','Track:LIGHT',...
                'Detect:COMBO','Track:COMBO'},'box','off','Location','northwest')
            ylim([0 120])
            set(gca,'XTick',1:catrows)
            set(gca,'YTick',0:20:100)
            if(bestperseg)
                set(gcf,'Name',['Compare -Seg R' num2str(rankix) ' ' cat_labels{k}])
            else
                set(gcf,'Name',['Compare -All R' num2str(rankix) ' ' cat_labels{k}])
            end
            set(gca,'box','off','FontSize',16,'FontName','Times New Roman')
            saveas(gcf,[out_dir get(gcf,'Name') '.fig'],'fig')
            %         saveas(gcf,[out_dir get(gcf,'Name') '.pdf'],'pdf')
        end
    end
end

%%
% Plot showing performance across target velocities
% Load all videos
% Figure 3(A)
% TODO: Convert screen widths/s to degrees /s (DONE)
%       Convert to percentage and errorbars (DONE)
close all
camera_fov=84;
camera_mul=84/1920;
vidnum_list=3:6;
vardo=5;
velsize_suc_count=[];
num_in_bin=[];
for t=1:numel(vidnum_list)
    vidnum=vidnum_list(t);
    a=load([load_dir 'vid' num2str(vidnum) '.mat']);
    velsize_suc_count=cat(2,velsize_suc_count,a.best_overall{vardo}.velsize_suc_count);
    num_in_bin=cat(2,num_in_bin,a.best_overall{vardo}.num_in_bin);
end

edges_vel=a.best_overall{vardo}.edges_vel;
edges_size=a.best_overall{vardo}.edges_size;
% Sum across segments
app_speed_perf=sum(velsize_suc_count,2)./sum(num_in_bin,2);
if(~doingkalman)
    app_speed_perf=reshape(app_speed_perf,10,numel(edges_vel)-1,numel(edges_size)-1);
    totalsuc=reshape(sum(velsize_suc_count,2),10,numel(edges_vel)-1,numel(edges_size)-1);
    totalbin=reshape(sum(num_in_bin,2),10,numel(edges_vel)-1,numel(edges_size)-1);
else
    app_speed_perf=reshape(app_speed_perf,1,numel(edges_vel)-1,numel(edges_size)-1);
    totalsuc=reshape(sum(velsize_suc_count,2),1,numel(edges_vel)-1,numel(edges_size)-1);
    totalbin=reshape(sum(num_in_bin,2),1,numel(edges_vel)-1,numel(edges_size)-1);
end
% Calculate error bars at 95% CI
alpha=0.05;
% colours_list=[0 0 0;
%     0.4 0.4 0.4;...
%     0.7 0.7 0.7];
% chars_list={'x-','o:','^--'};
for oix=1:2%numel(edges_size)-1
    figure(oix)
    pci=zeros(numel(edges_vel)-1,2);
    for t=1:numel(edges_vel)-1
        [~,pci(t,:)]=binofit(totalsuc(1,t,oix),totalbin(1,t,oix));
    end
    xbase=0.5*(edges_vel(1:end-1)+edges_vel(2:end));
    xco=[xbase,xbase(end:-1:1)];
    yco=[pci(:,1);pci(end:-1:1,2)];
    % h=fill(xco,yco,'b','FaceAlpha',0.3,'EdgeAlpha',0);
    % plot(xbase,app_speed_perf(1,:),'bx-',...
    %     xbase,pci(:,1),'b--',...
    %     xbase,pci(:,2),'b--',...
    %     'LineWidth',2)

    p=errorbar(camera_fov*xbase,100*app_speed_perf(1,:,oix),100*(app_speed_perf(1,:,oix)'-pci(:,1)),...
        100*(pci(:,2)-app_speed_perf(1,:,oix)'),'kx:');
    set(gca,'Box','off')
%     set(p,'Color',colours_list(oix,:))
    ylim([0 100])
    xlabel('Apparent velocity (deg / s)')
    ylabel('%frames detected')
    title([num2str(edges_size(oix)) '-' num2str(edges_size(oix+1))])
%     hold on
end

for oix=1:3
    figure(3+oix)
    plot(camera_fov*xbase,totalbin(1,:,oix),'x-','LineWidth',2)
    title([num2str(edges_size(oix)) '-' num2str(edges_size(oix+1))])
    xlabel('Apparent velocity (deg / s)')
    ylabel('Frames in bin')
end

% [px,py]=meshgrid(0.5*(edges_vel(1:end-1)+edges_vel(2:end)),0.5*(edges_size(1:end-1)+edges_size(2:end)));
figure
hold on
[px,py]=meshgrid(1:19,1:3);
pz=NaN(size(px));
for k=1:numel(px)
    pz(k) = sum(totalsuc(1,px(k),py(k)))./totalbin(1,px(k),py(k));
end
pz(isnan(pz))=0;
contourf(camera_fov*edges_vel(px),edges_size(py)/1920*camera_fov,pz)
colorbar
colormap('hot')
xlabel('Apparent velocity (deg / s)')
ylabel('Apparent size (deg)')

% figure
% for oix=1:2
%     pci=zeros(numel(edges_vel)-1,2);
%     for t=1:numel(edges_vel)-1
%         [~,pci(t,:)]=binofit(totalsuc(1,t,oix),totalbin(1,t,oix));
%     end
%     xbase=0.5*(edges_vel(1:end-1)+edges_vel(2:end));
%     xco=[xbase,xbase(end:-1:1)];
%     yco=[pci(:,1);pci(end:-1:1,2)];
%     % h=fill(xco,yco,'b','FaceAlpha',0.3,'EdgeAlpha',0);
%     % plot(xbase,app_speed_perf(1,:),'bx-',...
%     %     xbase,pci(:,1),'b--',...
%     %     xbase,pci(:,2),'b--',...
%     %     'LineWidth',2)
% 
%     p=errorbar(camera_fov*xbase+(oix-1.5)*1,100*app_speed_perf(1,:,oix),100*(app_speed_perf(1,:,oix)'-pci(:,1)),...
%         100*(pci(:,2)-app_speed_perf(1,:,oix)'),'kx:');
%     hold on
%     
% %     set(p,'Color',colours_list(oix,:))
% end
% set(gca,'Box','off')
% ylim([0 100])
% xlabel('Apparent velocity (deg / s)')
% ylabel('%frames detected')

%%
% Performance as a function of pixel2PR
% Figure 3(B)
% TODO: Convert to percentage
%       Only 3 coloured lines for the model at different blur sizes. For random, just have a black or grey dashed line and discuss that in the text.
%       Convert spacing to degrees
%       Make things black with different symbols
%       Need 
close all
camera_mul=84/1920;
vidnum_list=3:6; % Excluding the 720p video to make analysis easier
% vidnum_list=2:6;
vardo=5;
% totalposs=zeros(numel(vidnum_list),1);
perf=zeros([7 size(pixel2PR_s)]);
totalposs=perf;
base_prob=zeros([numel(vidnum_list) size(pixel2PR_s)]);
field_pixels=base_prob;
% totalposs=[];

for t=1:numel(vidnum_list)
    vidnum=vidnum_list(t);
    a=load([load_dir 'vid' num2str(vidnum) '.mat']);
    [numsegments,~]=size(a.segment_time_list);
    tempsegperf=zeros([numsegments size(pixel2PR_s)]);
    for k=1:numsegments
        tempsegperf(k,:,:)=a.best_forpr_forseg{vardo}.segperf(1,k,:,:);
        temp_totalposs=reshape(a.numframes_inseg(:,k),[1 size(pixel2PR_s)]);
        totalposs=cat(1,totalposs,temp_totalposs);
    end
    
%     perf=cat(1,perf,a.best_forpr{vardo}.perf(1,:,:));
    perf=cat(1,perf,tempsegperf);
    base_prob(t,:,:)=reshape(a.base_probability,size(pixel2PR_s));
    field_pixels(t,:,:)=reshape(a.field_pixels,size(pixel2PR_s));
end

% Sum perf and total possible across videos
perf=reshape(sum(perf(perf_mask.mask_upper,:,:),1),size(pixel2PR_s));
totalposs=reshape(sum(totalposs(perf_mask.mask_upper,:,:),1),size(pixel2PR_s));

% perf=reshape(perf,size(pixel2PR_s));
% contourf(pixel2PR_s,sigma_hw_s,perf)

% For minimum blur, take a slice along pixel2PR
% IMPORTANT: THIS ASSUMES THAT ALL VIDEOS HAVE SAME DIMENSIONS
blur_slice_range=[1 2 5];
% colorchar_range='bgr'
figure
% hold on
% for t=1:3
%     plot([1 1],[1 1],[colorchar_range(1) '-'],...
%         [1 1],[1 1],[colorchar_range(2) '-'],...
%         [1 1],[1 1],[colorchar_range(3) '-'])
% end
% b=legend(num2str(sigma_hw_s(blur_slice_range,1)))

chartype={'x','o','^'};
linecols=[0 0 0; 0.4 0.4 0.4; 0.6 0.6 0.6];
for t=1:numel(blur_slice_range)
    blur_slice=blur_slice_range(t);
    p_perf=perf(blur_slice,:)./totalposs(blur_slice,:);
    expected=base_prob(1,blur_slice,:);
    expected=expected(:);
    modelunits=field_pixels(1,blur_slice,:);
    modelunits=modelunits(:);
    
%     colorchar=colorchar_range(t);
    h=plot(unique(pixel2PR_s)*camera_mul,100*p_perf,['k' char(chartype(t)) '-'],...
        'MarkerSize',9,'LineWidth',2);
    set(h,'Color',linecols(t,:));
%         unique(pixel2PR_s),expected,[colorchar 'o--'],'LineWidth',2)
    hold on
%     plot([-1 -1],[-1 -1],'b-',...
%         [-1 -1],[-1 -1],'g-',...
%         [-1 -1],[-1 -1],'r-','LineWidth',2)
end
h=plot(unique(pixel2PR_s)*camera_mul,100*expected,'--');
set(h,'Color',[0.6 0.6 0.6]);
h=plot(unique(pixel2PR_s)*camera_mul,100*modelunits/max(modelunits),'.:','MarkerSize',16);
set(h,'Color',[0.2 0.2 0.2]);
ylim([0 100])
xlim(camera_mul*[min(pixel2PR_s(:)) max(pixel2PR_s(:))])
xlabel('Degrees between model units')
ylabel('%frames detected')
blurvals=sigma_hw_s(blur_slice_range,1)*camera_mul;
legend({num2str(blurvals(1)),num2str(blurvals(2)),num2str(blurvals(3))},'box','off')
set(gca,'FontSize',10,'FontName','Times New Roman','box','off')

% Figure 3(C)
% TODO: Convert spacing to degrees
figure
bar(unique(pixel2PR_s),modelunits/max(modelunits))
colormap([0.5 0.5 0.5])
xlabel('Pixels between model units')
ylabel('Normalised number of model units')
set(gca,'FontSize',10,'FontName','Times New Roman','box','off')
movegui(gcf,'east')

%%
% New Figure 4(A)
close all
vardo=5;

detect_perf=zeros(numel(perf_mask.vidnum_list),1);
track_perf=detect_perf;
totalpos=detect_perf;
detect_totalperf=detect_perf;
track_totalperf=detect_perf;
dperf_bestseg=detect_perf;
tperf_bestseg=detect_perf;
% Make use of perf_mask


% Load everything then mask it out later
for t=1:numel(perf_mask.vidnum_list)
    vidnum=perf_mask.vidnum_list(t);
    segnum=perf_mask.segnum(t);
    
    a=load([detect_dir 'vid' num2str(vidnum) '.mat']);
    b=load([track_dir 'vid' num2str(vidnum) '.mat']);
    
    detect_perf(t)=a.best_overall{vardo}.segperf(1,segnum);
    track_perf(t)=b.best_overall{vardo}.segperf(segnum);
    
    dperf_bestseg(t)=a.best_seg{vardo}.perf(1,segnum);
    tperf_bestseg(t)=b.best_seg{vardo}.perf(1,segnum);
    
    totalpos(t)=a.numframes_inseg(1,segnum);
end

detect_perc=detect_perf./totalpos;
track_perc=track_perf./totalpos;

dperf_bestseg_perc=dperf_bestseg./totalpos;
tperf_bestseg_perc=tperf_bestseg./totalpos;

low_perf=sum(perf(perf_mask.mask_lower));
mid_perf=sum(perf(perf_mask.mask_mid));
high_perf=sum(perf(perf_mask.mask_upper));


mask_to_use={perf_mask.mask_lower;...
    perf_mask.mask_mid;...
    perf_mask.mask_upper};

detect_colour=[0.3 0.3 0.3];
track_colour=[0 0 0];
line_types={'-','--',':'};

% n1=histcounts(detect_perc,edges); n2=histcounts(dperf_bestseg_perc,edges); plot(0.5*(edges(1:end-1)+edges(2:end)),[n1' n2'])

figure
delta_track_store=cell(3,1);
delta_best_store=delta_track_store;
for k=1:3
    mask=mask_to_use{k};
    h=plot(1:sum(mask_to_use{k}),detect_perc(mask),line_types{k});
    set(h,'Color',detect_colour);
    hold on
    h=plot(1:sum(mask_to_use{k}),track_perc(mask),line_types{k})
    set(h,'Color',track_colour);
    
    % Calculate deltas
    delta_track_store{k}=track_perc(mask) - detect_perc(mask);
    delta_best_store{k}=dperf_bestseg_perc(mask) - detect_perc(mask);
end

figure
plot(sort(delta_track_store{1}),'r-')
hold on
plot(sort(delta_track_store{2}),'g-')
plot(sort(delta_track_store{3}),'b-')
legend({'low','med','high'},'box','off')
figure
plot(sort(delta_best_store{1}),'r-')
hold on
plot(sort(delta_best_store{2}),'g-')
plot(sort(delta_best_store{3}),'b-')
legend({'low','med','high'},'box','off')

%
figure
n2=cell(3,1);
edges2=n2;
for k=1:3
%     edges=linspace(-0.1+min(delta_best_store{k}),0.1+max(delta_best_store{k}),15);
    edges=linspace(-0.4,0.4,17);
    n1=histcounts(delta_best_store{k},edges)/numel(delta_best_store{k});
    
    edges2{k}=linspace(-0.1+min(delta_track_store{k}),0.1+max(delta_track_store{k}),17);
    edges2{k}=linspace(-0.4,0.4,17);
    n2{k}=histcounts(delta_track_store{k},edges2{k})/numel(delta_track_store{k});
    
    plot(0.5*(edges(1:end-1)+edges(2:end)),n1,'.-','MarkerSize',12)
    hold on
end
title('Parameters')
legend({'Low','Med','High'},'box','off')

figure
for k=1:3
    plot(0.5*(edges2{k}(1:end-1)+edges2{k}(2:end)),n2{k},'.-','MarkerSize',12)
    hold on
end
title('Tracking')
legend({'Low','Med','High'},'box','off')
movegui(gcf,'east')

% Cumulative detect and track distributions
figure
cusum=zeros(numel(detect_perc),3);
sort_detect=sort(detect_perc);
sort_track=sort(track_perc);
sort_best=sort(dperf_bestseg_perc);
for t=1:numel(detect_perc)
    cusum(t,1)=sum(detect_perc(1:t));
    cusum(t,2)=sum(track_perc(1:t));
    cusum(t,3)=sum(dperf_bestseg_perc(1:t));
end
% for t=1:3
%     cusum(:,t)=cusum(:,t)/max(cusum(:,t));
% end
plot(linspace(0,1,numel(detect_perc)),cusum,'-')
legend({'Detect','Track','Best'},'box','off')

% % Cumulative plot of improvement
% figure
% cusum=zeros(numel(delta_track_store{1}),1);
% dsort=sort(delta_track_store{1});
% for k=1:numel(delta_track_store{1})
%     cusum(k)=sum(dsort(1:k));
% end
% plot(cusum)


figure
plot(totalpos(perf_mask.mask_lower),track_perf(perf_mask.mask_lower)./detect_perf(perf_mask.mask_lower),'x',...
    totalpos(perf_mask.mask_mid),track_perf(perf_mask.mask_mid)./detect_perf(perf_mask.mask_mid),'o',...
    totalpos(perf_mask.mask_upper),track_perf(perf_mask.mask_upper)./detect_perf(perf_mask.mask_upper),'^')
%%
figure
uvid=unique(perf_mask.vidnum_list);
for k=1:numel(uvid)
    mask = perf_mask.vidnum_list == uvid(k);
    plot(totalpos(mask),track_perf(mask)./detect_perf(mask),'x')
    hold on
end

%%
close all
plot(totalpos(mask_to_use{1}),track_perf(mask_to_use{1})./detect_perf(mask_to_use{1}),'rx',...
    totalpos(mask_to_use{2}),track_perf(mask_to_use{2})./detect_perf(mask_to_use{2}),'gx',...
    totalpos(mask_to_use{3}),track_perf(mask_to_use{3})./detect_perf(mask_to_use{3}),'bx')
for k=1:3
    mn=track_perf(mask_to_use{k})./detect_perf(mask_to_use{k}); mn(isinf(mn))=NaN; mean(mn,'omitnan');
end

tracknums=zeros(3,1);
detectnums=tracknums;
possnums=tracknums;
for k=1:3
    tracknums(k)=sum(track_perf(mask_to_use{k}));
    detectnums(k)=sum(detect_perf(mask_to_use{k}));
    possnums(k)=sum(totalpos(mask_to_use{k}));
end

ptrack=zeros(3,3);
[ptrack(:,2),ptrack(:,[1 3])]=binofit(tracknums,possnums);
[phat,~]=binofit(detectnums,possnums);

return
%%
% Compare not tracked to tracked average performance
% Figure 4(A)
% TODO: Reduce legend sizes
%       Make grayscale
%       Make the X division Low / Medium / High performance (e.g. 0-30, 31-60, 61+)
%       when using detection only

close all
vidnum_range=2:6;
vardo_range=[1 3 5];
detect_perf=zeros(numel(vardo_range),numel(vidnum_range));
track_perf=detect_perf;
totalpos=detect_perf;
detect_totalperf=detect_perf;
track_totalperf=detect_perf;

for t_outer=1:numel(vardo_range)
    vardo=vardo_range(t_outer);
    for t=1:numel(vidnum_range)
        vidnum=vidnum_range(t);
        a=load([detect_dir 'vid' num2str(vidnum) '.mat']);
        b=load([track_dir 'vid' num2str(vidnum) '.mat']);
        
        detect_perf(t_outer,t)=a.best_overall{vardo}.perf(1);
        track_perf(t_outer,t)=b.best_overall{vardo}.perf(1);
        
        totalpos(t_outer,t)=sum(a.numframes_inseg(1,:));
        detect_totalperf(t_outer,t)=sum(a.best_overall{vardo}.segperf(1,:));
        track_totalperf(t_outer,t)=sum(b.best_overall{vardo}.segperf(1,:));
    end
end
detect_p=sum(detect_totalperf,2)./sum(totalpos,2);
track_p=sum(track_totalperf,2)./sum(totalpos,2);
% Interleave the columns
docombined=true;
if(~docombined)
    inter_perf=zeros(numel(vardo_range),2);
    inter_perf(:,1)=detect_p;
    inter_perf(:,2)=track_p;
    bar(inter_perf)
else
    inter_perf=zeros(numel(vidnum_range),2);
    inter_perf(:,1)=detect_totalperf(3,:)./totalpos(3,:);
    inter_perf(:,2)=track_totalperf(3,:)./totalpos(3,:);
    [~,pci_detect]=binofit(detect_totalperf(3,:),totalpos(3,:));
    [~,pci_track]=binofit(track_totalperf(3,:),totalpos(3,:));
    bar(100*inter_perf)
    c=get(gca,'Children');
    colormap([0.5 0.5 1; 1 0.5 0.5])
    hold on
    errorbar((1:5)+c(2).XOffset,100*inter_perf(:,1),100*(inter_perf(:,1)-pci_detect(:,1)),100*(pci_detect(:,2)-inter_perf(:,1)),'k.','LineWidth',1);
    errorbar((1:5)+c(1).XOffset,100*inter_perf(:,2),100*(inter_perf(:,2)-pci_track(:,1)),100*(pci_track(:,2)-inter_perf(:,2)),'k.','LineWidth',1);
    xlabel('Video file')
    ylabel('%frames detected')
    legend({'Without track','With'},'box','off','Location','Northwest')
    set(gca,'box','off','FontName','Times New Roman','FontSize',10)
end

%%
% Bar graph showing generic parameters vs best for segment parameters
% Figure 4(B)
%       Same categories as for A
%       grayscale
%       Reduce legend sizes
close all
vidnum_range=2:6;
vardo=5;
best_pci=zeros(numel(vidnum_range),2);
overall_pci=best_pci;

best_perf=zeros(numel(vidnum_range),1);
overall_perf=best_perf;
poss=best_perf;
for t=1:numel(vidnum_range)
    vidnum=vidnum_range(t);
    a=load([detect_dir 'vid' num2str(vidnum) '.mat']);
    best_perf(t)=sum(a.best_seg{vardo}.perf(1,:));
    overall_perf(t)=sum(a.best_overall{vardo}.segperf(1,:));
    % IMPORTANT: This relies on the best pixel2PR being the lowest
    poss(t)=sum(a.numframes_inseg(1,:));
end
[~,best_pci]=binofit(best_perf,poss);
[~,overall_pci]=binofit(overall_perf,poss);
% 2 columns per video
inter_perf_p=[overall_perf./poss best_perf./poss];
bar(100*inter_perf_p)
c=get(gca,'Children');
% colormap([0.5 0.5 1; 1 0.5 0.5])
colormap([0.7 0.7 0.7; 0.2 0.2 0.2])
hold on
errorbar((1:numel(vidnum_range))-c(1).XOffset,100*inter_perf_p(:,1),100*(inter_perf_p(:,1)-overall_pci(:,1)),100*(overall_pci(:,2)-inter_perf_p(:,1)),'k.','LineWidth',1)
errorbar((1:numel(vidnum_range))-c(2).XOffset,100*inter_perf_p(:,2),100*(inter_perf_p(:,2)-best_pci(:,1)),100*(best_pci(:,2)-inter_perf_p(:,2)),'k.','LineWidth',1)
legend({'Best for video','Best per segment'},'box','off','Location','Northwest')
xlabel('Video file')
ylabel('%frames detected')
set(gca,'FontName','TimesNewRoman','FontSize',10,'box','off')

%%
% Show ranks for the three variants
% Figure 4(C)
%       Make crosses smaller or replace with dots or something
%       Might be worth splitting this up into low/medium/high performance as well

close all
vardo_range=[1 3 5];

vidnum_range=2:6;

% Get the overall performance
perf=zeros(numel(vardo_range),numel(vidnum_range),10);

poss=zeros(numel(vardo_range),numel(vidnum_range));

rixperf=zeros(numel(vardo_range),10);
rixpci=zeros(numel(vardo_range),10,2);
for t_outer=1:numel(vardo_range)
    vardo=vardo_range(t_outer);
    for t=1:numel(vidnum_range)
        vidnum=vidnum_range(t);
        a=load([detect_dir 'vid' num2str(vidnum) '.mat']);
        
        perf(t_outer,t,:)=sum(a.best_overall{vardo}.segperf,2);
        poss(t_outer,t)=sum(a.numframes_inseg(1,:));
        for rix=1:10
            rixperf(:,rix)=sum(perf(:,:,rix),2)./sum(poss,2);
            for k=1:3
                [~,rixpci(k,rix,:)]=binofit(sum(perf(k,:,rix),2),sum(poss(k,:),2));
            end
        end
    end
end
rix_inc=zeros(3,10);
rix_inc(:,1)=rixperf(:,1);
for rix=2:10
    rix_inc(:,rix)=rixperf(:,rix)-rixperf(:,rix-1);
end
bar(100*rix_inc')
c=get(gca,'Children')
set(c,'BarWidth',0.8)
colormap([0 0 0; 1 1 1; 0.5 0.5 0.5])
hold on
% plot((1:10)-c(1).XOffset,100*rixperf(1,:),'kx--','LineWidth',1)
% plot((1:10)-c(2).XOffset,100*rixperf(2,:),'kx:','LineWidth',1)
plot((1:10)-c(3).XOffset,100*rixperf(3,:),'kx-.','LineWidth',1)
% errorbar((1:10)-c(1).XOffset,100*rixperf(1,:),100*(rixperf(1,:)-rixpci(1,:,1)),100*(rixpci(1,:,2)-rixperf(1,:)),'k.--','LineWidth',2)
% errorbar((1:10)-c(2).XOffset,100*rixperf(2,:),100*(rixperf(2,:)-rixpci(2,:,1)),100*(rixpci(2,:,2)-rixperf(2,:)),'k.:','LineWidth',2)
% errorbar((1:10)-c(3).XOffset,100*rixperf(3,:),100*(rixperf(3,:)-rixpci(3,:,1)),100*(rixpci(3,:,2)-rixperf(3,:)),'k.-','LineWidth',2)
hold off
xlabel('Rank')
ylabel('%frames detected')
legend({'Dark','Light','Combined'},'box','off','Location','Northwest')
set(gca,'box','off','FontName','TimesNewRoman','FontSize',10)
