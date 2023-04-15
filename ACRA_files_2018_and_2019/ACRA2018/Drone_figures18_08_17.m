clear all
close all

doingkalman=true;
if(~doingkalman)
    load_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_17\';
    out_dir='F:\MrBlack\CONFdronefiles\autofigures\';
else
    load_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_19_kalman\';
    out_dir='F:\MrBlack\CONFdronefiles\autofigures_kalman\';
end

detect_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_17\';
track_dir='F:\MrBlack\CONFdronefiles\CollapsedAnalysis18_08_19_kalman\';


vidname_list={'2018_05_31_13_48_46';...
    'Pursuit1';...
    'Pursuit2';...
    'Stationary1';...
    'Stationary2'};

vidnum_list=[2;3;4;5;6];

% Numbers for the different types
collisions=cat(1,[repmat(5,3,1) (12:14)'],[repmat(3,4,1) [5 7 8 9]']);
dogfights=cat(1,[repmat(5,7,1) (1:7)'],[repmat(4,21,1) (1:21)']);
flyby6m=[repmat(5,11,1) (1:11)'];
flyby12m=[repmat(6,11,1) (1:11)'];

cat_ix={collisions;dogfights;flyby6m;flyby12m};
cat_labels={'Collisions','Dogfights','Fly-by at 6m','Fly-by at 12m'};

%%
% Figure showing Light, Dark, and Combo best per seg (at different ranks)
close all
if(~doingkalman)
    rix_range=repmat([1 5],1,2);
else
    rix_range=[1 1];
end
bestperseg=[false(1,numel(rix_range)/2) true(1,numel(rix_range)/2)];
for rix=1:numel(rix_range)
    rankix=rix_range(rix);
    for k=1:numel(cat_ix)
        [catrows,~]=size(cat_ix{k});
        catperf=zeros(catrows,3);
        for t=1:catrows
            vidnum=cat_ix{k}(t,1);
            segnum=cat_ix{k}(t,2);
            a=load([load_dir 'vid' num2str(vidnum) '.mat']);
            if(bestperseg(rix))
                catperf(t,1)=a.best_seg{1}.percent(rankix,segnum);
                catperf(t,2)=a.best_seg{3}.percent(rankix,segnum);
                catperf(t,3)=a.best_seg{5}.percent(rankix,segnum);
            else
                catperf(t,1)=a.best_overall{1}.percent(rankix,segnum);
                catperf(t,2)=a.best_overall{3}.percent(rankix,segnum);
                catperf(t,3)=a.best_overall{5}.percent(rankix,segnum);
            end
        end
        figure(k+(rix-1)*numel(cat_ix))
        bar(100*catperf)
        legend({'Dark','Light','Combined'},'box','off','Location','northwest')
        ylim([0 120])
        set(gca,'XTick',1:catrows)
        set(gca,'YTick',0:20:100)
        if(bestperseg(rix))
            set(gcf,'Name',['Seg R' num2str(rankix) ' ' cat_labels{k}])
        else
            set(gcf,'Name',['All R' num2str(rankix) ' ' cat_labels{k}])
        end
        set(gca,'box','off','FontSize',16,'FontName','Times New Roman')
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
            set(gcf,'Name',['Seg R' num2str(rankix) ' ' cat_labels{k}])
        else
            set(gcf,'Name',['All R' num2str(rankix) ' ' cat_labels{k}])
        end
        set(gca,'box','off','FontSize',16,'FontName','Times New Roman')
        saveas(gcf,[out_dir get(gcf,'Name') '.fig'],'fig')
    %         saveas(gcf,[out_dir get(gcf,'Name') '.pdf'],'pdf')
    end
end




% % Look at differences between DARK/LIGHT/COMBO
% 
% dark_combo_diff=a.best_seg{1}.percent - a.best_seg{5}.percent;
% light_combo_diff=a.best_seg{3}.percent - a.best_seg{5}.percent;
% 
% figure(3)
% bar(dark_combo_diff)
% figure(4)
% bar(light_combo_diff)
% 
% plot(a.best_seg{3}.percent(1,:),light_combo_diff(1,:),'x')