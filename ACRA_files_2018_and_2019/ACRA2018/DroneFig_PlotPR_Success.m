clear all
close all

for vidnum=1
    a=load(['F:\MrBlack\CONFdronefiles\CollapsedAnalysis_HIGHRES_WIDE18_09_14\vid' num2str(vidnum) '.mat']);

    for vardo=[1 3 5];
        suc_trace=a.best_overall{vardo}.suc_trace;

        ssgt=a.best_overall{vardo}.ssgt;
        
        expected=sum(a.base_expected{vardo}(1,:,:,:,:),2);
        expected=expected(a.best_overall{vardo}.refnum(1));
        

        N=a.best_overall{vardo}.hfov * a.best_overall{vardo}.vfov;

        pr_summed=a.best_overall{vardo}.pr_summed;
        pr_norm=pr_summed/N;

        pr_edges = 0:0.1:2;

        % Pr data does not include first 150 frames
%         gtbuttonfill=ssgt.gtbuttonfill(151:end);
        gtbuttonfill=ssgt.gtbuttonfill;
%         gtbuttonfill=[gtbuttonfill';false(numel(pr_norm)-numel(gtbuttonfill),1)];
        gtbuttonfill=gtbuttonfill(1:numel(pr_norm));
        gtbuttonfill=gtbuttonfill(:);

        suc_in_bin=zeros(numel(pr_edges)-1,1);
        num_in_bin=suc_in_bin;

        % Binning by pr
        for t=1:numel(pr_edges)-1
            mask = pr_norm > pr_edges(t) & pr_norm <= pr_edges(t+1) & gtbuttonfill == 1;
            suc_in_bin(t)=sum(suc_trace(mask));
            num_in_bin(t)=sum(mask);
        end
        figure
    %     plot(0.5*(pr_edges(1:end-1)+pr_edges(2:end)),suc_in_bin./num_in_bin,...
    %         0.5*(pr_edges(1:end-1)+pr_edges(2:end)),num_in_bin ./ (sum(num_in_bin)))

        [phat,conf]=binofit(suc_in_bin,num_in_bin);
        LB=phat-conf(:,1);
        UB=conf(:,2)-phat;
        errorbar(0.5*(pr_edges(1:end-1)+pr_edges(2:end)),phat,LB,UB)
        %%
        % Binning by both pr and apparent velocity
        app_vel=a.best_overall{vardo}.app_speed_mag;
        pr_norm=pr_norm(2:end); % Align with app_vel
        app_vel=app_vel(1:numel(pr_norm));
        app_vel=app_vel(:);
        suc_trace=suc_trace(2:end); % Align with app_vel
        suc_trace=suc_trace(1:numel(pr_norm));
        suc_trace=suc_trace(:);
        gtbuttonfill=gtbuttonfill(2:end); % Align with app_vel
        
        pr_edges = linspace(0,1.7,10);
        vel_edges= linspace(0,0.6,10);
        suc_in_bin=zeros(numel(vel_edges)-1,numel(pr_edges)-1);
        num_in_bin=suc_in_bin;
        
        for t=1:numel(vel_edges)-1
            for k=1:numel(pr_edges)-1
                
                mask=pr_norm >= pr_edges(k) & pr_norm < pr_edges(k+1) &...
                    app_vel >= vel_edges(t) & app_vel < vel_edges(t+1);
                
                suc_in_bin(t,k)=sum(suc_trace(mask));
                num_in_bin(t,k)=sum(gtbuttonfill == 1 & mask);
            end
        end
        [px,py]=ndgrid(vel_edges(1:end-1),pr_edges(1:end-1));
        
        contourf(px,py,num_in_bin./max(num_in_bin(:)))
        caxis([0 1])
        colormap('hot')
        xlabel('vel');ylabel('pr')
        movegui(gcf,'west')
        
        figure(2)
        contourf(px,py,suc_in_bin./num_in_bin)
        caxis([0 1])
        colormap('hot')
        xlabel('vel');ylabel('pr')
        colorbar
        movegui(gcf,'east')
    return
    end
end
%%
close all
vraw=a.field_size(1,1);
hraw=a.field_size(1,2);

figure(1)
pp=a.best_forpr{1}.aboveguess(1,:,:);
pp=reshape(pp,10,7);
px=a.best_forpr{1}.mini_pixel2PR_s;
py=a.best_forpr{1}.mini_sigma_hw_s;
% contourf(px*84/hraw,py*84/hraw,pp)
contourf(px,py,pp)
xlabel('pixel spacing')
ylabel('blur')

figure(2)
pp2=a.best_forpr{1}.cs_kernel_size(1,:,:);
pp2=reshape(pp2,10,7);
% contourf(px*84/hraw,py*84/hraw,pp2)
contourf(px,py,pp2)