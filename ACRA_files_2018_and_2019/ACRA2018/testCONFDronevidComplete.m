clear all
close all

%numgind=74400;
dodata=false;
if(dodata)
    numgind=26800;
    target='CONF_Maxlocs18_08_10';
else
    vidnum_range=2:6;
    pr_range=1:80;
    vardo_range=[1 3 5];
    target='CONF_AnalysisKalman18_08_19';
    
    [vid_s,pr_s,var_s]=ndgrid(vidnum_range,pr_range,vardo_range);
end

if(dodata)
    ispresent=false(numgind,1);
    for t=1:numgind
        filename=['/fast/users/a1119946/simfiles/dronefiles/' target '/data/gind' num2str(t) '.mat'];
        if(exist(filename,'file'))
            ispresent(t)=true;
        end
    end
    all_inds=1:numgind;
    g_todo=all_inds(~ispresent);
    rand_todo=g_todo(randperm(sum(~ispresent)));
    save(['/fast/users/a1119946/completionfiles/' target '.mat'],'ispresent','all_inds','g_todo','rand_todo');

else
    ispresent=false(size(vid_s));
    for t=1:numel(vid_s)
        vidnum=vid_s(t);
        prnum=pr_s(t);
        varnum=var_s(t);
        filename=['/fast/users/a1119946/simfiles/dronefiles/' target '/v' num2str(vidnum) '-p' num2str(prnum) '-v' num2str(varnum) '.mat'];
        if(exist(filename,'file'))
            ispresent(t)=true;
        end
    end
    % Collapse ispresent down to vid and varnum, to match the submission
    % script
    [vid_s,var_s]=meshgrid(vidnum_range,vardo_range);
    ispresent=sum(ispresent,2);
    ispresent=ispresent == numel(pr_range);
    vid_todo=vid_s(~ispresent);
    var_todo=var_s(~ispresent);
end

%ispresent(38801:52000)=true; % Skip stationary1 
%ispresent(38801:70000)=true; % Skip the stationary videos
%ispresent(70001:74400)=true; % skip video14 and video15
% 
