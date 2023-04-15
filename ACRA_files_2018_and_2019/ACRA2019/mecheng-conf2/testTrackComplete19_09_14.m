% [fdsr, linear, sub, linsub, thresh, linthresh, threshlin, nil]

if(set == 1)
    targets={'TrackerInitial19_09_14'};
    grp_range_master{1} = 1:18576;
    num_tests = 30;
end

num_targets=numel(targets);

load_dir='/fast/users/a1119946/simfiles/conf2/data/';

if(~exist('resume','var'))
    resume=false;
end

tic
for t=1:num_targets
    savename=['/fast/users/a1119946/completionfiles/' targets{t} '.mat'];
    grp_range=grp_range_master{t};
    
    if(resume==true)
        a=load(savename);
        ispresent=a.ispresent;
    else
        ispresent=false(size(grp_range));
    end
    
    corrupt = false(size(ispresent));
    first_missing = NaN(size(ispresent));
    
    disp([num2str(t) ' ' num2str(toc)])
    for gix=1:numel(grp_range)
        if(mod(gix,100) == 0)
            disp(['gix: ' num2str(gix) ' avg_nan = ' num2str(mean(first_missing,'omitnan')) ' ' num2str(toc)]);
        end
        g=grp_range(gix);
        
        fname=[load_dir targets{t} '/data/trackres_' num2str(g) '.mat'];
        
        if(~ispresent(gix))
            if(exist(fname,'file'))
                try
                    a=load(fname);
                    
                    % Assume complete and test whether it isn't
                    complete=true;
                    
                    if(sum(outer_log.completed < num_tests(1)))
                        complete=false;
                        first_missing = find(~outer_log.completed,1,'first');
                    end
                    
                    if(complete)
                        ispresent(gix)=true;
                    end
                catch
                    corrupt(gix)=true;
                end
            end
        end
    end
    g_todo=grp_range(~ispresent);
    numtodo=sum(~ispresent(:));
    
    save(['/fast/users/a1119946/completionfiles/' targets{t} '.mat'],'ispresent','grp_range','g_todo','numtodo','corrupt','first_missing');
    disp([targets{t} ': ' num2str(sum(ispresent(:))) '/' num2str(numel(ispresent))])
end