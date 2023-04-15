clear all
close all

t_lost_it=NaN(100,1);

for t=1:100
    a=load(['D:\simfiles\conf2\results\track_saccades_distract_4-' num2str(t) '.mat']);
    
    ind=find(a.t_th_rel_history > 20 | a.t_th_rel_history < -20,1,'first');
    
    if(~isempty(ind))
        t_lost_it(t) = ind;
    end
end

t_lost_it(isnan(t_lost_it)) = 30250;

plot(t_lost_it*0.001)

