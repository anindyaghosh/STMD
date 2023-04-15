clear all
close all

% Convert the histograms into probabilities for use by the tracker

load_dir = 'E:\PHD\conf2\data\results\';
output_dir = 'E:\PHD\conf2\data\trackerinfo\';
output_integration = 4;

a=load([load_dir 'integ_' num2str(output_integration) '-' num2str(20) '.mat']);
[numrows,~]=size(a.hist_R_v);

t_vel_range=20:20:800;
% t_vel_range=[20:20:400 440:40:520];
numcols = numel(t_vel_range);

all_H = NaN(numrows, numcols);

for t=1:numel(t_vel_range)
    a=load([load_dir 'integ_' num2str(output_integration) '-' num2str(t_vel_range(t)) '.mat']);
    all_H(:,t)=a.hist_R_v;
end

% If the model produces no target related outputs then everything is zero.
% Enforce a minimum probabiltiy of 0.001
for k=1:numcols
    if(sum(all_H(:,k)) == 0)
        all_H(2:end,k)=0.001;
        all_H(1,k)=1-sum(all_H(2:end,k));
    end
end

all_prob=all_H ./ repmat(sum(all_H,1),numrows,1);

plot(all_prob)

all_prob=max(all_prob,0.001);

% Normalise such that columns sum to 1

all_prob = all_prob ./repmat(sum(all_prob,1),numrows,1);

% crunch=all_prob(1:2:end-1,:) + all_prob(2:2:end,:);

pv=NaN(size(all_prob));

for v=1:numrows
%     pv=crunch(v,:) ./ sum(crunch(v,:));
    pv(v,:)=all_prob(v,:) ./ sum(all_prob(v,:));
%     bar(t_vel_range,pv)
%     waitforbuttonpress
end

plot(a.resp_thresholds(1:end-1),pv)
xlabel('Model output')
ylabel('Probability')

plot(t_vel_range,pv(40,:))
% pv is a lookup table of form [response level, velocity probabilities]

% Smooth prob_R_v with a length = 5 gaussian

smoother = gaussmf([-2 -1 0 1 2],[1,0]);
smoother=smoother(:);
prob_R_v=NaN(size(pv));

for k=1:numcols
    for t=3:numrows-2
        prob_R_v(t,k) = sum(smoother.*(pv(t-2:t+2,k)));
    end
    % Deal with edges
    prob_R_v(1,k) = sum(smoother(3:5).*pv(1:3,k));
    prob_R_v(2,k) = sum(smoother(2:5).*pv(1:4,k));
    prob_R_v(numrows-1,k) = sum(smoother(2:5).*pv(end-3:end,k));
    prob_R_v(numrows,k) = sum(smoother(3:5).*pv(end-2:end,k));
end

resp_thresholds = a.resp_thresholds;

% Ensure columns sum to 1
prob_R_v = prob_R_v ./ repmat(sum(prob_R_v,1),numrows,1);

save([output_dir 'obs_model_' num2str(output_integration) '_botanic.mat'],'prob_R_v','t_vel_range','resp_thresholds');

