clear all
close all

Ts=0.001;
num_distractors=2;

hfov=40;
vfov=20;

d=-0.1+0.2*rand(num_distractors,1); % Horizontal displacement

% Ensure that distractors at least initially sit inside the field of view

umin = abs(d)/tand(hfov/2);

timesteps=1000;
u=NaN(timesteps,num_distractors);
theta=u;

% For each horizontal displacement, there will be a minimum forward
% displacement such that the target is actually on the screen

u(1,:)=1.1*umin + 1 * rand(num_distractors,1); %forward displacement

theta(1,:) = atand(abs(d')./u(1,:));

v=-1; % The rate at which the pursuer is moving forward through the environment

for t=2:timesteps
   % Update u according to v
   
   u(t,:) = u(t-1,:)+v*Ts;
   theta(t,:) = atand(d'./u(t,:));
   
   for k=1:num_distractors
       if(abs(theta(t,k)) > hfov/2)
           u(t,k) = NaN;
       end
   end
end

subplot(2,1,1)
plot(theta)
subplot(2,1,2)
plot(d(:),u','x')