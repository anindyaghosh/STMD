clear all
close all

timelen=2000;

Qval=1;
Rval=0.01;

A=eye(2);
H=eye(2);

x_initial=[10;10];
x_post=x_initial;
x_priori=x_initial;

x_actual=[1,2];

P_post=eye(2);
Q=Qval*eye(2);
R=Rval*eye(2);

x_hist=zeros(timelen,2);
z_hist=x_hist;
confidence_hist=zeros(numel(ssgt.gtbuttonfill),1);

confidence=0;

ddx=[1;1];

for t=1:timelen
    % Generate signal with perturbation
    x_actual = A*x_actual+ddx;
    
    isrel=rand(100);
    if(isrel > 40)
        z=x_actual;
    else
        z=[randi(40);randi(40)];
    end
    
    if( ((z(1)-x_post(1))^2 + (z(2)-x_post(2))^2)^0.5 < reliability_threshold)
        guess_reliable=true;
        confidence=min(confidence+1,cmval);
    else
        guess_reliable=false;
        confidence=max(0,confidence-1);
    end
    
    
end


