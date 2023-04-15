clear all
close all

% Want to make sure there's no difference between using normpdf or gaussmf

a=rand([100,1]); % representing the input prob weights

b=randi(100,[100 1]);

pb = gaussmf(b, [10 50]);
pc = normpdf(b, 50, 10);

r1 = pb.*a;
r2 = pc.*a;

r1n = r1 / sum(r1(:));
r2n = r2 / sum(r2(:));

% In the end the differences come down to the level of numerical precision