clear all
close all

datafile1=load('E:\PHD\TEST\temp_lost_it_19_10_22.mat');
datafile2=load('E:\PHD\TEST\lost_it19_10_31_set_ 2.mat');
datafile3=load('E:\PHD\TEST\lost_it19_10_31_set_ 3.mat');

paramfile1=load('E:\PHD\TEST\TrackerParamFile19_10_03.mat');
paramfile2=load('E:\PHD\TEST\TrackerParamFile19_10_30.mat');
paramfile3=load('E:\PHD\TEST\TrackerParamFile19_11_01.mat');

get_range = 1:10;
% Low gain, my implementation
fac_mask = paramfile1.outer_var.sets.tracker_num(paramfile1.outer_ix) == 3; % fac
facdark_mask = paramfile1.outer_var.sets.tracker_num(paramfile1.outer_ix) == 4; % fac

p13 = datafile1.logs.ispresent(fac_mask,get_range);
p14 = datafile1.logs.ispresent(facdark_mask,get_range);

d13 = datafile1.logs.t_lost_it.first(fac_mask,get_range);
d14 = datafile1.logs.t_lost_it.first(facdark_mask,get_range);

% If the first loss is nan it indicates that the target was never lost, if
% the run is complete
d13(isnan(d13) & ~isnan(p13)) = 30000;
d14(isnan(d14) & ~isnan(p14)) = 30000;

% High gain, my implementation
fac_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 1; % fac
facdark_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 2; % fac

p21 = datafile2.logs.ispresent(fac_mask,get_range);
p22 = datafile2.logs.ispresent(facdark_mask,get_range);

d21 = datafile2.logs.t_lost_it.first(fac_mask,get_range);
d22 = datafile2.logs.t_lost_it.first(facdark_mask,get_range);

d21(isnan(d21) & ~isnan(p21)) = 30000;
d22(isnan(d22) & ~isnan(p22)) = 30000;

% ZB implementation
fac_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 5; % fac
facdark_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 6; % fac

p25 = datafile2.logs.ispresent(fac_mask,get_range);
p26 = datafile2.logs.ispresent(facdark_mask,get_range);

d25 = datafile2.logs.t_lost_it.first(fac_mask,get_range);
d26 = datafile2.logs.t_lost_it.first(facdark_mask,get_range);

d25(isnan(d25) & ~isnan(p25)) = 30000;
d26(isnan(d26) & ~isnan(p26)) = 30000;

% ZB implementation blanking out turns
fac_mask = paramfile3.outer_var.sets.tracker_num(paramfile3.outer_ix) == 1; % fac
facdark_mask = paramfile3.outer_var.sets.tracker_num(paramfile3.outer_ix) == 2; % fac

p31 = datafile3.logs.ispresent(fac_mask,get_range);
p32 = datafile3.logs.ispresent(facdark_mask,get_range);

d31 = datafile3.logs.t_lost_it.first(fac_mask,get_range);
d32 = datafile3.logs.t_lost_it.first(facdark_mask,get_range);

d31(isnan(d31) & ~isnan(p31)) = 30000;
d32(isnan(d14) & ~isnan(p32)) = 30000;

% Const gain, low gain
cg_mask = paramfile1.outer_var.sets.tracker_num(paramfile1.outer_ix) == 7; % constgain
cgdark_mask = paramfile1.outer_var.sets.tracker_num(paramfile1.outer_ix) == 8; % constgain_dark

p17 = datafile1.logs.ispresent(cg_mask,get_range);
p18 = datafile1.logs.ispresent(cgdark_mask,get_range);

d17 = datafile1.logs.t_lost_it.first(cg_mask,get_range);
d18 = datafile1.logs.t_lost_it.first(cgdark_mask,get_range);

d17(isnan(d17) & ~isnan(p17)) = 30000;
d18(isnan(d18) & ~isnan(p18)) = 30000;

% Const gain, high gain
cg_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 3; % constgain
cgdark_mask = paramfile2.outer_var.sets.tracker_num(paramfile2.outer_ix) == 4; % constgain_dark

p23 = datafile2.logs.ispresent(cg_mask,get_range);
p24 = datafile2.logs.ispresent(cgdark_mask,get_range);

d23 = datafile2.logs.t_lost_it.first(cg_mask,get_range);
d24 = datafile2.logs.t_lost_it.first(cgdark_mask,get_range);

d23(isnan(d23) & ~isnan(p23)) = 30000;
d24(isnan(d24) & ~isnan(p24)) = 30000;

disp(['d13: ' num2str(sum(isnan(d13)))])
disp(['d14: ' num2str(sum(isnan(d14)))])
disp(['d21: ' num2str(sum(isnan(d21)))])
disp(['d22: ' num2str(sum(isnan(d22)))])
disp(['d25: ' num2str(sum(isnan(d25)))])
disp(['d26: ' num2str(sum(isnan(d26)))])
disp(['d31: ' num2str(sum(isnan(d31)))])
disp(['d32: ' num2str(sum(isnan(d32)))])

sum(isnan([d13(:);d14(:);d21(:);d22(:);d25(:);d26(:);d31(:);d32(:)]))

figure(1)
plot(sort([d13(:) d14(:) d21(:) d22(:) d25(:) d26(:) d31(:) d32(:)]))
legend({'Mine1','Mine2','High1','High2','ZB1','ZB2','ZB_turn1','ZB_turn2'})

figure(2)
plot(sort([d13(:) d14(:) d31(:) d32(:)]))
legend({'Mine1','Mine2','ZB_turn1','ZB_turn2'},'Interpreter','none')

% It looks like my implementation with high gain had reasonable performance
% in some of the challenging situations although seems to have missed some
% of the easy ones.


% Const gain
figure(3)
plot(sort([d17(:) d18(:) d23(:) d24(:)]))
legend({'Low1','Low2','High1','High2'},'Interpreter','none')