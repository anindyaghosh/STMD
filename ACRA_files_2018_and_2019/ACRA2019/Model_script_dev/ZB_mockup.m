clear all
close all

settings.hfov=40;
settings.vfov=20;

nFrames = 500;
dataGreenChannel = randi(255,[480,720,nFrames]);

image_size_m = size(dataGreenChannel,1) ;                         %number of rows
image_size_n = size(dataGreenChannel,2);                         %number of columns

degrees_in_image = settings.hfov;

pixels_per_degree = image_size_n/degrees_in_image;                          %pixels_per_degree = #horiz. pixels in output / horizontal degrees (97.84)
pixel2PR = ceil(pixels_per_degree);                                         %ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling...                                                                
sigma_deg = 1.4/2.35; 
sigma_pixel = sigma_deg*pixels_per_degree;                                  %sigma = (sigma in degrees (0.59))*pixels_per_degree
kernel_size = 2*(ceil(sigma_pixel));
ParameterResize=ceil(image_size_m/pixel2PR);
K1=1:pixel2PR:image_size_m;
K2=1:pixel2PR:image_size_n;
YDim=length(K1);
XDim=length(K2);
Facilitation_Mode='on';

Facilitation_sigma=5/2.35482;
Ts = 0.05; %Sampling time

wb = 0.05;

FacilitationGain =25; %root of facilitation kernel gain
RC_wb = 5; 
parameter.RC_fac_wb = 1;
Threshold = 0.1;
Facilitation_Matrix=ones(YDim, XDim);
% TargetLocationRow=zeros(nFrames,1);
% TargetLocationCol=zeros(nFrames,1);
% framePTime=zeros(nFrames,1);
% blurPTime=zeros(nFrames,1);
Delay=200;

initialCol = 20;
initialRow = 10;

for i=1:nFrames+Delay
   if i>=2
       test=1;
       start=1;
   else
       test=0;
       start=0;
   end 
   if i>Delay
        Input=dataGreenChannel(:,:,i-Delay);
        Input=double(Input);
        
   else
       Input=ones(image_size_m,image_size_n);
   end
   
   tic;
   tStart = tic;
   opticOut=OpticBlur(Input, pixels_per_degree);
   tBlur = toc(tStart);
   SubSampledGreen=Subsample(opticOut, pixel2PR);
   PR_Output=PhotoReceptor(SubSampledGreen, XDim, YDim);
   LMC_Output=LMC(PR_Output, XDim,YDim);
   RTC_Output=RTC(LMC_Output, Ts);
   ESTMD_OUT=ESTMD(RTC_Output, Facilitation_Matrix, Facilitation_Mode, start, Threshold); 
   [Direction_Horizontal,Direction_Vertical]=Direction(ESTMD_OUT,  RC_wb, Ts,XDim, YDim);
   [ col_index, row_index] = Target_Location(ESTMD_OUT, YDim);
   [default, col_index2,row_index2] = Velocity_Vector(test, col_index, row_index,Direction_Horizontal,Direction_Vertical, XDim);
   if i<=Delay
       default=1;
   end
       
   if i<Delay+1
       col_index2=initialCol;
       row_index2=initialRow;
   end
   
   Grid=FacilitationGrid(col_index2,row_index2, ESTMD_OUT, Facilitation_sigma, FacilitationGain);
   Facilitation_Matrix=FacilitationMatrix(Grid, col_index2, default, Ts, wb);
   tFrame=toc;
   if i>Delay
%        TargetLocationRow(i-Delay)=row_index;
%        TargetLocationCol(i-Delay)=col_index;
   end
%    framePTime(i)=tFrame;
%    blurPTime(i)=tBlur; 
    f=figure(1);
    subplot(2,1,1)
    imagesc(Facilitation_Matrix)
    subplot(2,1,2)
    imagesc(dataGreenChannel(:,:,i))
    drawnow
end
