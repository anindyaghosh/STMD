%JJ_Draw_setup_17_06_12
% The idea behind this function is that most of the work of drawing a frame
% can be done once as a setup or initialisation step.
% So, this function will do that work and return the relevant angles
% matrices and so forth for the frame renderer to use
%
% OUTPUTS
% V_coarse, V_fine, H_coarse, H_fine - These map the pixels of the source
% image (i.e. the panorama) to the flattened image.
% vpix_coarse, hpix_coarse, vpix_fine, hpix_fine - The size of the
% flattened image
% src - The source panorama
% panpix - The horizontal dimension of the source panorama
% kernel_dim_fine, kernel_dim_coarse - The blur kernel sizes
% fine_h, fine_v, coarse_h, coarse_v - Which pixels of the flattened image
% are sampled (subsampling)
% kernel_fine, kernel_coarse - the blur kernels
% alp_coarse, alp_fine - Which vertical angle the pixels of the flattened
% image correspond to.
% theta_coarse, theta_fine - Which horizontal angle the pixels of the
% flattened image correspond to
% blurbound_coarse, blurbound_fine - How many pixels to add to allow for
% valid blurring
% hdist_coarse, hdist_fine - See RenderRepeatPath18_06_12 for usage.
% t_pixelleft_coarse, t_pixelright_coarse
% t_pixeldown_coarse, t_pixelup_coarse, t_pixelleft_fine,
% t_pixelright_fine, t_pixeldown_fine, t_pixelup_fine - How many pixels to
% the left/right/up/down to colour in the target
% colour.
%
% INPUTS
% hfov_coarse, vfov_coarse, hfov_fine, vfov_fine - Horz/Vert field of view
% in degrees
% impath - Full path to panoramic image
% t_width, t_height - Target dimensions in degrees
% pixel_per_degree - When flattening image, use this many pixels per degree
% of the field of view
% degree_per_PR_coarse, degree_per_PR_fine - Degrees per photoreceptor
% (relevant to subsampling)
% fwhm_coarse, fwhm_fine - Full width at half maximum of blur kernels

function [  V,...
            H,...
            vpix,...
            hpix,...
            src,...
            panpix,...
            kernel_size,...
            h,...
            v,...
            kernel,...
            alp,...
            theta,...
            blurbound,...
            hdist,...
            t_pixelleft,...
            t_pixelright,...
            t_pixeldown,...
            t_pixelup]=JJ_Draw_Setup19_09_05(...
    hfov,...
    vfov,...
    impath,...
    t_width,...
    t_height,...
    pixel_per_degree,...
    degree_per_PR,...
    fwhm)

pixel_per_PR=degree_per_PR*pixel_per_degree;

hPR= hfov/degree_per_PR;% Number of photoreceptors, horizontal dimension
vPR= vfov/degree_per_PR;% "" "" vertical dimension

% Enforce integer values for hPR, vPR, pixel_per_PR
if(hPR ~= floor(hPR))
    disp('Non-integer hPR_coarse. Setting hPR_coarse = floor(hPR_coarse)');
    hPR=floor(hPR);
end
if(vPR ~= floor(vPR))
    disp('Non-integer vPR_coarse. Setting vPR_coarse = floor(vPR_coarse)');
    vPR=floor(vPR);
end
if(pixel_per_PR ~= ceil(pixel_per_PR))
    disp('Warning: pixel_per_PR_coarse not an integer. Setting pixel_per_PR_coarse=ceil(pixel_per_PR_coarse)')
    pixel_per_PR=ceil(pixel_per_PR);
end

sigma_deg = fwhm/(2*(2*log(2))^0.5); % Guassian blur std dev in degrees
sigma_pixel = sigma_deg*pixel_per_degree; % how many pixels per std dev

% Force kernel_size to be odd
kernel_size = 2*(ceil(sigma_pixel));
if(~mod(kernel_size,2))
    kernel_size=kernel_size+1;
end

% 
t_pixelwidth=round(pixel_per_PR*t_width); % t_width is in degrees, so convert to pixels
t_pixelheight=round(pixel_per_PR*t_height);

t_pixelleft=floor(t_pixelwidth/2); % how many pixels to color black to the left
t_pixelright=t_pixelwidth-t_pixelleft-1; % how many pixels to color black to the right of the centre

t_pixeldown=floor(t_pixelheight/2); % how many pixels to color black below the centre 
t_pixelup=t_pixelheight-t_pixeldown-1;% how many pixels to color black above the centre

blurbound=floor(kernel_size/2);
hpix=1+(hfov-1)*pixel_per_PR+2*blurbound;
vpix=1+(vfov-1)*pixel_per_PR+2*blurbound;

% Load image
src=imread(impath);
[panheight,panpix,~]=size(src);
r=panpix/(2*pi);
src=src(:,:,2); %green only
src=double(horzcat(src,src,src))/255; % replicate the image horizontally to ensure against reading out of bounds & normalise

% Generate the horizontal and vertical angles of the "screen"
% The screen is defined with equal linear distance spacing between pixels
dxor=linspace(-sind(hfov/2),sind(hfov/2),hpix);
dyor=linspace(-sind(vfov/2),sind(vfov/2),vpix)';

theta=atand(dxor/cosd(hfov/2));

f=1./(2^0.5*cosd(theta)); %normalised
f=repmat(1./f,vpix,1);

Y=repmat(dyor,1,hpix);
alp=atand(Y.*f);

V=round(r*tand(alp))+panheight/2; % which vertical pixel
H=round(r*theta/180*pi)+panpix;
H=repmat(H,vpix,1);

mask=logical((V>panheight) + (V<1)); %identify invalid pixels
V(mask)=NaN;

% % Set up "blurring", which is actually implemented by integrating the
% % correlation of a gaussian kernel with part of the image.
% kernel_dim=ceil(2*sigma_pixel);
% kernel_dim=kernel_dim+mod(kernel_dim+1,2);

[x,y]=meshgrid(...
    -floor(kernel_size/2):floor(kernel_size/2),...
    -floor(kernel_size/2):floor(kernel_size/2));
d=(x.^2 + y.^2).^0.5;
kernel=gaussmf(d,[sigma_pixel 0]);
kernel=kernel/sum(kernel(:));

im_ss_h=(1+blurbound):...
    pixel_per_PR:...
    hpix-blurbound;
im_ss_h=round(im_ss_h);

im_ss_v=1+blurbound:...
    pixel_per_PR:...
    vpix-blurbound;
im_ss_v=round(im_ss_v);

[h,v]=meshgrid(im_ss_h,im_ss_v);

hdist=hpix/2/tand(hfov/2);

return
