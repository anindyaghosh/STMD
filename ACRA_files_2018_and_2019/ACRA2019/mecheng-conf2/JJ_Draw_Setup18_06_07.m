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

function [  V_coarse,...
            V_fine,...
            H_coarse,...
            H_fine,...
            vpix_coarse,...
            hpix_coarse,...
            vpix_fine,...
            hpix_fine,...
            src,...
            panpix,...
            kernel_dim_fine,...
            kernel_dim_coarse,...
            fine_h,...
            fine_v,...
            coarse_h,...
            coarse_v,...
            kernel_fine,...
            kernel_coarse,...
            alp_coarse,...
            alp_fine,...
            theta_coarse,...
            theta_fine,...
            blurbound_coarse,...
            blurbound_fine,...
            hdist_coarse,...
            hdist_fine,...
            t_pixelleft_coarse,...
            t_pixelright_coarse,...
            t_pixeldown_coarse,...
            t_pixelup_coarse,...
            t_pixelleft_fine,...
            t_pixelright_fine,...
            t_pixeldown_fine,...
            t_pixelup_fine]=JJ_Draw_Setup18_06_07(...
    hfov_coarse,...
    vfov_coarse,...
    hfov_fine,...
    vfov_fine,...
    impath,...
    t_width,...
    t_height,...
    pixel_per_degree,...
    degree_per_PR_coarse,...
    degree_per_PR_fine,...
    fwhm_coarse,...
    fwhm_fine)

pixel_per_PR_coarse=degree_per_PR_coarse*pixel_per_degree;
pixel_per_PR_fine=degree_per_PR_fine*pixel_per_degree;

hPR_coarse= hfov_coarse/degree_per_PR_coarse;% Number of photoreceptors, horizontal dimension
vPR_coarse= vfov_coarse/degree_per_PR_coarse;% "" "" vertical dimension
hPR_fine= hfov_fine/degree_per_PR_fine;
vPR_fine= vfov_fine/degree_per_PR_fine;

if(hPR_coarse ~= floor(hPR_coarse))
    disp('Non-integer hPR_coarse. Setting hPR_coarse = floor(hPR_coarse)');
    hPR_coarse=floor(hPR_coarse);
end
if(vPR_coarse ~= floor(vPR_coarse))
    disp('Non-integer vPR_coarse. Setting vPR_coarse = floor(vPR_coarse)');
    vPR_coarse=floor(vPR_coarse);
end
if(hPR_fine ~= floor(hPR_fine))
    disp('Non-integer hPR_fine. Setting hPR_fine = floor(hPR_fine)');
    hPR_fine=floor(hPR_fine);
end
if(vPR_fine ~= floor(vPR_fine))
    disp('Non-integer vPR_fine. Setting vPR_fine = floor(vPR_fine)');
    vPR_fine=floor(vPR_fine);
end

if(pixel_per_PR_coarse ~= ceil(pixel_per_PR_coarse))
    disp('Warning: pixel_per_PR_coarse not an integer. Setting pixel_per_PR_coarse=ceil(pixel_per_PR_coarse)')
    pixel_per_PR_coarse=ceil(pixel_per_PR_coarse);
end
if(pixel_per_PR_fine ~= ceil(pixel_per_PR_fine))
    disp('Warning: pixel_per_PR_fine not an integer. Setting pixel_per_PR_fine=ceil(pixel_per_PR_fine)')
    pixel_per_PR_fine=ceil(pixel_per_PR_fine);
end

sigma_deg_coarse = fwhm_coarse/(2*(2*log(2))^0.5); % Degrees
sigma_deg_fine = fwhm_fine/(2*(2*log(2))^0.5);

sigma_pixel_coarse = sigma_deg_coarse*pixel_per_degree; % how many pixels per std dev
sigma_pixel_fine = sigma_deg_fine*pixel_per_degree;

kernel_size_coarse = 2*(ceil(sigma_pixel_coarse));
if(~mod(kernel_size_coarse,2))
    kernel_size_coarse=kernel_size_coarse+1;
end

kernel_size_fine = 2*(ceil(sigma_pixel_fine));
if(~mod(kernel_size_fine,2))
    kernel_size_fine=kernel_size_fine+1;
end

t_pixelwidth_coarse=round(pixel_per_PR_coarse*t_width); % t_width is in degrees, so convert to pixels
t_pixelheight_coarse=round(pixel_per_PR_coarse*t_height);

t_pixelwidth_fine=round(pixel_per_degree*t_width);
t_pixelheight_fine=round(pixel_per_degree*t_height); 

t_pixelleft_coarse=floor(t_pixelwidth_coarse/2); % how many pixels to color black to the left
t_pixelright_coarse=t_pixelwidth_coarse-t_pixelleft_coarse-1; % how many pixels to color black to the right of the centre

t_pixeldown_coarse=floor(t_pixelheight_coarse/2); % how many pixels to color black below the centre 
t_pixelup_coarse=t_pixelheight_coarse-t_pixeldown_coarse-1;% how many pixels to color black above the centre

t_pixelleft_fine=floor(t_pixelwidth_fine/2); 
t_pixelright_fine=t_pixelwidth_fine-t_pixelleft_fine-1; 

t_pixeldown_fine=floor(t_pixelheight_fine/2);
t_pixelup_fine=t_pixelheight_fine-t_pixeldown_fine-1;


blurbound_coarse=floor(kernel_size_coarse/2);
hpix_coarse=1+(hfov_coarse-1)*pixel_per_PR_coarse+2*blurbound_coarse;
vpix_coarse=1+(vfov_coarse-1)*pixel_per_PR_coarse+2*blurbound_coarse;

blurbound_fine=floor(kernel_size_fine/2);
hpix_fine=1+(hPR_fine-1)*pixel_per_PR_fine+2*blurbound_fine;
vpix_fine=1+(vPR_fine-1)*pixel_per_PR_fine+2*blurbound_fine;

% Load image
src=imread(impath);
[panheight,panpix,~]=size(src);
r=panpix/(2*pi);
src=src(:,:,2); %green only
src=double(horzcat(src,src,src))/255; % replicate the image horizontally to ensure against reading out of bounds & normalise

% Generate the horizontal and vertical angles of the "screen"
% The screen is defined with equal linear distance spacing between pixels
dxor_coarse=linspace(-sind(hfov_coarse/2),sind(hfov_coarse/2),hpix_coarse);
dyor_coarse=linspace(-sind(vfov_coarse/2),sind(vfov_coarse/2),vpix_coarse)';
dxor_fine=linspace(-sind(hfov_fine/2),sind(hfov_fine/2),hpix_fine);
dyor_fine=linspace(-sind(vfov_fine/2),sind(vfov_fine/2),vpix_fine)';

theta_coarse=atand(dxor_coarse/cosd(hfov_coarse/2));
theta_fine=atand(dxor_fine/cosd(hfov_fine/2));

f_coarse=1./(2^0.5*cosd(theta_coarse)); %normalised
f_fine=1./(2^0.5*cosd(theta_fine)); %normalised

f_coarse=repmat(1./f_coarse,vpix_coarse,1);
f_fine=repmat(1./f_fine,vpix_fine,1);

Y_coarse=repmat(dyor_coarse,1,hpix_coarse);
Y_fine=repmat(dyor_fine,1,hpix_fine);

alp_coarse=atand(Y_coarse.*f_coarse);
alp_fine=atand(Y_fine.*f_fine);

V_coarse=round(r*tand(alp_coarse))+panheight/2; % which vertical pixel
V_fine=round(r*tand(alp_fine))+panheight/2;

H_coarse=round(r*theta_coarse/180*pi)+panpix;
H_coarse=repmat(H_coarse,vpix_coarse,1);
H_fine=round(r*theta_fine/180*pi)+panpix;
H_fine=repmat(H_fine,vpix_fine,1);

mask_coarse=logical((V_coarse>panheight) + (V_coarse<1)); %identify invalid pixels
V_coarse(mask_coarse)=NaN;
mask_fine=logical((V_fine>panheight) + (V_fine<1)); 
V_fine(mask_fine)=NaN;

% Set up "blurring", which is actually implemented by integrating the
% correlation of a gaussian kernel with part of the image.
kernel_dim_coarse=ceil(2*sigma_pixel_coarse);
kernel_dim_coarse=kernel_dim_coarse+mod(kernel_dim_coarse+1,2);

kernel_dim_fine=ceil(2*sigma_pixel_fine);
kernel_dim_fine=kernel_dim_fine+mod(kernel_dim_fine+1,2);

[x_fine,y_fine]=meshgrid(...
    -floor(kernel_dim_fine/2):floor(kernel_dim_fine/2),...
    -floor(kernel_dim_fine/2):floor(kernel_dim_fine/2));
d_fine=(x_fine.^2 + y_fine.^2).^0.5;
kernel_fine=gaussmf(d_fine,[sigma_pixel_fine 0]);
kernel_fine=kernel_fine/sum(kernel_fine(:));

[x_coarse,y_coarse]=meshgrid(...
    -floor(kernel_dim_coarse/2):floor(kernel_dim_coarse/2),...
    -floor(kernel_dim_coarse/2):floor(kernel_dim_coarse/2));
d_coarse=(x_coarse.^2 + y_coarse.^2).^0.5;
kernel_coarse=gaussmf(d_coarse,[sigma_pixel_coarse 0]);
kernel_coarse=kernel_coarse/sum(kernel_coarse(:));

im_ss_fine_h=1+blurbound_fine:...
    pixel_per_PR_fine:...
    hpix_fine-blurbound_fine;
im_ss_fine_h=round(im_ss_fine_h);

im_ss_fine_v=1+blurbound_fine:...
    pixel_per_PR_fine:...
    vpix_fine-blurbound_fine;
im_ss_fine_v=round(im_ss_fine_v);

[fine_h,fine_v]=meshgrid(im_ss_fine_h,im_ss_fine_v);

im_ss_coarse_h=(1+blurbound_coarse):...
    pixel_per_PR_coarse:...
    hpix_coarse-blurbound_coarse;
im_ss_coarse_h=round(im_ss_coarse_h);

im_ss_coarse_v=1+blurbound_coarse:...
    pixel_per_PR_coarse:...
    vpix_coarse-blurbound_coarse;
im_ss_coarse_v=round(im_ss_coarse_v);

[coarse_h,coarse_v]=meshgrid(im_ss_coarse_h,im_ss_coarse_v);

hdist_coarse=hpix_coarse/2/tand(hfov_coarse/2);
hdist_fine=hpix_fine/2/tand(hfov_fine/2);
%
% noise_amp_fine=0.05;
% noise_amp_coarse=0.2;

% tic
% for obs_th_pos=0:1:360
%     offset=floor(obs_th_pos/360*panpix);
%     ind_coarse=sub2ind(size(src),V_coarse(:),round(offset+H_coarse(:)));
%     img_coarse=reshape(src(ind_coarse),vpix_coarse,hpix_coarse);
%     ind_fine=sub2ind(size(src),V_fine(:),round(offset+H_fine(:)));
%     img_fine=reshape(src(ind_fine),vpix_fine,hpix_fine);
%     
%     % Add noise
%     noise_fine=noise_amp_fine*(1-2*rand(size(img_fine)));
%     noise_coarse=noise_amp_coarse*(1-2*rand(size(img_coarse)));
% 
%     img_coarse=img_coarse+noise_coarse;
%     img_fine=img_fine+noise_fine;
% 
%     img_ss_fine=zeros(size(fine_h));
%     for t=1:numel(fine_h)
%         ref_h=fine_h(t);
%         ref_v=fine_v(t);
%         img_ss_fine(t)=sum(sum(img_fine(ref_v-floor(kernel_dim_fine/2):ref_v+floor(kernel_dim_fine/2),...
%             ref_h-floor(kernel_dim_fine/2):ref_h+floor(kernel_dim_fine/2)).*kernel_fine));
%     end
% 
%     
% 
%     img_ss_coarse=zeros(size(coarse_h));
%     for t=1:numel(coarse_h)
%         ref_h=coarse_h(t);
%         ref_v=coarse_v(t);
%         img_ss_coarse(t)=sum(sum(img_coarse(ref_v-floor(kernel_dim_coarse/2):ref_v+floor(kernel_dim_coarse/2),...
%             ref_h-floor(kernel_dim_coarse/2):ref_h+floor(kernel_dim_coarse/2)).*kernel_coarse));
%     end
% 
%     figure(1)
%     subplot(2,1,1)
%     imshow(img_ss_fine,'InitialMagnification',1000)
%     subplot(2,1,2)
%     imshow(img_ss_coarse,'InitialMagnification',1000)
%     drawnow
% end
% toc

return
