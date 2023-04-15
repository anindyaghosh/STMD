clear all
close all

impath = 'D:\simfiles\texture\HDR_Botanic_RGB.png';
t_width = 1; % Degrees
t_height = 1;
t_value = [0 0.5];
hfov = 40;
vfov = 20;
fwhm = 1.4; % blur half-width, in degrees
frames=1;
delay=1;
Ts=0.001;
drawmode = 'pixels';
obs_th = 0; % Angle for the (center?) of the visual field
t_al_pos = [0 9]; % Vertical angle, 1 entry per target
t_th_pos = [0 10]; % Horizontal angle, 1 entry per target (absolute angle)
pixel_per_degree = 12;
degree_per_PR = 1;

% Draw setup
           [V,...
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
    fwhm);

t_th_rel=t_th_pos-repmat(obs_th,size(t_th_pos));

h_pixelpos=round(hdist*tand(t_th_rel)+hpix/2);
v_pixelpos=round(hdist*tand(t_al_pos)./cosd(t_th_rel)+vpix/2);

ind_logic=false(vpix,hpix);

% Start rendering a frame here

% Background
offset=floor(obs_th/360*panpix);
ind=sub2ind(size(src),V(:),round(offset+H(:))); % Select the appropriate pixels to fill the background
ind(isnan(ind))=1;
img=reshape(src(ind),vpix,hpix); % img now contains the background

img_white = ones(size(img)); % Pure white background

for t=1:numel(t_th_pos) % For each target
    if(t_th_rel(t) > -hfov/2 && t_th_rel(t) < hfov/2 && t_al_pos(t) < vfov/2 && t_al_pos(t) > -vfov/2) % Check whether the target should be drawn
        if(strcmp(drawmode,'pixels')) %In this mode, the target has a constant pixel size, which means the angles change as it goes. Movement is by angles.
            % Draw the target
            x_left= max(h_pixelpos(t)-t_pixelleft,1);
            x_right= min(h_pixelpos(t)+t_pixelright,hpix);
            y_up= max(v_pixelpos(t)-t_pixelup,1);
            y_down= min(v_pixelpos(t)+t_pixeldown,vpix);
            ind_logic=false(size(ind_logic));
            ind_logic(y_up:y_down,x_left:x_right)=true;
        end
        img(ind_logic)=t_value(t);
        img_white(ind_logic)=t_value(t);
    end
end

f=figure(1);
imshow(img)

frame=zeros(size(h));
frame_white=frame;

for t=1:numel(h)
    ref_h=h(t);
    ref_v=v(t);
    frame(t)=sum(sum(img(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
        ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));
    frame_white(t)=sum(sum(img_white(ref_v-floor(kernel_size/2):ref_v+floor(kernel_size/2),...
        ref_h-floor(kernel_size/2):ref_h+floor(kernel_size/2)).*kernel));
end

% Blurred image
f=figure(2);
imshow([frame;frame_white],'InitialMagnification',1000)


