clear all
close all

Ts=0.001;

emd_time_constant = 0.040; % As per "An autonomous robot inspired by insect neurophysiology pursues moving features in natural environments"

emd_n1=Ts*emd_time_constant/(Ts*emd_time_constant+2);
emd_n2=Ts*emd_time_constant/(Ts*emd_time_constant+2);
emd_d2=-(Ts*emd_time_constant-2)/(Ts*emd_time_constant+2);

for p=26:49
    dataset=['D:\simfiles\conf2\Zahra datasets\4496768\' num2str(p) '\'];
    a=dlmread([dataset 'GroundTruth.txt']);
    
    refimg=imread([dataset 'IMG00001.jpg']);
    [vfov,hfov,~]=size(refimg);
    
    emd_ind = 1;
    emd_input_buffer = zeros(vfov,hfov,2);
    emd_output_buffer = emd_input_buffer;
    
    for k=1:10000
        subplot(2,1,1)
        try
            I=imread([dataset 'IMG' num2str(k,'%0.5d') '.jpg']);
            imshow(I)
            hold on
            b=rectangle('Position',[a(k,1) a(k,2) a(k,3) a(k,4)],'EdgeColor',[1 0 0]);
            drawnow
            hold off
            subplot(2,1,2)
            I_green=I(:,:,2);
            emd_input_buffer(:,:,emd_ind) = I_green;
            emd_output_buffer(:,:,emd_ind) = emd_n1*emd_input_buffer(:,:,emd_ind) +...
                emd_n2*emd_input_buffer(:,:,3-emd_ind)+...
                emd_d2*emd_output_buffer(:,:,3-emd_ind);
            
            % For rightwards: delayed left side with undelayed right
            emd_rw_correl = emd_output_buffer(1:end-1,:,emd_ind).*emd_input_buffer(2:end,:,emd_ind);
            emd_lw_correl = emd_output_buffer(2:end,:,emd_ind).*emd_input_buffer(1:end-1,:,emd_ind);
            
            emd_result = emd_rw_correl - emd_lw_correl;
            emd_result = max(emd_rw_correl,0);
            
            imagesc(emd_result)

            emd_ind = 3-emd_ind;
        catch
            break
        end
    end

end