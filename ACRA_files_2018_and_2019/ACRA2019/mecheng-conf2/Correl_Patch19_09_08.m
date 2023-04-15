function res=Correl_Patch19_09_08(kernel,source,index)
    % Multiply patch by source at index and return result

    [iy,ix]=size(source);
    [py,px]=ind2sub([iy,ix],index);
    
    kerdim=size(patch);
    
    if(kerdim(1) ~= kerdim(2))
        error('Assumed square kernel')
    end
    if(mod(kerdim(1),2) == 0)
        error('Need odd kerdim)')
    end
    
    kerhalf=floor(kerdim / 2);
    kercentre = floor(kerdim/2)+1;
    
    if(px < kercentre) % Need to trim the kernel on the left side
        kerleft = kercentre-px+1;
        kerright = kerdim(2);
    elseif(ix - px < kerhalf) % trim on the right
        kerleft=1;
        kerright = kercentre + ix-px;
    else
        kerleft=1;
        kerright=kerdim(2);
    end
    
    if(py < kercentre)
        kertop = kercentre - py+1;
        kerbottom = kerdim(1);
    elseif(ix-py < kerhalf)
        kertop=1;
        kerbottom = kercentre + iy-py;
    else
        kertop = 1;
        kerbottom = kerdim(1);
    end
    
    src_inds_y = py - (kercentre-kertop) : py + (kerbottom - kercentre);
    src_inds_x = px - (kercentre-kerleft) : px + (kerright - kercentre);
    
    res=sum(kernel(kertop:kerbottom,kerleft:kerright) .* source(src_inds_y,src_inds_x));
    
end