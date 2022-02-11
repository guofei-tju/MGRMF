function [K_poly] = getPolyKernel(y,Power)

dotx  = y*y';
K_poly = (dotx+1).^Power;
end

