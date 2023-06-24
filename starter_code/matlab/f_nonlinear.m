function [z,z_dot,should_be_zero] = f_nonlinear(t,y)
% Summary of this function goes here
% Detailed explanation goes here

z = -10*exp(-t/2)*sin(2*t) + 0.05*randn(1);
z_dot = 10/2*exp(-t/2)*sin(2*t) - 20*exp(-t/2)*cos(2*t);
should_be_zero = abs(z + y);


end

