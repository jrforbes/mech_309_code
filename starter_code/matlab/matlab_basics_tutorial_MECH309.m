%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% James Richard Forbes
% 2018/01/02

% This is a piece of sample code written by me, Prof. Forbes. Some of this 
% sample code is take from, or modified from, Prof. Legrand. 
% The purpose of this sample code is to show students how to do some basic 
% things in MATLAB. 

% If you're unsure about what a function does, just type ``help XXX'' in
% the command windown. For example, type ``help roots" to find out about
% the function ``roots".

% Advanced functions and toolboxes, such as MATLAB's symbolic toolbox, are 
% not to be used in this course unless you're given permission to do so. 
% For example, the code ``x = A\b" finds x given A and b (i.e., solves 
% A*x = b), but in this course you are going to code your own linear-
% system solver. As such, if you're given a question where you must solve
% A*x = b, unless specified, you must code your own linear-system solver,
% and not just use the backslash command. 

% Functions should be used as much as possilble to create modular, well
% orginized, code. 

% To stop execution of a script or comand, press Ctrl+C or Ctrl+Break. 
% On Apple platforms, you also can use Command+. (the Command key and the 
% period key).

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Getting started. Clean up workspace. 

clear all   % Clears all variables.
clc         % Clears the command window.
close all   % Closes all figures/plots.

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign existing variables
format short % default setting
a = pi;    % Semi-colon does not print statement
b = exp(1) % No semi-colon prints it in the command window
% For the sake of exposition, semi-colons are not used much in this code,
% but in genearl, use them!
format long % Presents numerical values in a ``long" format (i.e., 15 decimal places).
a
b

% return    % This stops the code here. If you want to run the code below, 
            % just comment out the ``return" using a `` % ''.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic Operations
% commands: +, -, *, /, ^, sqrt
clc

d = a + b
e = a - b
f = b*a
g = b/a
h = a^5 % exponents
m = sqrt(5)

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rounding
% commands: round, ceil, floor
clc

a = 10.5;
b = 1.51;
round_a = round(a) % rounds towards nearest decimal or integer.
ceiling_a = ceil(a) % rounds to the nearest integers towards infinity.
floor_a = floor(a) % rounds to the nearest integers towards minus infinity.

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If statements
% commands: if, elseif, else, end
clc

if (b < 1.5)
    c = 0.5555555;
elseif (b > 100)
    c = 100;
else
    c = 5;
end

% return

% and, or, not operators
% and operator
if ((b > 1.5) && (a < 10)) % AND
    c = 999999;
end
c

if ((b > 1.5) || (a < 10)) % OR
    c = 888888;
end
c

% equal to
if (c == 888888) % equal to, be CAREFUL, do not confuse with one equal sign
    d = 333;
else
    d = 555;
end
d

% not equal to
if (c ~= 888888) % NOT equal
    d = 777;
else
    d = 2222;
end

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loops
% commands: for, end, while
clc

% Note, do not use i and j as loop counters. i and j are both reserved in
% MATLAB to represent imaginary numbers. 
% I suggest using ii, jj, or lv1, lv2 where ``lv1" means ``loop variable
% 1". 

% For loop
a = 0
for lv1 = 1:5 % colon is a slicing operation seen later
    a = a + lv1;
end
% a

% return

% while loop; very useful for iterative methods
b = 1;
tolerance = 1e-4;
iterations = 1;
while ((b > tolerance) && (iterations < 1000))
    % Careful with infinite loops
    % If your program runs for a very very long time, the error is probably
    % in the condition statement of your while loop
    % I suggest always having a ``terminal count" conditions, that is, the
    % condition ``&& (iterations < 1000)".
    
    b = b/2;
    iterations = iterations + 1;
end
b

% For loop with if statements
div_by_2 = [];
div_by_5 = [];
for lv1 = 1:10
    
    if rem(lv1,2) == 0
        div_by_2 = [div_by_2; lv1];
        
    elseif rem(lv1,5) == 0        
        div_by_5 = [div_by_5; lv1];
        
    else
       % Do nothing.
       
    end   
end
div_by_2
div_by_5

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matricies
% commands: [], ', det, rank, null, rref, orth, cond, eig, eigs, svd
clear all
clc

% Create some row, column, and full matrices.
col_mat = [pi; exp(1); 2^5]
row_mat = [1 2 3]
mat1 = [1 2 1; 4 5 6; 7 8 9]
mat2 = [11 12 13; ... % Continuation line can make code more readable
        14 15 16; ...
        17 18 19]
    
% Access element of matrices
% Indices start at 1 in MATLAB
element_col_mat = col_mat(2)
element_col_mat = col_mat(2,1)
element_mat_row2_col3 = mat2(2,3)

% Transpose a matrix
mat2_transpose = mat2'

% Addition and substraction
vec_sum = col_mat + row_mat' % Correct dimensions!
mat_sum = mat1 + mat2

% Scalar multiplication
scale_mat = 2*mat1
% Add to all entries
add_100_to_mat1 = mat1 + 100

% Matrix multiplication
mat_times_col_mat = mat1*col_mat
% mat_times_row_vec = mat1 * row_mat % Should NOT work, wrong dimensions

% Matrix-matrix multiplication
mat_mat_mult = mat1*mat2

% Element-wise operations
mat_element_mult = mat1.*mat2

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving Ax = b, and inverses
% commands: \, inv
clc

A = mat1;
b = col_mat;
x = inv(A)*b % Bad inverse
x = A\b % Good inverse

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Slicing (colon)
% commands: ``:"
clc

% This is the section that you will benefit the most from playing around
x = [1:5]'
y = 1:2:20
z = 50:-5:10
% Make a column matrix
xyz = [0:10:40]'
mat1 = [x 2*x x/2]

% Access elements with slices
% 3rd row of mat1, last 2 columns
thirdrow = mat1(3,2:3)
% Or use keyword 'end'
secondrow = mat1(2,end-1:end)

% return

% If you want the entire row/column, you can leave slice empty
secondcol = mat1(:,2)

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-allocation
% commands: zeros, eye
clc

Z = zeros(2,2) % Useful to allocate memory, we will see that later
uno = ones(1,10)
identity = eye(5)

% disp('Starting to evaluate matrix...')
% tic
% for ii = 1:2200
%     for jj = 1:2200
%         matrix1(ii,jj) = ii + jj;
%     end
% end
% toc
% disp('Done evaluating matrix. This took a while')

% return

% clc

disp('Try again with pre-allocation...')
tic
matrix2 = zeros(2200,2200);
for ii = 1:2200
    for jj = 1:2200
        matrix2(ii,jj) = ii + jj;
    end
end
toc
disp('Done evaluating matrix. Much faster')

% return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting, with a while loop example
% commands: plot, figure, subplot, xlabel, ylabel, legend, surf, contour
clc

% First, create font size, line size, and line width variables. 
% Do not hand in plots without clear (large) labels.
font_size = 15;
line_size = 15;
line_width = 2;

% Create something to plot. 
n = 500;
t_max = 10;
t = linspace(0,t_max,n); % n points between 0 and t_max.

% Generate the output of a function.
lv1 = 1; % loop counter (``loop variable 1") for while loop
while (lv1 <= length(t))
    
    y(lv1) = f_non(t(lv1)); % evaluate the function at t(lv1); this function is a the bottom of this script. Don't do this, unless you have a really good reason!
    [z(lv1),z_dot(lv1),should_be_zero(lv1)] = f_nonlinear(t(lv1),y(lv1)); % This function is called as a funciton. Do this!
    lv1 = lv1 + 1; % update counter
    
end

% Plot y and z vs time.
figure
plot(t,y,'Linewidth',line_width);
hold on
plot(t,z,'--m','Linewidth',line_width);
xlabel('Time (s)','fontsize',font_size,'Interpreter','latex');
ylabel('Output $y$ and $z$ (units)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
legend({'$y$','$z$'},'fontsize',20,'Interpreter','latex')
grid on
print('output_vs_time','-depsc','-r720');

% Have two plots, in a ``1 x 2" matrix
figure
subplot(1,2,1) 
plot(t,z,'Linewidth',line_width);
hold on
xlabel('Time (s)','fontsize',font_size,'Interpreter','latex');
ylabel('Output $z$ (units)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
subplot(1,2,2)
plot(t,z_dot,'Linewidth',line_width);
hold on
xlabel('Time (s)','fontsize',font_size,'Interpreter','latex');
ylabel('Output $\dot{z}$ (units/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
% print('output_vs_time_1x2','-depsc','-r720');

figure
semilogy(t,should_be_zero,'Linewidth',line_width);
hold on
xlabel('Time (s)','fontsize',font_size,'Interpreter','latex');
ylabel('$\log_{10}(e)$','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
% print('log_error_vs_time','-depsc','-r720');


%% A function
function f_out = f_non(t)

    f_out = 10*exp(-t/2)*sin(2*t);
    
end

