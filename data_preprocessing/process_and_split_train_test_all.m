clear all
ctime = datestr(now, 30);
tseed = str2double(ctime((end - 5) : end));
rand('seed', tseed);

load("..\data\comb_2_7186_all.mat");
num_test=200;

num_total=length(normal00);
rods=int64(length(normal00(1,:))/6);

% Seperate step data out
normal00=int64(normal00);
temp_normal_step=normal00(end-185:end,:);
normal00=normal00(1:end-186,:);
temp_TL_step=TL(end-185:end,:);
temp_TR_step=TR(end-185:end,:);
TL=TL(1:end-186,:);
TR=TR(1:end-186,:);

% Shuffle
idx = randperm(length(TL),num_test);
temp_normal00_test=normal00(idx,:);
normal00(idx,:)=[];
normal00=[normal00;temp_normal_step];

temp_TL_test=TL(idx,:);
TL(idx,:)=[];
temp_TR_test=TR(idx,:);
TR(idx,:)=[];

% Seperate int and float of spectrum
TL_float=[TL;temp_TL_step];
TR_float=[TR;temp_TR_step];
TL_TR_float=horzcat(TL_float, TR_float);

TL_int=int64(TL_float*1000);
TR_int=int64(TR_float*1000);
TL_TR_int=horzcat(TL_int, TR_int);

% Process and save train dataset
x=double(normal00(:, 1:6:end));
y=double(normal00(:, 2:6:end));
z=double(normal00(:, 3:6:end));
l=double(normal00(:, 4:6:end));
w=double(normal00(:, 5:6:end));
t=double(normal00(:, 6:6:end));
r=l./w;
% x_mean = mean(x, 'all');
x_mean = 0;
% x_std = std(x, 1, 'all');
x_std=100.99;
norm_x = (x-x_mean)./x_std;
% y_mean = mean(y, 'all');
y_mean=0;
% y_std = std(y, 1, 'all');
y_std=100.99;
norm_y = (y-y_mean)./y_std;
% z_mean = mean(z, 'all');
z_mean=0;
% z_std = std(z, 1, 'all');
z_std=176.06;
norm_z = (z-z_mean)./z_std;
% l_mean = mean(l, 'all');
l_mean=180;
% l_std = std(l, 1, 'all');
l_std=70.71;
norm_l = (l-l_mean)./l_std;
% t_mean = mean(t, 'all');
t_mean=0;
% t_std = std(t, 1, 'all');
t_std=53.38;
norm_t = (t-t_mean)./t_std;
norm_w=norm_l./r;

% Initialize the rebuilt matx with the same size as the original matrixri
norm_normal00 = zeros(size(normal00));

% Place the normalized columns back into the correct positions
norm_normal00(:, 1:6:end) = norm_x;
norm_normal00(:, 2:6:end) = norm_y;
norm_normal00(:, 3:6:end) = norm_z;
norm_normal00(:, 4:6:end) = norm_l;
norm_normal00(:, 5:6:end) = norm_w;
norm_normal00(:, 6:6:end) = norm_t;

save(['..\data\comb_', num2str(rods), '_', num2str(num_total), '_train.mat'], ...
    'normal00','TL_int','TR_int', 'TL_TR_int','TL_float','TR_float', 'TL_TR_float' ...
    , 'x', "y", "z", "l", "w", "t", "r", "x_mean", "x_std", "norm_x", ...
    "y_mean", "y_std", "norm_y", "z_mean", "z_std", "norm_z", ...
    "l_mean", "l_std", "norm_l", "t_mean", "t_std", "norm_t", "norm_w", ...
    "norm_normal00", "tseed");

% Process and save test dataset
normal00=temp_normal00_test;
TL_float=temp_TL_test;
TR_float=temp_TR_test;
TL_TR_float=horzcat(TL_float, TR_float);
TL_int=int64(TL_float*1000);
TR_int=int64(TR_float*1000);
TL_TR_int=horzcat(TL_int, TR_int);
x=double(normal00(:, 1:6:end));
y=double(normal00(:, 2:6:end));
z=double(normal00(:, 3:6:end));
l=double(normal00(:, 4:6:end));
w=double(normal00(:, 5:6:end));
t=double(normal00(:, 6:6:end));
r=l./w;
norm_x = (x-x_mean)./x_std;
norm_y = (y-y_mean)./y_std;
norm_z = (z-z_mean)./z_std;
norm_l = (l-l_mean)./l_std;
norm_t = (t-t_mean)./t_std;
norm_w=norm_l./r;

norm_normal00 = zeros(size(normal00));

norm_normal00(:, 1:6:end) = norm_x;
norm_normal00(:, 2:6:end) = norm_y;
norm_normal00(:, 3:6:end) = norm_z;
norm_normal00(:, 4:6:end) = norm_l;
norm_normal00(:, 5:6:end) = norm_w;
norm_normal00(:, 6:6:end) = norm_t;

save(['..\data\comb_', num2str(rods), '_', num2str(num_total), '_test.mat'], ...
    'normal00','TL_int','TR_int', 'TL_TR_int','TL_float','TR_float', 'TL_TR_float' ...
    , 'x', "y", "z", "l", "w", "t", "r", "x_mean", "x_std", "norm_x", ...
    "y_mean", "y_std", "norm_y", "z_mean", "z_std", "norm_z", ...
    "l_mean", "l_std", "norm_l", "t_mean", "t_std", "norm_t", "norm_w", ...
    "norm_normal00", "tseed");

