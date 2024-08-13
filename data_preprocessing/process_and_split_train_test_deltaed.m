clear all
ctime = datestr(now, 30);
tseed = str2double(ctime((end - 5) : end));
rand('seed', tseed);

load("..\data\comb_2_7186_all.mat");
num_test=200;

num_total=length(normal00);
rods=int64(length(normal00(1,:))/6);
normal00=double(normal00);
temp=normal00;

% Calculate relative position 
x=double(normal00(:, 1:6:end));
normal00(:, 1:6:end)=[];
y=double(normal00(:, 1:5:end));
normal00(:, 1:5:end)=[];
z=double(normal00(:, 1:4:end));
normal00(:, 1:4:end)=[];

delta_x=x(:,2)-x(:,1);
delta_y=y(:,2)-y(:,1);
delta_z=z(:,2)-z(:,1);
for ii=1:length(delta_x)
    if delta_x(ii)>170
        delta_x(ii)=delta_x(ii)-170;
    else
        if delta_x(ii)<-170
            delta_x(ii)=delta_x(ii)+170;
        end
    end
    if abs(delta_x(ii))<1e-3
        delta_x(ii)=0;
    end
end

for ii=1:length(delta_y)
    if delta_y(ii)>170
        delta_y(ii)=delta_y(ii)-170;
    else
        if delta_y(ii)<-170
            delta_y(ii)=delta_y(ii)+170;
        end
    end
    if abs(delta_y(ii))<1e-3
        delta_y(ii)=0;
    end
end

for ii=1:length(delta_z)
    if delta_z(ii)>300
        delta_z(ii)=delta_z(ii)-300;
    else
        if delta_z(ii)<-300
            delta_z(ii)=delta_z(ii)+300;
        end
    end
    if abs(delta_z(ii))<1e-3
        delta_z(ii)=0;
    end
end
delta_x_mean = mean(delta_x, 'all');
delta_x_std = std(delta_x, 1, 'all');
delta_y_mean = mean(delta_y, 'all');
delta_y_std = std(delta_y, 1, 'all');
delta_z_mean = mean(delta_z, 'all');
delta_z_std = std(delta_z, 1, 'all');
normal00=horzcat(normal00(:,1:3), delta_x, delta_y, delta_z, normal00(:,4:end));

% Process and save train dataset
delta_x=double(normal00(:, 4));
delta_y=double(normal00(:, 5));
delta_z=double(normal00(:, 6));
l=double(normal00(:, 1:6:end));
w=double(normal00(:, 2:6:end));
t=double(normal00(:, 3:6:end));
r=l./w;

norm_delta_x = (delta_x-delta_x_mean)./delta_x_std;
norm_delta_y = (delta_y-delta_y_mean)./delta_y_std;
norm_delta_z = (delta_z-delta_z_mean)./delta_z_std;
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
norm_normal00(:, 1:6:end) = norm_l;
norm_normal00(:, 2:6:end) = norm_w;
norm_normal00(:, 3:6:end) = norm_t;
norm_normal00(:, 4) = norm_delta_x;
norm_normal00(:, 5) = norm_delta_y;
norm_normal00(:, 6) = norm_delta_z;

% Seperate step data out
normal00=int64(normal00);
temp_normal_step=normal00(end-185:end,:);
normal00=normal00(1:end-186,:);
temp_norm_normal_step=norm_normal00(end-185:end,:);
norm_normal00=norm_normal00(1:end-186,:);
temp_TL_step=TL(end-185:end,:);
temp_TR_step=TR(end-185:end,:);
TL=TL(1:end-186,:);
TR=TR(1:end-186,:);

% Shuffle
idx = randperm(length(TL),num_test);
temp_normal00_test=normal00(idx,:);
temp_norm_normal00_test=norm_normal00(idx,:);
normal00(idx,:)=[];
normal00=[normal00;temp_normal_step];
norm_normal00(idx,:)=[];
norm_normal00=[norm_normal00;temp_norm_normal_step];

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

save(['..\data\deltaed_comb_', num2str(rods), '_', num2str(num_total), '_train.mat'], ...
    'normal00','TL_int','TR_int', 'TL_TR_int','TL_float','TR_float', 'TL_TR_float' ...
    , 'x', "y", "z", "l", "w", "t", "r", "delta_x_mean", "delta_x_std", "norm_delta_x", ...
    "delta_y_mean", "delta_y_std", "norm_delta_y", "delta_z_mean", "delta_z_std", "norm_delta_z", ...
    "l_mean", "l_std", "norm_l", "t_mean", "t_std", "norm_t", "norm_w", ...
    "norm_normal00", "tseed", 'delta_x', "delta_y", "delta_z");

% Process and save test dataset
normal00=temp_normal00_test;
norm_normal00=temp_norm_normal00_test;
TL_float=temp_TL_test;
TR_float=temp_TR_test;
TL_TR_float=horzcat(TL_float, TR_float);
TL_int=int64(TL_float*1000);
TR_int=int64(TR_float*1000);
TL_TR_int=horzcat(TL_int, TR_int);


save(['..\data\deltaed_comb_', num2str(rods), '_', num2str(num_total), '_test.mat'], ...
    'normal00','TL_int','TR_int', 'TL_TR_int','TL_float','TR_float', 'TL_TR_float' ...
    , 'x', "y", "z", "l", "w", "t", "r", "delta_x_mean", "delta_x_std", "norm_delta_x", ...
    "delta_y_mean", "delta_y_std", "norm_delta_y", "delta_z_mean", "delta_z_std", "norm_delta_z", ...
    "l_mean", "l_std", "norm_l", "t_mean", "t_std", "norm_t", "norm_w", ...
    "norm_normal00", "tseed", 'delta_x', "delta_y", "delta_z");

