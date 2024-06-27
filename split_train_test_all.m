clear all
load("C:\Users\HSZhang.NANO\Desktop\script\data\comb_2_6086.mat")
notmal00=int64(normal00);
TL=int64(TL*1000);
TR=int64(TR*1000);
temp_normal_step=normal00(end-85:end,:);
temp_TL_step=TL(end-85:end,:);
temp_TR_step=TR(end-85:end,:);
normal00=normal00(1:end-86,:);
TL=TL(1:end-86,:);
TR=TR(1:end-86,:);
ctime = datestr(now, 30);
tseed = str2double(ctime((end - 5) : end));
rand('seed', tseed);
num_test=100;
idx = randperm(length(TL),num_test);
temp_normal00_test=normal00(idx,:);
normal00(idx,:)=[];
temp_TL_test=TL(idx,:);
TL(idx,:)=[];
temp_TR_test=TR(idx,:);
TR(idx,:)=[];
normal00=[normal00;temp_normal_step];
TL=[TL;temp_TL_step];
TR=[TR;temp_TR_step];
TL_TR=horzcat(TL, TR);
save('C:\Users\HSZhang.NANO\Desktop\lstm_final\data\comb_2_6086_train.mat','normal00','TL','TR', 'TL_TR');
normal00=temp_normal00_test;
TL=temp_TL_test;
TR=temp_TR_test;
TL_TR=horzcat(TL, TR);
save('C:\Users\HSZhang.NANO\Desktop\lstm_final\data\comb_2_6086_test.mat','normal00','TL','TR',"TL_TR");