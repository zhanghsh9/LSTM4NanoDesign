clear all;
load('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220225.mat');
load('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220225add.mat');
temp1=[];
temp2=[];
temp3=[];
temp4=[];
for ii=1:num
    if sum(normal00(ii,:)~=normal00temp(ii,:))
        temp1=[temp1 ii];
    end
    if sum(normal01(ii,:)~=normal01temp(ii,:))
        temp2=[temp2 ii];
    end
    if sum(normal10(ii,:)~=normal10temp(ii,:))
        temp3=[temp3 ii];
    end
    if sum(normal11(ii,:)~=normal11temp(ii,:))
        temp4=[temp4 ii];
    end
end