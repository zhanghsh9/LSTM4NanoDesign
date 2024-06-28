load('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220225.mat');
temp00=[];
temp01=[];
temp10=[];
temp11=[];
for ii=1:500
    if if2(n,normal00(ii,:))
        temp00=[temp00 ii];
    end
    if ~if2(n,normal01(ii,:))
        temp01=[temp01 ii];
    end
    if if2(n,normal10(ii,:))
        temp10=[temp10 ii];
    end
    if ~if2(n,normal11(ii,:))
        temp11=[temp11 ii];
    end
end
%disp(temp00);