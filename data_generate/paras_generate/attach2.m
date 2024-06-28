load("C:\Users\zhanghsh\Desktop\script\Lumerical\normal20230422.mat")
normal=normal00;
load('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220329.mat')
for ii=1:length(normal00)
    if ismember(normal00(ii,:),normal,'rows')
        disp("Failed");
        break
    end
end