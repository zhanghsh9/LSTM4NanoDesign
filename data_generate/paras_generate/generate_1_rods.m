clear all
num=413;
ctime = datestr(now, 30);
tseed = str2double(ctime((end - 5) : end));
rand('seed', tseed);
normal00=zeros(num,6);
rangel = 60:5:300;
rangelwr = 2:0.2:10;%10-150nm
% for ii=1:n
%     normal00(ii,1)=int64(randi(400)-200);
%     normal00(ii,2)=int64(randi(400)-200);
%     normal00(ii,3)=int64(randi(600)-300);
%     %normal00(ii,4)=int64(rangel(randi(length(rangel))));
%     %normal00(ii,5)=int64(round(l/rangelwr(randi(length(rangelwr)))));
%     %normal00(ii,6)=int64(randi(180)-90);
% end

p=1;
for l=60:40:300
    for rangelwr = 2:1:10
        if (l==60 || l==100) && (rangelwr==9||rangelwr==10)
            continue
        end
        for theta=[-90,-60,-30,0,30,60,90]
            normal00(p,4)=l;
            normal00(p,5)=int64(l/rangelwr);
            normal00(p,6)=theta;
            p=p+1;
        end
        
    end
end

temp=ones(num,4);
temp(:,1:2)=400;
temp(:,3)=1;
temp(:,4)=300;
normal00=[normal00,temp];

save(['C:\Users\zhanghsh\Desktop\script\Lumerical\normal' ctime(1:8) '.mat'],'normal00','num')







