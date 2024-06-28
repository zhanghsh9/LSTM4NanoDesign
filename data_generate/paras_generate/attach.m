load('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220225.mat');
temp00=[30 116 131];

%此脚本用于生成一组决定n根棒结构的随机数，同时对(1)有棒的端点落在fdtd范围外，(2)
%有棒相互重叠这两种情况做了分类。最终生成的随机数存在以normal开头的数组中，格式
%为[第一根棒的(x,y,z,l,w,theta) 第二根棒的(x,y,z,l,w,theta) ... 
%第n根棒的(x,y,z,l,w,theta) Px Py n max(rangeZ)]，长度的单位均为nm，角度的单位均为度

%此脚本默认fdtd的中心点为(0,0,0)

%每根棒有6个自由度按顺序分别为(x,y,z,l,w,theta)，n根棒就有6n个自由度，考虑到整个
%结构的边长Px，材料的介电常数εm，背景介电常数εb，则给定6n+3个参数就可以唯
%一地确定一个含有n根棒的结构，但εm与εb变化范围较小，前期Px固定为500nm，在此脚本
%中不生成。

%clear all;

%样本数量num
num=500;

%确定n
n = 2;

%确定Px的范围
rangepx = 500:5:500;%500nm-500nm,每隔5个点取一个点，可调
rangepy = 500:5:500;%500nm-500nm,每隔5个点取一个点，可调


%确定(z,l,w,theta)各参数的范围
rangeZ = -300:1:300;%-300nm-300nm，每1个点取一个
rangel = 1:1:450;%0-450nm，每1个点取一个
rangew = 1:1:100;%0-100nm，每1个点取一个
rangetheta = -90:1:90;%-90°-90°，每1个点取一个

%声明随机数矩阵，矩阵每一行为((x,y,z,l,w,theta)*n,Px,Py,n)
%normal00：既不存在(1)也不存在(2)
%normal10：存在(1)但不存在(2)
%normal01：存在(2)但不存在(1)
%normal11：(1)和(2)均存在
normal00add = zeros(num,6*n+4);
normal10add = zeros(num,6*n+4);
normal01add = zeros(num,6*n+4);%去掉注释以判断情况(2)
normal11add = zeros(num,6*n+4);%去掉注释以判断情况(2)

%开始生成随机数矩阵
x = 0;
y = 0;
z = 0;
l = 0;
w = 0;
theta = 0;
row00 = 1;
row01 = 1;
row10 = 1;
row11 = 1;
%count00=1;
%count10=1;
%count01=1;
%count11=1;
temp = zeros(1,6*n+4);
count = 0;
ctime = datestr(now, 30);
tseed = str2double(ctime((end - 5) : end));
rand('seed', tseed);

while (row00<=num)||(row10<=num)||(row01<=num)||(row11<=num)
    %生成Px，Py
    px = rangepx(randi(length(rangepx)));
    py = rangepy(randi(length(rangepy)));
    
    %确定(x,y)的范围
    rangeX = -(px/2):1:(px/2);%在Px范围内取点，每5个取一个
    rangeY = -(py/2):1:(py/2);%在Py范围内取点，每5个取一个
    
    %生成(x,y,z,l,w,theta)
    x = rangeX(randi(length(rangeX)));
    y = rangeY(randi(length(rangeY)));
    z = rangeZ(randi(length(rangeZ)));
    l = rangel(randi(length(rangel)));
    w = rangew(randi(length(rangew)));
    if w>l
        continue
    end
    theta = rangetheta(randi(length(rangetheta)));
    count = count+1;
    
    temp((6*count-5):(6*count)) = [x y z l w theta];
    if count==n
        temp((end-3):(end)) = [px py n max(rangeZ)];
        count=0;
        if if1(px,py,rangeZ,n,temp)
            if if2(n,temp)
                if row11<=num
                    if ~ismember(temp,normal11,'rows')
                        normal11add(row11,:)=temp;
                        row11=row11+1;
                    end
                end
            else
                if ~ismember(temp,normal10,'rows')
                    if row10<=num
                        normal10add(row10,:)=temp;
                        row10=row10+1;
                    end
                end
            end
        else
            if if2(n,temp)
                if row01<=num
                    if ~ismember(temp,normal01,'rows')
                        normal01add(row01,:)=temp;
                        row01=row01+1;
                    end
                end
            else
                if row00<=num
                    if ~ismember(temp,normal00,'rows')
                        normal00add(row00,:)=temp;
                        row00=row00+1;
                    end
                end
            end
        end
    end
end


cont=1;
for ii=temp00
    normal00(ii,:)=normal00add(cont,:);
    cont=cont+1;
end
% cont=1;
% for ii=temp10
%     normal10(ii,:)=normal10add(cont,:);
%     cont=cont+1;
% end
% cont=1;
% for ii=temp01
%     normal01(ii,:)=normal01add(cont,:);
%     cont=cont+1;
% end
% cont=1;
% for ii=temp11
%     normal11(ii,:)=normal11add(cont,:);
%     cont=cont+1;
% end



save('C:\Users\zhanghsh\Desktop\script\Lumerical\normal20220225add2.mat','normal00','num','temp00')