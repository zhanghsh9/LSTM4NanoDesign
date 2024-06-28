clear all;
%文件父路径
num=1000;
dirc='\\groupnas\AI_Data4\AI_Chiral\Generated_Test_2_6\generate00\';
filename={'\xPolL.mat';'\xPolU.mat';'\yPolL.mat';'\yPolU.mat'};
%normaldir='C:\Users\HSZhang\Desktop\script\Lumerical\normal.mat';
%load(normaldir);

ToLCP=[1;0;1j;0]/sqrt(2);
ToRCP=[1;0;-1j;0]/sqrt(2);
load([dirc '1' filename{1}]);
TL=zeros(num,length(lamda));
TR=zeros(num,length(lamda));

for jj=1:num
    load([dirc num2str(jj) filename{1}]);
    mtxtemp = zeros(4,4,length(lamda));
    for ii = 1:4
        load([dirc num2str(jj) filename{ii}]);
        mtxtemp(1,ii,:) = EthetaL;
        mtxtemp(2,ii,:) = EthetaU;
        mtxtemp(3,ii,:) = EphiL;
        mtxtemp(4,ii,:) = EphiU;
    end
    mtx = 1i * conj(mtxtemp);
    f_p=length(lamda);
    Temp_L=zeros(4,1,f_p);
    Temp_R=zeros(4,1,f_p);
    T_L=zeros(f_p,1);
    T_R=zeros(f_p,1);
    for kk=1:f_p
        Temp_L(:,:,kk)=mtx(:,:,kk)*ToLCP;
        Temp_R(:,:,kk)=mtx(:,:,kk)*ToRCP;
    end
    for kk=1:f_p
        T_R(kk)=(abs(Temp_L(2,1,kk)).^2+abs(Temp_L(4,1,kk)).^2);
        T_L(kk)=(abs(Temp_R(2,1,kk)).^2+abs(Temp_R(4,1,kk)).^2);
    end
    save([dirc num2str(jj) '\result.mat'],'T_L','T_R');
    TL(jj,:)=T_L(:);
    TR(jj,:)=T_R(:);
    %figure(jj);
    %plot(lamda,T_L);
end
save([dirc '\result.mat'],'TL','TR');
