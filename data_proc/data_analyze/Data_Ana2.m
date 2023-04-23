cd('D:\Generated_Test_2\generate00\1\');
ToLCP=[1;0;1j;0]/sqrt(2);
ToRCP=[1;0;-1j;0]/sqrt(2);
load('ScatteringMatrix.mat');
f_p=length(MTXdat.lambda);
Temp_L=zeros(4,1,f_p);
Temp_R=zeros(4,1,f_p);
T_L=zeros(f_p,1);
T_R=zeros(f_p,1);
for ii=1:f_p
    Temp_L(:,:,ii)=MTXdat.mtx(:,:,ii)*ToLCP;
    Temp_R(:,:,ii)=MTXdat.mtx(:,:,ii)*ToRCP;
end
for ii=1:f_p
    T_L(ii)=(abs(Temp_L(2,1,ii).^2)+abs(Temp_L(4,1,ii).^2));
    T_R(ii)=(abs(Temp_R(2,1,ii).^2)+abs(Temp_R(4,1,ii).^2));
end
plot(MTXdat.lambda,T_L);