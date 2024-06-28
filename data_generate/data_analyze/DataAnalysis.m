% This program is to analyze data from lumerical FDTD
% To create scattering matrix for 3D transmission mode.
% The coordinate system is reflection configuration.
% Important! Only works for homogenious media!!!

% clear;
indicator = 0;
% 0: aribitrary case
% 1: structure is RSFS
% 2: structure is RS, so only xPolL & xPolU are needed
% 9: structure is user defined

bLocal = 'F:\Generated_Test\generate00\1\';
bName = {'xPolL';'xPolU';'yPolL';'yPolU'};
% order: x polarization from lower; x polarization from upper; y
% polarization from lower; y polarization from upper

load([bLocal bName{1}]);
MTXdat.lambda = lamda;
mtxtemp = zeros(4,4,length(MTXdat.lambda));

switch indicator
    case 0
        for ii = 1:4
            load([bLocal bName{ii}]);
            mtxtemp(1,ii,:) = EthetaL;
            mtxtemp(2,ii,:) = EthetaU;
            mtxtemp(3,ii,:) = EphiL;
            mtxtemp(4,ii,:) = EphiU;
        end
    case 1
        mtxtemp(1,1,:) = EthetaL;
        mtxtemp(2,2,:) = EthetaL;
        mtxtemp(3,3,:) = EthetaL;
        mtxtemp(4,4,:) = EthetaL;
        mtxtemp(2,1,:) = EthetaU;
        mtxtemp(1,2,:) = EthetaU;
        mtxtemp(4,3,:) = EthetaU;
        mtxtemp(3,4,:) = EthetaU;
        mtxtemp(4,1,:) = EphiU;
        mtxtemp(1,4,:) = EphiU;
        mtxtemp(3,2,:) = -EphiU;
        mtxtemp(2,3,:) = -EphiU;
    case 2
        load([bLocal bName{1}]);
        mtxtemp(1,1,:) = EthetaL;
        mtxtemp(3,3,:) = EthetaL;
        mtxtemp(2,1,:) = EthetaU;
        mtxtemp(1,2,:) = EthetaU;
        mtxtemp(4,3,:) = EthetaU;
        mtxtemp(3,4,:) = EthetaU;
        mtxtemp(4,1,:) = EphiU;
        mtxtemp(1,4,:) = EphiU;
        mtxtemp(3,2,:) = -EphiU;
        mtxtemp(2,3,:) = -EphiU;
        load([bLocal bName{2}]);
        mtxtemp(2,2,:) = EthetaU;
        mtxtemp(4,4,:) = EthetaU;
    otherwise
        error("wrong indicator!!!")
end

MTXdat.mtx = 1i * conj(mtxtemp);
save([bLocal 'ScatteringMatrix'],'MTXdat');

disp('finished!');