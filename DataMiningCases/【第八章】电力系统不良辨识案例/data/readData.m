clear;
clc;

fid = fopen('dabieshan.csv');
raw_linet = fgetl(fid);
raw_linet = fgetl(fid);

Nmax = 10;

DEV_ID = [126000529
                    124000137
                    126000529
                    126000532
                    124000139
                    126000532
                    124000137
                    124000139
                    128000028
                    128000029];
METER_SERIAL = [1
                                    1
                                    0
                                    1
                                    1
                                    0
                                    0
                                    0
                                    0
                                    0];

Tmax = 109536;
% Tmax = 10000;
DataM = zeros(Nmax,Tmax,2,2,2,2,2);%维度含义：~,~,量值/状态,增量/底码,一/二次侧,有功/无功,正向/反向
METER_RAW_STATUS = zeros(Nmax,Tmax,2);%记录各表各时刻的METER_RAW_STATUS_VALUE和METER_RAW_STATUS_STATUS

Tstart = 1288541100;
dT = 300;
% counti = 1;
% Ttag = 1;

while raw_linet~=-1
    raw_linet = raw_linet(2:end-1);
    rawt = regexp(raw_linet,'","', 'split');
    rawt = str2double(rawt);
    Ttag = (rawt(1)-Tstart)/300+1;
    
    
    MeterError = 1;
    for i=1:Nmax
        if (rawt(2)==DEV_ID(i))&&(rawt(3)==METER_SERIAL(i))
            MeterError = 0;
            break;
        end
    end
    
    if MeterError
        error('Undefined Meter:\tDEV_ID = %d\tMETER_SERIAL = %d',rawt(2),rawt(3));
    end
    
    DataM(i,Ttag,:,:,:,:,:) = reshape(rawt(4:end-2),2,2,2,2,2);
    METER_RAW_STATUS(i,Ttag,:) = rawt(end-1:end);
    raw_linet = fgetl(fid);
    
    if mod(Ttag,10000)==0
        fprintf('MeterNo:%d\tTtag = %d(Total:%d)\n',i,Ttag,Tmax);
        save ('MeterData_lxz.mat');
    end
%     counti = mod(counti,Nmax)+1;
%     if counti==1
%         if mod(Ttag,10000)==0
%             fprintf('Ttag = %d(Total:%d)\n',Ttag,Tmax);
%             save ('MeterData_lxz.mat');
%         end
%         Ttag = Ttag+1;
%     end
end


save ('MeterData_lxz.mat');