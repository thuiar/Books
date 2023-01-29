clc;clear;

load('..\MeterData_lxz.mat);
kC = 8;
kP = 4;
D = kP*Nmax;
X = reshape(DataM(:,:,1,1,1,:,:),Nmax,Tmax*kP)';
X = [X(1:Tmax,:) X(Tmax+1:2*Tmax,:) X(2*Tmax+1:3*Tmax,:) X(3*Tmax+1:4*Tmax,:)];

% wMo = zeros(Tmax,kC);
classn = zeros(Tmax,1);
% classo = ones(Tmax,1);
threClass = 0.0001;
threC = 0.1;

minP = [min(PAPVALUE,[],2);min(RAPVALUE,[],2);min(PRPVALUE,[],2);min(RRPVALUE,[],2)]';
maxP = [max(PAPVALUE,[],2);max(RAPVALUE,[],2);max(PRPVALUE,[],2);max(RRPVALUE,[],2)]';

% minP = [min(PAPVALUE,[],2);max(RAPVALUE,[],2);min(PRPVALUE,[],2);max(RRPVALUE,[],2)]';
% maxP = [max(PAPVALUE,[],2);min(RAPVALUE,[],2);max(PRPVALUE,[],2);min(RRPVALUE,[],2)]';

% minP = (rand([1 kP*Nmax])*10+1)*50;
% maxP = (rand([1 kP*Nmax])*5+6)*50;

Cn = ones(kC,1)*minP+([0:kC-1]')*(maxP-minP)/(kC-1);

count = 0;
maxCount = 1000;
maxClassChange = 1;
maxcChange = threC+1;
nClassChanged = 1;
while (nClassChanged~=0||maxcChange>threC)&&(count<=maxCount)
    classo = classn;
    classn = updateClass(X,Cn,Tmax,kC);
    Co = Cn;
    Cn = updateC(X,classn,Tmax,kC,D);
    count = count+1;
    nClassChanged = length(find((classn-classo)~=0));
    maxcChange = max(max(abs(Cn-Co)));
    fprintf('count = %3d\tnClassChanged = %10d\tmaxcChange = %f\n',count,nClassChanged,maxcChange);
%     fprintf('%d,%d,%d,%d\n',maxClassChange>threClass,maxcChange>threC,(maxClassChange>threClass)||(maxcChange>threC),(maxClassChange>threClass||maxcChange>threC)&&(count<=maxCount));
end
class = classn;
SSE = sum(min(calDist2M(X,Cn,Tmax,kC),[],2));
fprintf('SSE = %d\n',SSE);
% end
save(['MeterData_lxz_K(kC=' num2str(kC) ').mat']);