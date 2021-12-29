clc;clear;

load('..\MeterData_lxz.mat);

% kC = max([NPAP;NPRP;NRAP;NRRP]);
kC = 10;
X = reshape(DataM(:,:,1,1,1,:,:),Nmax,Tmax*4)';
X = [X(1:Tmax,:) X(Tmax+1:2*Tmax,:) X(2*Tmax+1:3*Tmax,:) X(3*Tmax+1:4*Tmax,:)];

% wMo = zeros(Tmax,kC);
wMn = ones(Tmax,kC);
threW = 0.001;
threC = 0.1;

minP = (rand([1 4*Nmax])*10+1)*50;
maxP = (rand([1 4*Nmax])*5+6)*50;

Cn = ones(kC,1)*minP+([0:kC-1]')*(maxP-minP)/(kC-1);

count = 0;
maxCount = 1000;
maxwChange = threW+1;
maxcChagne = threC+1;
while (maxwChange>threW||maxcChange>threC)&&(count<=maxCount)
    wMo = wMn;
    wMn = updateW(X,Cn,p);
    Co = Cn;
    Cn = updateC(X,wMn,p);
    count = count+1;
    maxwChange = max(max(abs(wMn-wMo)));
    maxcChange = max(max(abs(Cn-Co)));
    fprintf('count = %d\tmaxwChange = %f\tmaxcChange = %f\n',count,maxwChange,maxcChange);
%     fprintf('%d,%d,%d,%d\n',maxwChange>threW,maxcChange>threC,(maxwChange>threW)||(maxcChange>threC),(maxwChange>threW||maxcChange>threC)&&(count<=maxCount));
end
wM = wMn;
SSE = calSSE(X,Cn,wM,2);
fprintf('SSE = %d\n',SSE);
classn = ditributeClass(X,Cn);

% end
save(['MeterData_lxz_W_C(kC=' num2str(kC) ').mat']);