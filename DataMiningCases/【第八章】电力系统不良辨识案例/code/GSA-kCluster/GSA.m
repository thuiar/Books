NN = 10;
Gap = zeros(1,NN);
Sd = zeros(1,NN);
S = zeros(1,NN);
logW = zeros(1,NN);
ElogW = zeros(1,NN);
data = error;

%% 聚类分析，聚类个数为K
for K=1:NN
    if K==1
        [~,mu] = kmeans(data,K);
        c= ones(1,size(data,2));
        Wk = getWk( c,K,data,mu );
        % 把数据分成100组，分别计算Wk
        Wkn = zeros(1,NumRef);
        for m = 1:NumRef
            [~,muRef] = kmeans(reshape(dataRef(m,:,:),size(dataRef,2),size(dataRef,3)),K);
            cRef= ones(1,size(dataRef,3));
            Wkn(m) = getWk(cRef,K,reshape(dataRef(m,:,:),size(dataRef,2),size(dataRef,3)),muRef );
        end
        Sd(K) = sqrt(sum((log(Wkn)-mean(log(Wkn))).^2)/NumRef);
        S(K) = Sd(K)*sqrt(1+1/NumRef);
        logW(K) = log(Wk);
        ElogW(K) = mean(log(Wkn));
        Gap(K) = mean(log(Wkn))-log(Wk);
    else
        [c,mu] = kmeans(data,K);
        % 计算Wk
        Wk = getWk( c,K,data,mu );
        % 把数据分成100组，分别计算Wk
        Wkn = zeros(1,NumRef);
        for m = 1:NumRef
            [cRef,muRef] = kmeans(reshape(dataRef(m,:,:),size(dataRef,2),size(dataRef,3)),K);
            Wkn(m) = getWk(cRef,K,reshape(dataRef(m,:,:),size(dataRef,2),size(dataRef,3)),muRef );
        end
        Sd(K) = sqrt(sum((log(Wkn)-mean(log(Wkn))).^2)/NumRef);
        S(K) = Sd(K)*sqrt(1+1/NumRef);
        logW(K) = log(Wk);
        ElogW(K) = mean(log(Wkn));
        Gap(K) = mean(log(Wkn))-log(Wk);
    end
end
plot(Gap)   
% Gap(1:end-1)-Gap(2:end)+S(2:end)
