
K = 6;
data = [PAP_pro;PRP_pro;RAP_pro;RRP_pro;];
[c,mu] = kmeans(error,K);
%% ÆÀ¼Û
Cmp = 0;
for i = 1:K
    dd = data(:,c==i);
%     Cmp = Cmp+sum(sum((dd-mu(:,i)*ones(1,size(dd,2))).^2));
end
Prox = 0;
for i=1:K
    for j=1:K
        Prox = Prox+exp(-1e-5*sum((mu(:,j)-mu(:,i)).^2));
    end
end
disp(['************** K=',num2str(K),'  **********'])
disp(['Cmp:',num2str(Cmp/1e10)]);
disp(['Prox:',num2str(Prox)]);
        