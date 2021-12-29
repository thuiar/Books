% data = [PAP_pro;PRP_pro;RAP_pro;RRP_pro;];
% data = data(:,find(error(1,:)<1000 & error(2,:)<1000 & error(3,:)<1000 & error(4,:)<1000));
data = error;
data = data(:,find(sum(error,1)<1000));

%% 先聚类（20）
[c,mu] = kmeans(data,20);
%% 生成参考数据集
NumRef = 5;
dataRef = zeros(NumRef,size(data,1),size(data,2));
for i=1:NumRef
    dataRR = zeros(size(data,1),0);
    for k=1:20
        index = find(c==k);
        index_length = sum(c==k);
        if index_length>200
            dataRR = [dataRR,...
             min(data(:,index),[],2)*ones(1,index_length)+(max(data(:,index),[],2)-min(data(:,index),[],2))*ones(1,index_length).*rand(size(data,1),index_length)];
        else
            dataRR = [dataRR,zeros(size(data,1),index_length)];
        end
    end
    dataRef(i,:,:) = dataRR;
end
clearvars -except  error dataRef data NumRef PAP_pro PRP_pro RAP_pro RRP_pro