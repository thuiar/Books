
%% ��ջ�������
% clc
% clear
clear 
load data_PRO
%% �Ⱦ��ࣨ20��
dataPRO = [PAP_pro;PRP_pro;RAP_pro;RRP_pro;];
[c,mu] = kmeans(dataPRO,20);
dataBP = zeros(20,0);
for i=1:20
    num1 = sum(c==i);
    num2 = 20*floor(num1/500);
    rr = rand(1,num1);
    [~,rr] = sort(rr);
    index = find(c==i);
    dataBP = [dataBP,dataPRO(:,index(rr(1:num2)))];
end
%% ѵ������Ԥ��������ȡ����һ��
%�ҳ�ѵ�����ݺ�Ԥ������
% input_train=dataBP([1,3:11,13:21,23:31,33:40],:);
% output_train=dataBP([2,12,22,32],:);
% % output_train=dataBP([1:5,11:15,21:25,31:35],:);
% input_test=dataPRO([1,3:11,13:21,23:31,33:40],:);
% output_test=dataPRO([2,12,22,32],:);

input_train=dataBP;
% output_train=dataBP([1,9,10,11,12,13,14,19,20,23,24,27,28,31,35,39],:);
% output_train=dataBP([1:5,11:15,21:25,31:35],:);
output_train=dataBP([9,10,19,35,39],:);
input_test=dataPRO;
output_test=dataPRO([9,10,19,35,39],:);

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
% %��ʼ������ṹ
net=newff(inputn,outputn,5);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;

%����ѵ��
net=train(net,inputn,outputn);

%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����
an=sim(net,inputn_test);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

%Ԥ�����
error=(BPoutput-output_test).^2;
clearvars -except  dataBP error PAP_pro PRP_pro RAP_pro RRP_pro