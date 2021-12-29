format long; clear; clc;

% ===== READ DATA =====
EXCEL = xlsread('平台关注量增长预测-基础数据',1);

% ===== TRAINING =====
OriginalInput = EXCEL(1:30,4:7);
OriginalOutput = EXCEL(1:30,1);
OriginalInput = OriginalInput';
OriginalOutput = OriginalOutput';
[ Input,minI,maxI ]  = premnmx(OriginalInput);
[ Output,minO,maxO ] = premnmx(OriginalOutput);
Net = newff(minmax(Input),[2 1],{'logsig','purelin'},'traingdx'); 
Net.trainparam.show = 80000;
Net.trainparam.epochs = 80000;
Net.trainparam.goal = 0.001;
Net.trainParam.lr = 0.015;
Net.trainParam.min_grad = 1e-8;
Net = train(Net,Input,Output);

% ===== TESTING =====
TestInput = EXCEL(31:49,4:7);
TestOutput = EXCEL(31:49,1);
TestInput = TestInput';
TestOutput = TestOutput;
FinalInput = tramnmx(TestInput,minI,maxI);
Y = sim(Net,FinalInput);
Y = Y';
N = length(Y);
for i = 1:N
    YR(i) = minO + 0.5*(Y(i)+1)*(maxO - minO);
end
YR = YR';


