function classn = ditributeClass(X,C)

[Xn,~] = size(X);
[kC,~] = size(C);
classn = zeros(Xn,1);

for i=1:Xn
    [~,classn(i)] = min(calDist2(ones(kC,1)*X(i,:),C));
end

end