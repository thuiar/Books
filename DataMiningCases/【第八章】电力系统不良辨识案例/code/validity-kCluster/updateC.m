function Cn = updateC(X,classn,Tmax,kC,D)

Cn = zeros(kC,D);
numC = zeros(kC,1);

for i=1:Tmax
    classt = classn(i);
    Cn(classt,:) = Cn(classt,:)+X(i,:);
    numC(classt) = numC(classt)+1;
end

for i=1:kC
    Cn(i,:) = Cn(i,:)/max(numC(i),1);
end

end