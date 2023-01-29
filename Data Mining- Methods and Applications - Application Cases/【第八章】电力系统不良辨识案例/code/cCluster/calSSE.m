function SSE = calSSE(X,C,wM,p)

[Xn D] = size(X);
[Cn D2] = size(C);
[wrow wcol] = size(wM);
if D~=D2||Xn~=wrow||Cn~=wcol
    error('Dimension Error');
end

SSE = 0;
for i=1:Xn
    for j=1:Cn
        SSE = SSE+calDist2(X(i,:),C(j,:))*(wM(i,j)^p);
    end
end

end