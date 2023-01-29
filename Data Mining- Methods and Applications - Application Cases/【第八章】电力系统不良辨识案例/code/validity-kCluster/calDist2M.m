function dist2M = calDist2M(X,C,Tmax,kC)

dist2M = zeros(Tmax,kC);

for i=1:kC
    dist2M(:,i) = sum((X-ones(Tmax,1)*C(i,:)).^2,2);
end

end