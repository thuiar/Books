function classn = updateClass(X,C,Tmax,kC)

dist2M = calDist2M(X,C,Tmax,kC);

[~,classn] = min(dist2M,[],2);

end