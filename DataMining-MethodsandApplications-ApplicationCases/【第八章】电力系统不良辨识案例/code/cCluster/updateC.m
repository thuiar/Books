function C = updateC(X,wM,p)

[Xnum D] = size(X);
wMp = wM.^p;
C = ((X'*wMp)./(ones(D,1)*sum(wMp)))';

end