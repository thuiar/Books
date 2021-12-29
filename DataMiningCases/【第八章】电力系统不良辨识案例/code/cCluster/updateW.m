function wM = updateW(X,C,p)

[Xnum D] = size(X);
[Cnum DD] = size(C);

wM = zeros(Xnum,Cnum);

if D~=DD
    error('Dimension Error!');
end

for i=1:Xnum
    wt = zeros(Cnum,1);
    xt = X(i,:);
    flag_cal = 1;
    for j=1:Cnum
        ct = C(j,:);
        wt(j) = calDist2(xt,ct);
        if wt(j)==0
            wM(i,j) = 1;
            flag_cal = 0;
            break;
        end
    end
    if flag_cal
        wt = (1./wt).^(1/(p-1));
        sumwt = sum(wt);
        wM(i,:) = wt/sumwt;
    end
end

end