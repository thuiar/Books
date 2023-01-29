function Wk = getWk( c,K,d,mu )
Wk=0;
for k=1:K
    index = find(c==k);
    index_num = length(index);
    if index_num ~= 0
        data = d(:,index);
%         Dr = sum(sum((data-mu(:,k)*ones(1,size(data,2))).^2));
        Dr = index_num*sum(sum(data.^2))-sum(sum(data,2).^2);
%         Dr2 = 0;
%         for i=1:index_num
%             for j=1:index_num
%                 Dr2 = (data(i)-data(j))^2+Dr2;
%             end
%         end
        Wk = Wk+Dr/index_num;
%         Wk = Wk+sum((data-mu(k)).^2)/2/index_num;
    end
end
end

