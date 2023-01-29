function [c,mu] = kmeans(x,K,mu)
% K-means clustering
% [c,mu] = kmeans(x,K)
%	x  - d*n samples
%	K  - number of clusters wanted
%	mu	- d*K initial guess for cluster centroids
% returns:
%	c  - 1*n calculated membership vector
%	mu	- d*K cluster centroids
[d,n] = size(x);
K=round(K);
oldmu = Inf*ones(d,K);
c = zeros(1,n);
D = zeros(K,n);

if nargin<3
   p=randperm(n);
   mu = x(:,p(1:K));
end

while(1)
   for j=1:K,                        
      center = mu(:,j);              
      if ~isequal(center,oldmu(:,j)) 
         D(j,:) = sum((x-center*ones(1,size(x,2))).^2,1);
      end
   end
   oldmu = mu;
   
   % find minimum
   [Dmin,index] = min(D);
   moved = sum(index~=c);
   c = index;

   for i=1:K
      ci=find(c==i);
      mu(:,i)=mean(x(:,ci),2);
   end
   
   if (moved==0), break, end
   
end


