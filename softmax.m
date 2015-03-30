function a = softmax(n)

nmax = max(n,[],2);
n = bsxfun(@minus,n,nmax);

numer = exp(n);
denom = sum(numer,2); 
denom(denom == 0) = 1;
a = bsxfun(@rdivide,numer,denom);