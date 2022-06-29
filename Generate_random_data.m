Adata=zeros(n*s,s);
Bdata=zeros(n,s);
for i=1:n
    Temp = randn(s);
    Temp = (Temp+Temp')/2;
    [Q,Lam] = eig(Temp);
    Avector = 1000*rand(s,1) + ones(s,1);
    Adata((i-1)*s+1:(i-1)*s+s, 1:s) = Q*diag(Avector)*Q';
    Bdata(i,:) = ones(1,s)+rand(1,s).*30;
end
