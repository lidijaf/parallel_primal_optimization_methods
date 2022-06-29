A=zeros(n);
for i=1:n-1
  if mod(i,width)!=0    A(i,i+1)=1;    end;
  if (i+width)<=n    A(i,i+width)=1;    end;
end;
for i=1:n
  for j=i+1:n
    A(j,i)=A(i,j);
  end;
end;
degreeSensor = A * ones(n,1);
