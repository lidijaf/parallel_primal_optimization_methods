U = repmat(1:n,1,d); A=sparse(n,n);
edgTest=0; reps=1;
while ~isempty(U) && reps < matIter    
    edgTest = edgTest + 1;
    i1 = ceil(rand*length(U)); i2 = ceil(rand*length(U));
    v1 = U(i1); v2 = U(i2);
    if (v1 == v2) || (A(v1,v2) == 1)
        if (edgTest == n*d)           
            reps=reps+1; edgTest = 0;
            U = repmat(1:n,1,d); A = sparse(n,n);
        end
    else
        A(v1, v2)=1; A(v2, v1)=1; v = sort([i1,i2]);
        U = [U(1:v(1)-1), U(v(1)+1:v(2)-1), U(v(2)+1:end)];
    end
end
if (norm(G-G','fro')>0) || (max(G(:))>1)
    || (min(sum(G))<sum(G)(1) || max(sum(G))>sum(G)(1)) || 
    norm(diag(G))>0
    ok=0;
else
    ok=1;
end
