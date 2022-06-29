r = sqrt(log(n)/n); P = rand(n,2); %x,y coordinates
for k=1:n
    for l=1:n
        if k<l
            distSquared = (P(k,1)-P(l,1))^2 + (P(k,2)-P(l,2))^2;
            if distSquared < r^2
                PROB2(k,l) = 0; PROB2(l,k) = 0;
            else
                PROB2(k,l) = 1; PROB2(l,k) = 1;
%here come the end 4 times, for the branches and loops
A = sign(ones(n) - eye(n) - PROB2);
degreeSensor = A * ones(n,1);
