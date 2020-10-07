function [w,f,normgrad] = SGD(w,Y,fun,gfun,maxiter,stepsize,batchsize,dec_step)
% Stochastic Gradient Descent
[n, d] = size(Y);
f = zeros(maxiter,1);
normgrad = zeros(maxiter,1);
f(1) = fun(1:n,Y,w);
dec = stepsize/maxiter;
for k = 1 : maxiter
    if dec_step == 1
        stepsize = stepsize - dec;
    end
    I = randperm(n,batchsize);
    w = w - stepsize * gfun(I,Y,w);    
    f(k) = fun(1:n,Y,w);
    g = gfun(1:n,Y,w);
    normgrad(k) = norm(g);
end