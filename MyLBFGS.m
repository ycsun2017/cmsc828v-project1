function [x,f,gnorm,runtime] = MyLBFGS(func,gfun,x0,Y,maxiter,N_g,N_h,M)
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
tol = 1e-10;
m = 5; % the number of steps to keep in memory
n = size(Y,1);
d = size(x0,1);
II = 1:n;

%% 
s = zeros(d,m);
y = zeros(d,m);
rho = zeros(1,m);
gnorm = zeros(1,maxiter+1);
f = zeros(1,maxiter+1);
runtime = zeros(1,maxiter+1);
%
x = x0;
Ig = randperm(n,N_g);
g = gfun(Ig,Y,x);
gnorm(1) = norm(g);
f(1) = func(II,Y,x);
runtime(1) = 0;
tic;
% first do steepest decend step
a = linesearch(x,II,Y,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
Ig = randperm(n,N_g);
gnew = gfun(Ig,Y,xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
gnorm(2) = nor;
f(2) = func(II,Y,x);
runtime(2) = toc;
iter = 1;
while nor > tol
    if iter >= maxiter
        break
    end
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,j] = linesearch(x,Ig,Y,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = linesearch(x,Ig,Y,p,g,func,eta,gam,jmax);
    end
    step = a*p;
    xnew = x + step;
    Ig = randperm(n,N_g);
    gnew = gfun(Ig,Y,xnew);
    % update the pairs for Hessian
    if mod(iter,M) == 0
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        s(:,1) = step;
        Ih = randperm(n,N_h);
        g_h = gfun(Ih,Y,x);
        gnew_h = gfun(Ih,Y,xnew);
        y(:,1) = gnew_h - g_h;
        rho(1) = 1/(step'*y(:,1));
    end
    x = xnew;
    g = gnew;
    nor = norm(g);
    iter = iter + 1;
    % save statistics
    gnorm(iter+1) = nor;
    f(iter+1) = func(II,Y,x);
    runtime(iter+1) = toc;
end
% if nor < tol
%     iter
%     f
% end
% gnorm(iter+1:end) = [];
% f(iter+1:end) = [];
% runtime(iter+1:end) = [];
end

%%
function [a,j] = linesearch(x,I,Y,p,g,func,eta,gam,jmax)
    a = 1;
    f0 = func(I,Y,x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = func(I,Y,xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end