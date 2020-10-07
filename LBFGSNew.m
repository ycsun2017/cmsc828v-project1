function [x,f,normgrad,toctime] = LBFGSNew(func,gfun,Y,x,N_g,N_H,kmax)
%% Stochastic Limited Memory BFGS
% N_g, N_H are batch size for grad and inverse Hessian
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor

tol = 1e-10;
% tol = -1; % this is only for tracking run time; always run kmax iterations
m = 5; % the number of steps to keep in memory
M = 10; % update inverse Hessian every M iterations
dim = size(x,1); % size of variable
n = size(Y,1); % data size
N_g = min(n,N_g); % batch size
N_H = min(n,N_H);

% s, y, rho are for LBFGS (finddirection.m)
s = zeros(dim,m); 
y = zeros(dim,m);
rho = zeros(1,m);

% save history
I = 1:n;
% kmax = 1e3; % max of iterations, which is an input
f = zeros(kmax,1); % save f values
normgrad = zeros(kmax,1); % save norm of grad f
toctime = zeros(kmax,1); % save run time 
%%
% first do full batch steepest descent step, so that s and y are not empty
Ig = I;
g = gfun(I,Y,x);
a = linesearch(Ig,x,-g,g,func,Y,eta,gam,jmax)
xnew = x - a*g
gnew = gfun(I,Y,xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);

% stochastic L-BFGS
iter_m = 1; % check if we have m recent pairs. if not, use all previous pairs
tic;
for k = 1 : kmax
    if mod(k,M) == M - 1
        % update Hessian (s, y, rho) using batch of size IH
        IH = randperm(n,N_H); % batch index for Hessian
        I_use = IH;
    else
        % just update gradient and variable this iteration
        Ig = randperm(n,N_g); % batch index for grad
        I_use = Ig;
    end
    g = gfun(I_use,Y,x); % estimate grad using batch

    % find direction using m recent pairs
    if iter_m < m
        I_m = 1 : iter_m;
        p = finddirection(g,s(:,I_m),y(:,I_m),rho(I_m));
    else
        p = finddirection(g,s,y,rho);
    end
    
    % stepsize via backtracking linesearch
    [a,j] = linesearch(I_use,x,p,g,func,Y,eta,gam,jmax);
    if j == jmax
        % linesearch fails. use SGD direction instead of BFGS
        p = -g;
        [a,j] = linesearch(I_use,x,p,g,func,Y,eta,gam,jmax);
    end
    
    % update variable
    if mod(k,M) == M - 1
        step = a*p;
        xnew = x + step;
        gnew = gfun(I_use,Y,xnew);
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        s(:,1) = step;
        y(:,1) = gnew - g;
        rho(1) = 1/(step'*y(:,1));
        
        iter_m = iter_m + 1;
        x = xnew;
        g = gnew;
    else
        x = x + a*p;
        g = gfun(I_use,Y,x);
    end
    
    nor = norm(g);
    p_tmp = gfun(I,Y,x);
    normgrad(k) = norm(p_tmp);
    f(k) = func(I,Y,x);
    
    if nor < tol
        f(k+1:end) = [];
        normgrad(k+1:end) = [];
        fprintf('Stochastic L-BFGS: A local solution is found, iter = %d, with grad norm = %d\n',k, nor);
        return
    end  
    toctime(k) = toc;
end
fprintf('Stochastic L-BFGS: Fail to find a local solution, iter = %d, with grad norm = %d\n',k, nor);
end
%% backtracking linesearch for step length
function [a,j] = linesearch(Ig,x,p,g,fun,Y,eta,gam,jmax)
% return a: step length, j is the number of iterations
% input g: grad; p is direction; Ig batch index for grad
    a = 1;
    f0 = fun(Ig,Y,x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = fun(Ig,Y,xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end