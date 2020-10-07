function main()
clear; clc; close all
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];
%% select CA, OR, WA, NJ, NY counties
ind = find((A(:,1)>=6000 & A(:,1)<=6999)); %...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
A = A(ind,:);

[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties
% ngop = length(igop);
% ndem = length(idem);
% if ngop > ndem
%     rgop = randperm(ngop,ndem);
%     Adem = A(idem,:);
%     Agop = A(igop(rgop),:);
%     A = [Adem;Agop];
% else
%     rdem = randperm(ndem,ngop);
%     Agop = A(igop,:);
%     Adem = A(idem(rdem),:);
%     A = [Adem;Agop];
% end  
% [n,dim] = size(A)
% idem = find(A(:,2) >= A(:,3));
% igop = find(A(:,2) < A(:,3));
% num = A(:,2)+A(:,3);
% label = zeros(n,1);
% label(idem) = -1;
% label(igop) = 1;

%% set up data matrix and visualize
close all
figure;
hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',15);
plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',15);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% rescale data to [0,1] and visualize
figure;
hold on; grid;
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',15);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',15);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% Solve unconstrained optimization problem in Q2
[n,dim] = size(XX); % dim=3 here
lam = 0.1;% 0.01
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];% Y upper-left matrix in (56);
w = [-1;-1;1;1];
fun = @(I,Y,w)fun0(I,Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v)Hvec0(I,Y,w,v,lam);

% tic % start timer
% Using Subsampled Inexact Newton (Excercise 3)
% [w,f,gnorm] = SINewton(fun,gfun,Hvec,Y,w,100,1e5); 

% Using Stochastic Gradient Descent (Exercise 2)
% [w,f,gnorm] = SGD(fun,gfun,Y,w,128,'decreasing',1e5);

% Using Stochastic L-BFGS (Excersise 4)
% N_g = 128; % batch size for gradient
% N_H = 256; % batch size for Hessian
% [w,f,gnorm] = LBFGS(fun,gfun,Y,w,N_g,N_H,100);
% toc % end timer

%% Solve soft SVM via ASM using (55) (56) in Q1
c = 1e2;
[A_SVM,b_SVM] = constraints_SVM(dim,n,Y); % get contraints
% find a feasible point
tic
w = [-1;-1;1;1];
xi = zeros(n,1);
x_init = [w;xi];
[x_init,l_findinit,lcomp_findinit] = FindInitGuess(x_init,A_SVM,b_SVM);
% solve via active set method
save('test','x_init','A_SVM','b_SVM')
W = []; % working set, parameter of ASM
gfun = @(x)g_SVM(x,dim,n,c);
Hfun = @(x)H_SVM(x,dim,n,c);
[xiter,lm] = ASM(x_init,gfun,Hfun,A_SVM,b_SVM,W);
w = xiter(1:4);
% % toc
%% having found w; plot decision boundary
fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'yellow'; % green
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);
% saveas(gcf,'pic/1_CA_SVM.png')
%%
figure;
hold on;
grid;
niter = length(f);
plot((0:niter-1)',f,'Linewidth',2);
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
%%
figure;
hold on;
grid;
niter = length(gnorm);
plot((0:niter-1)',gnorm,'Linewidth',2);
set(gca,'Fontsize',fsz); 
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
%% plot statistics vs runtime and iterations (for Q2-Q4)
iter_max = 200;

w = [-1;-1;1;1];
% [w,f,gnorm,toctime] = SGD(fun,gfun,Y,w,256,'fixed',5e3);
% [w,f,gnorm,toctime] = SINewton(fun,gfun,Hvec,Y,w,5,2e2);
N_g = 128;N_H = 128;
[w,f,gnorm,toctime] = LBFGS(fun,gfun,Y,w,N_g,N_H,70);

f_history = f;
gnorm_history = gnorm;
toctime_history = toctime;

for k = 1 : iter_max
  w = [-1;-1;1;1];
%   [w,f,gnorm,toctime] = SGD(fun,gfun,Y,w,256,'fixed',5e3);
%   [w,f,gnorm,toctime] = SINewton(fun,gfun,Hvec,Y,w,5,2e2);
%    N_g = 30;N_H = 128;
  [w,f,gnorm,toctime] = LBFGS(fun,gfun,Y,w,N_g,N_H,70);
  f_history = f_history + f;
  gnorm_history = gnorm_history + gnorm;
  toctime_history = toctime_history + toctime;
end

f_history = f_history / (iter_max + 1);
gnorm_history = gnorm_history/ (iter_max + 1);
toctime_history = toctime_history / (iter_max + 1);

var(f_history)
var(gnorm_history)

niter = length(gnorm);
figure;
hold on;
grid;
plot((0:niter-1)',f_history,'Linewidth',2);
set(gca,'Fontsize',fsz); 
set(gca,'YScale','log');
xlabel('iterations','Fontsize',fsz);
ylabel('f','Fontsize',fsz);

figure;
hold on;
grid;
plot((0:niter-1)',gnorm_history,'Linewidth',2);
set(gca,'Fontsize',fsz); 
set(gca,'YScale','log');
xlabel('iterations','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);

figure;
hold on;
grid;
plot(toctime_history,f_history,'Linewidth',2);
set(gca,'Fontsize',fsz); 
set(gca,'YScale','log');
xlabel('runtime','Fontsize',fsz);
ylabel('f','Fontsize',fsz);

figure;
hold on;
grid;
plot(toctime_history,gnorm,'Linewidth',2);
set(gca,'Fontsize',fsz); 
set(gca,'YScale','log');
xlabel('runtime','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);


end




%% Utils function %%
%% Soft vector machine loss function and its derivates; see (55),(56) in lecture notes
% x = [w,b,xi]; c trade-off constant; d dim of data and n # of constraints
function f = f_SVM(x,d,n,c) % loss
H = [eye(d), zeros(d,n+1); zeros(n+1,d), zeros(n+1,n+1)];
v = [zeros(1,d+1), ones(1,n)]'; % column vector
f = 0.5 * x' * H * x + c * v' * x;
end
% grad
function g = g_SVM(x,d,n,c) 
H = [eye(d), zeros(d,n+1); zeros(n+1,d), zeros(n+1,n+1)];
v = [zeros(1,d+1), ones(1,n)]'; % column vector
g = H * x + c * v;
end
% Hessian 
function H = H_SVM(x,d,n,c) 
H = [eye(d), zeros(d,n+1); zeros(n+1,d), zeros(n+1,n+1)];
end
% return contraints (55),(56)
function [A,b] = constraints_SVM(d,n,Y) 
A = [Y, eye(n); zeros(n,d+1), eye(n)];
b = [ones(1,n), zeros(1,n)]';
end

%% loss function for unconstrained optimzation problem in exercise 2
function f = fun0(I,Y,w,lam) % f
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end
% grad of f
function g = gfun0(I,Y,w,lam) 
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end
% Hessian of f
function Hv = Hvec0(I,Y,w,v,lam) 
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end