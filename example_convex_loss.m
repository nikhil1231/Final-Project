% This is an example how to create a simple convex loss. For the SGD claims we will need non-convex loss thus at least 2
% layers will be needed.

% Matrix with normally distributed entries, mean=0, std=1
A = randn(2,2); 
% take upper right triangle
a = [A(1,1); A(1,2); A(2,2)];
% construct symmetic matrix (3 dof)
% I think this is the weight matrix
% sA = [a(1), a(2); a(2), a(3)];


% function applying sA*x
% Analogous to weights * inputs
sax = @(a,x) [a(1), a(2); a(2), a(3)]*x;

% generate data
% apply the matrix to randomly drawn vectors
% Inputs, predictions
X = randn(2,10);
Y = sax(a, X);


disp(Y)

% grid space for matrix entries
as = linspace(-3,3,300);

% L2 loss function as a function of paramters a and data set (X,Y)
loss2 = @(a,X,Y) sum(reshape((sax(a,X)-Y).^2,[],1));

%disp(X);
%disp(loss2([0,0,0],X,Y));

% Fix the coefficient a(3) and plot the loss2 while varying the other two coefficients. Loss2 in convex in the parameters
for i=1:length(as), for j=1:length(as),  landscape2(i,j) = loss2([as(i); as(j); a(3)], X,Y); end; end
figure,  contourf(as,as,landscape2); colorbar
