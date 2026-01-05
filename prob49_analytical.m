% Objective function handle
function f = prob49_obj(x)
    [f, ~, ~] = prob49_analytical(x);
end

% Gradient handle
function g = prob49_grad(x)
    [~, g, ~] = prob49_analytical(x);
end

% Hessian handle
function H = prob49_hess(x)
    [~, ~, H] = prob49_analytical(x);
end

function [F, g, H] = prob49_analytical(x)
% problem 49: Attracting-Repelling problem 
% F(x) = 1/2 * sum_{k=1}^m (f_k(x))^2, with m = 2(n-1)
% f_1(x) = x1 - 1;
% for k>1:
%   if k even: f_k(x) = 10*(x_i^2-x_{i+1}), i=k/2
%   if k odd: f_k(x) = 2*exp(-(x_i-x_{i+1})^2) +
%   exp(-2*(x_{i+1}-x{i+2})^2), i = floor(k/2)
%
% suggested start:
%   xbar_i = -1.2 if i odd, xbar_i = 1.0 of i even 

    n = length(x);

    % we should now build the residuals:
    f1 = x(1) -1; % f1

    % even residuals:
    xi = x(1:n-1);
    xip1 = x(2:n);
    e = 10 * (xi.^2 - xip1);

    % odd residuals:
    a = x(1:n-2) - x(2:n-1);
    b = x(2:n-1) - x(3:n);
    A = exp(-(a.^2));
    B = exp(-2 * (b.^2));
    o = 2 * A + B;

    % objective function:
    F = 0.5 * (f1^2 + sum(e.^2) + sum(o.^2));

    % Gradient: g=sum_k f_k * grad(f_k)
    g = zeros(n,1);

    g(1) = g(1) + f1; % contribution from f1
    
    % contribution from even residuals 
    % e_i = 10*(x_i^2 - x_{i+1})
    % de_i/dx_i   = 20*x_i
    % de_i/dx_{i+1} = -10
    g(1:n-1) = g(1:n-1) + e .* (20*xi);
    g(2:n) = g(2:n) + e .* (-10);

    % Contribution from odd residuals o_i
    % o_i = 2*exp(-a^2) + exp(-2*b^2)
    % do/dx_i     = -4*a*A
    % do/dx_{i+1} =  4*a*A - 4*b*B
    % do/dx_{i+2} =  4*b*B
    do_xi = -4 .* a .* A;
    do_xip1 = 4 .* a .* A - 4 .* b .* B;
    do_xip2 = 4 .* b .* B;

    g(1:n-2) = g(1:n-2) + o .* do_xi;
    g(2:n-1) = g(2:n-1) + o.* do_xip1;
    g(3:n) = g(3:n) + o .* do_xip2;

    % Hessian: H = sum_k [ grad(f_k)*grad(f_k)' + f_k * Hess(f_k) ]
    % This produces a sparse (banded) Hessian.

    % Each even residual touches (i, i+1) -> 2x2 block
    % Each odd residual touches (i, i+1, i+2) -> 3x3 block

    nnz_est = 4 *(n-1) + 9*(n-2) + 1;
    I = zeros(nnz_est, 1); 
    J = zeros(nnz_est, 1);
    V = zeros(nnz_est, 1);
    t = 0;

    % f1 term: grad=[1], Hess=0 -> add 1 at (1,1)
    t = t + 1; I(t) = 1; J(t) = 1; V(t) = 1;
    
    % Even residual blocks:
    for i = 1:(n-1)
        fi = e(i);
        xi_val = x(i);

        % grad components
        gi = 20 * xi_val; % de/dx_i
        gip = -10; % de/dx_{i+1}

        h_ii = 20; % Hess of e: only d2/dx_i^2 = 20

        % Contribution: grad*grad' + f*Hess
        %(i,i)
        t=t+1; I(t)=i; J(t)=i; V(t)=gi*gi + fi*h_ii;
        %(i, i+1)
        t=t+1; I(t)=i; J(t)=i+1; V(t)=gi*gip;
        %(i+1, i)
        t=t+1; I(t)=i+1; J(t)=i;   V(t)=gip*gi;
        %(i+1, i+1)
        t=t+1; I(t)=i+1; J(t)=i+1; V(t)=gip*gip;

    end
    
    % Odd residual blocks:
    for i = 1:(n-2)
        fi = o(i);

        ai = a(i);
        bi = b(i);
        Ai = A(i);
        Bi = B(i);

        % grad of o_i on (i, i+1, i+2)
        gi   = -4*ai*Ai;
        gip1 =  4*ai*Ai - 4*bi*Bi;
        gip2 =  4*bi*Bi;

        % Hess of 2*exp(-a^2) part (on i,i+1)
        % d2 = 4*A*(2a^2 - 1)
        cA = 4*Ai*(2*ai^2 - 1);

        H2A = [ cA, -cA,  0;
               -cA,  cA,  0;
                 0,   0,  0 ];

        % Hess of exp(-2 b^2) part (on i+1,i+2)
        % d2 = 4*B*(4b^2 - 1)
        cB = 4*Bi*(4*bi^2 - 1);

        HB = [ 0,   0,   0;
               0,  cB, -cB;
               0, -cB,  cB ];

        Hfi = H2A + HB;
        
        % Contribution: grad*grad' + f*Hess
        gg = [gi; gip1; gip2];
        block = (gg*gg.') + fi*Hfi;

        idx = [i, i+1, i+2];
        for arow = 1:3
            for acol = 1:3
                t = t + 1;
                I(t) = idx(arow);
                J(t) = idx(acol);
                V(t) = block(arow, acol);
            end
        end
    end

    % adds value V(k) to position (I(k), J(k)) of the matrix
    H = sparse(I(1:t), J(1:t), V(1:t), n, n);
       

end



