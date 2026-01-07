function [F, g, H] = prob49_analytical(x)
% PROB49_ANALYTICAL  Problem 49 (Attracting–Repelling), exact F, g, H (efficient)
%
% F(x) = 1/2 * sum f_k(x)^2,  m = 2(n-1)
% f1(x) = x1 - 1
% even residuals: e_i = 10*(x_i^2 - x_{i+1}),     i = 1..n-1
% odd  residuals: o_i = 2*exp(-(x_i-x_{i+1})^2) + exp(-2*(x_{i+1}-x_{i+2})^2),
%                 i = 1..n-2
%
% Hessian is pentadiagonal (half-bandwidth 2) and assembled via spdiags.

    n = length(x);

    % -----------------------------
    % Residuals
    % -----------------------------
    f1 = x(1) - 1;

    % even residuals (length n-1)
    xi   = x(1:n-1);
    xip1 = x(2:n);
    e = 10 * (xi.^2 - xip1);

    % odd residuals (length n-2) if n>=3
    if n >= 3
        a = x(1:n-2) - x(2:n-1);
        b = x(2:n-1) - x(3:n);
        A = exp(-(a.^2));
        B = exp(-2*(b.^2));
        o = 2*A + B;
    else
        a = zeros(0,1); b = zeros(0,1);
        A = zeros(0,1); B = zeros(0,1);
        o = zeros(0,1);
    end

    % -----------------------------
    % Objective
    % -----------------------------
    F = 0.5 * (f1^2 + sum(e.^2) + sum(o.^2));

    % -----------------------------
    % Gradient: g = sum f_k * grad(f_k)
    % -----------------------------
    g = zeros(n,1);

    % f1 contribution
    g(1) = g(1) + f1;

    % even contributions
    % de_i/dx_i = 20*x_i,  de_i/dx_{i+1} = -10
    g(1:n-1) = g(1:n-1) + e .* (20*xi);
    g(2:n)   = g(2:n)   + e .* (-10);

    % odd contributions (if any)
    if n >= 3
        % o_i derivatives:
        % do/dx_i   = -4*a*A
        % do/dx_{i+1} = 4*a*A - 4*b*B
        % do/dx_{i+2} = 4*b*B
        do_xi   = -4 .* a .* A;
        do_xip1 =  4 .* a .* A - 4 .* b .* B;
        do_xip2 =  4 .* b .* B;

        g(1:n-2) = g(1:n-2) + o .* do_xi;
        g(2:n-1) = g(2:n-1) + o .* do_xip1;
        g(3:n)   = g(3:n)   + o .* do_xip2;
    end

    % -----------------------------
    % Hessian: H = sum [ grad(f_k)grad(f_k)' + f_k*Hess(f_k) ]
    % Efficient pentadiagonal assembly via diagonals.
    % -----------------------------

    % Initialize 5 diagonals for half-bandwidth 2
    d0  = zeros(n,1);      % main diag
    d1p = zeros(n-1,1);    % +1 diag
    d1m = zeros(n-1,1);    % -1 diag
    d2p = zeros(n-2,1);    % +2 diag
    d2m = zeros(n-2,1);    % -2 diag

    % f1 term: grad=[1] => adds 1 at (1,1)
    d0(1) = d0(1) + 1;

    % ---- Even residual blocks (i = 1..n-1)
    gi_e  = 20 * xi;   % de/dx_i
    gip_e = -10;       % de/dx_{i+1}
    % Hess(e_i): only d2/dx_i^2 = 20
    % Contribution:
    % (i,i)     += gi^2 + e*20
    % (i,i+1)   += gi*gip
    % (i+1,i)   += gip*gi
    % (i+1,i+1) += gip^2
    d0(1:n-1) = d0(1:n-1) + gi_e.^2 + 20*e;
    d0(2:n)   = d0(2:n)   + (gip_e^2);

    d1p(1:n-1) = d1p(1:n-1) + gi_e * gip_e;
    d1m(1:n-1) = d1m(1:n-1) + gip_e * gi_e;

    % ---- Odd residual blocks (i = 1..n-2), if n>=3
    if n >= 3
        % Gradient pieces (length n-2)
        gi   = -4 .* a .* A;
        gip1 =  4 .* a .* A - 4 .* b .* B;
        gip2 =  4 .* b .* B;

        % Second-derivative coefficients for Hess(o_i)
        % For 2*exp(-a^2): cA = 4*A*(2a^2 - 1)
        % For exp(-2b^2):  cB = 4*B*(4b^2 - 1)
        cA = 4 .* A .* (2.*a.^2 - 1);
        cB = 4 .* B .* (4.*b.^2 - 1);

        % Add gg*gg' + o*Hfi to diagonals directly.

        % Main diagonal
        d0(1:n-2) = d0(1:n-2) + gi.^2     + o .* cA;          % (i,i)
        d0(2:n-1) = d0(2:n-1) + gip1.^2   + o .* (cA + cB);   % (i+1,i+1)
        d0(3:n)   = d0(3:n)   + gip2.^2   + o .* cB;          % (i+2,i+2)

        % +/-1 diagonals
        d1p(1:n-2) = d1p(1:n-2) + gi .* gip1   - o .* cA;     % (i,i+1)
        d1m(1:n-2) = d1m(1:n-2) + gip1 .* gi   - o .* cA;     % (i+1,i)

        d1p(2:n-1) = d1p(2:n-1) + gip1 .* gip2 - o .* cB;     % (i+1,i+2)
        d1m(2:n-1) = d1m(2:n-1) + gip2 .* gip1 - o .* cB;     % (i+2,i+1)

        % +/-2 diagonals
        d2p(1:n-2) = d2p(1:n-2) + gi .* gip2;                 % (i,i+2)
        d2m(1:n-2) = d2m(1:n-2) + gip2 .* gi;                 % (i+2,i)
    end

    % Assemble sparse pentadiagonal Hessian
    H = spdiags([d2m, d1m, d0, d1p, d2p], [-2, -1, 0, 1, 2], n, n);

    % Numerical symmetry safeguard (should already be symmetric)
    H = 0.5 * (H + H.');
end