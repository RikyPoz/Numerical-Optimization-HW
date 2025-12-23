% Funzione Obiettivo (F)
function f = prob31_obj(x)
    [f, ~, ~] = prob31_analitico(x);
end

% Gradiente (g)
function g = prob31_grad(x)
    [~, g, ~] = prob31_analitico(x);
end

% Hessiana (H)
function H = prob31_hess(x)
    [~, ~, H] = prob31_analitico(x);
end

function [f, g, H] = prob31_analitico(x)
% Analytical evaluation of the Broyden Tridiagonal function.
% This function computes the objective value, gradient, and Hessian for 
% Problem 31 from the UFO collection.

%important: Vectorization: Instead of using for loops, the code uses vector indexing (e.g., fk(2:n)) 
% to align x k−1 and xk+1 terms. This is essential for the performance requirements of large-scale problems (n=10^5).


    n = length(x);
    
    % --- 1. COMPUTE RESIDUALS (fk) ---
    % The residual formula is: fk = (3 - 2*xk)*xk - x(k-1) - 2*x(k+1) + 1.
    % We use boundary conditions x0 = 0 and xn+1 = 0.
    
    % Step A: Compute the central part of the formula for all k: (3 - 2*xk)*xk + 1
    fk = (3 - 2*x).*x + 1;
    
    % Step B: Vectorized subtraction of x(k-1).
    % fk(2:n) refers to indices k=2 to n; x(1:n-1) refers to indices k-1=1 to n-1.
    fk(2:n) = fk(2:n) - x(1:n-1);
    
    % Step C: Vectorized subtraction of 2*x(k+1).
    % fk(1:n-1) refers to indices k=1 to n-1; x(2:n) refers to indices k+1=2 to n.
    fk(1:n-1) = fk(1:n-1) - 2*x(2:n);
    
    % --- 2. COMPUTE OBJECTIVE FUNCTION (f) ---
    % The function is defined as a sum of squares: F(x) = 0.5 * sum(fk^2).
    f = 0.5 * sum(fk.^2);
    
    % --- 3. COMPUTE JACOBIAN MATRIX (J) ---
    % The Jacobian J contains partial derivatives of each fk with respect to each xi.
    % dfk/dxk = 3 - 4*xk
    % dfk/dx(k-1) = -1
    % dfk/dx(k+1) = -2
    main_diag = 3 - 4*x;
    
    % Use sparse matrices (spdiags) to handle large dimensions (n=10^5) efficiently.
    % We define the sub-diagonal (-1), main diagonal, and super-diagonal (-2).
    J = spdiags(main_diag, 0, n, n) + ...
        spdiags([-ones(n,1), [0; -2*ones(n-1,1)]], [-1, 1], n, n);
    
    % --- 4. COMPUTE EXACT GRADIENT (g) ---
    % For a sum of squares, the gradient is g = J' * f.
    g = J' * fk;
    
    % --- 5. COMPUTE EXACT HESSIAN (H) ---
    % The Hessian for a sum of squares is H = J'J + sum(fk * nabla^2(fk)).
    % Since fk is quadratic, its second derivative is constant: d^2(fk)/dxk^2 = -4.
    % S represents the second-order term matrix, which is diagonal here.
    S = spdiags(-4 * fk, 0, n, n);
    
    % H = J'J + S provides the exact curvature information for Newton methods.
    H = J' * J + S;
end