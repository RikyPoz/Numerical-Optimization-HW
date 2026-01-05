function H = fd_hessian_from_grad(gfun, x, k, use_relative, bandwidth)
% Finite-difference Hessian approximation from exact gradient
% Efficient implementation exploiting banded structure (coloring)
%
% gfun         : exact gradient handle
% x            : evaluation point
% k            : exponent in h = 10^{-k}
% use_relative : false -> h_j = 10^{-k}
%                true  -> h_j = 10^{-k} * |x_j|
% bandwidth    : half-bandwidth of Hessian (2 for prob 31 & 49 
%                because Hessian is nonzero only for |i-j|<=2)    
    
    n = length(x);
    g0 = gfun(x);
    h0 = 10^(-k);

    if use_relative
        h = h0 * abs(x);
        h(h == 0) = h0; % in case h becomes zero we avoid the division
    else 
        h = h0 * ones(n, 1);
    end 

    % Coloring
    p = bandwidth + 1; % # of colors we should use
    color = mod((1:n)-1, p) + 1;

    % preallocating sparse storages
    nnz_est = n * (2 * bandwidth + 1);
    I = zeros(nnz_est, 1);
    J = zeros(nnz_est, 1);
    V = zeros(nnz_est, 1);
    t = 0;

    for c = 1:p % one gradient evaluation per color
        idx = find(color == c); 
        % idx is the list of columns we will compute together
        d = zeros(n, 1); 
        d(idx) = h(idx); % stroing all the indices in a vector 
        g1 = gfun(x + d);
        dg = g1 - g0;
        % dg ≈ H(x)d = j∈idx ∑​hj​H(x)ej
        % in dg we have a superposition of multiple Hessian columns ​

        for s = 1:length(idx)
            j = idx(s); % we take one j from the color class
            rows = max(1, j-bandwidth):min(n, j+bandwidth);
            % rows = {j-m,...,j+m}
            % dg(rows) ≈ hjH(rows,j)

            vals = dg(rows) / h(j);
            % entries of the j-th column, but only within the band

            L = length(rows);
            I(t+1:t+L) = rows; % row indices
            J(t+1:t+L) = j; % column indices (same j repeated)
            V(t+1:t+L) = vals; % values
            % putting vals(k) into H(rows(k), j)
            t = t + L;
        end
    end

    H = sparse(I(1:t), J(1:t), V(1:t), n, n); 

    H = 0.5 * (H + H.'); % Enforcing symmetry
end