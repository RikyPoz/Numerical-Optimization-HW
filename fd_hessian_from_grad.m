function H = fd_hessian_from_grad(gfun, x, k, use_relative, bandwidth)
% FD Hessian from gradient with banded coloring (half-bandwidth = bandwidth)

    n  = length(x);
    g0 = gfun(x);
    h0 = 10^(-k);

    if use_relative
        h = h0 * abs(x);
        h(h == 0) = h0;         % safeguard only for zeros
    else
        h = h0 * ones(n,1);
    end

    % Correct coloring for half-bandwidth m: p = 2m+1
    m = bandwidth;
    p = 2*m + 1;
    color = mod((1:n)-1, p) + 1;

    nnz_est = n * (2*m + 1);
    I = zeros(nnz_est, 1);
    J = zeros(nnz_est, 1);
    V = zeros(nnz_est, 1);
    t = 0;

    for c = 1:p
        idx = find(color == c);

        d = zeros(n,1);
        d(idx) = h(idx);

        g1 = gfun(x + d);
        dg = g1 - g0;  % dg ≈ H * d

        for s = 1:numel(idx)
            j = idx(s);
            rows = max(1, j-m):min(n, j+m);

            vals = dg(rows) / h(j);

            L = numel(rows);
            I(t+1:t+L) = rows;
            J(t+1:t+L) = j;
            V(t+1:t+L) = vals;
            t = t + L;
        end
    end

    H = sparse(I(1:t), J(1:t), V(1:t), n, n);
    H = 0.5 * (H + H.');
end
