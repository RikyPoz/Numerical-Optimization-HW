function J = fd_jacobian_coloring(rfun, x, k, use_relative, bandwidth)
    r0 = rfun(x);
    n  = length(x);
    m  = length(r0);
    h0 = 10^(-k);

    if use_relative
        h = h0 * abs(x);
        h(h == 0) = h0;
    else
        h = h0 * ones(n,1);
    end

    bw = bandwidth;          % dependency radius in variables
    p  = 2*bw + 1;           % safe coloring
    color = mod((1:n)-1, p) + 1;

    % Store J sparse banded (optional). For simplicity, build sparse via triplets:
    nnz_est = n*(2*bw+1);
    I = zeros(nnz_est,1); Jc = zeros(nnz_est,1); V = zeros(nnz_est,1);
    t = 0;

    for c = 1:p
        idx = find(color == c);
        d = zeros(n,1);
        d(idx) = h(idx);

        r1 = rfun(x + d);
        dr = r1 - r0;  % dr ≈ J*d

        for s = 1:numel(idx)
            j = idx(s);
            % rows affected by x_j: depends on your problem’s residual locality.
            % If each residual touches vars within bw, then each column touches residual rows within bw too.
            % If you don’t want to derive exact residual-row bands, just take all rows: O(m) each -> slower.
            % Better: implement residual-row range per problem.
            rows = 1:m;  % <-- replace with tighter rows for efficiency if you can

            vals = dr(rows) / h(j);
            L = numel(rows);

            I(t+1:t+L)  = rows;
            Jc(t+1:t+L) = j;
            V(t+1:t+L)  = vals;
            t = t + L;
        end
    end

    J = sparse(I(1:t), Jc(1:t), V(1:t), m, n);
end
