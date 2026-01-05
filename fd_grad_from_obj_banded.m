function g = fd_grad_from_obj_banded(fobj, x, k, use_relative, bandwidth)
%FD_GRAD_FROM_OBJ_BANDED  Efficient FD gradient for local/banded objectives
%
% g_j ≈ ( f(x + h_j e_j) - f(x) ) / h_j
% Efficient by coloring when the objective is local and perturbations don't interfere.
%
% Inputs:
%   fobj        objective handle: f = fobj(x)
%   x           point
%   k           exponent: h = 10^{-k}
%   use_relative  false -> h_j = 10^{-k}; true -> h_j = 10^{-k} * |x_j|
%   bandwidth   choose based on locality (use 2 for prob31/prob49 in practice)
%
% Output:
%   g           FD gradient (n-by-1)

    n = length(x);
    f0 = fboj(x);

    h0 = 10^(-k);
    if use_relative
        h = h0 * abs(x);
        h(h == 0) = h0;
    else
        h = h0 * ones(n,1);
    end

    % For gradient FD, to avoid interference in objective changes,
    % we use p = 2*bandwidth + 1 colors (safer than bandwidth+1).
    % With bandwidth=2 => p=5 objective evaluations per gradient.

    p = 2 * bandwidth + 1;
    color = mod((1:n)-1, p) + 1;

    g = zeros(n, 1);
    for c = 1:p
            
        idx = find(color == c);
        d = zeros(n,1);
        d(idx) = h(idx);

        f1 = fobj(x+d);
        df = f1-f0;

        for s = 1:length(idx)
            j = idx(s);
            g(j) = df / h(j);
        end

    end
end