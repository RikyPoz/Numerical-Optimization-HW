function H = fd_hess49_vett(x, k, use_relative)
    n = length(x);
    h0 = 10^(-k);
    if use_relative, h = h0 * max(abs(x), 1e-8); else, h = h0 * ones(n,1); end

    % --- 1. Costruzione Jacobiana J (3 diagonali) ---
    main_J = zeros(n,1); main_J(1) = 1; % contributo f1
    xi = x(1:n-1); xip1 = x(2:n);
    e = 10 * (xi.^2 - xip1);
    de_dxi = (10*((xi + h(1:n-1)).^2 - xip1) - e) ./ h(1:n-1);
    
    % J per residui 'e' è n-1 x n
    Je = spdiags([de_dxi, -10*ones(n-1,1)], [0, 1], n-1, n);
    
    % J per residui 'o' è n-2 x n (se n>=3)
    if n >= 3
        ai = x(1:n-2); bi = x(2:n-1); ci = x(3:n);
        oi_fun = @(a, b, c) 2*exp(-(a-b).^2) + exp(-2*(b-c).^2);
        o = oi_fun(ai, bi, ci);
        do_dxi   = (oi_fun(ai + h(1:n-2), bi, ci) - o) ./ h(1:n-2);
        do_dxip1 = (oi_fun(ai, bi + h(2:n-1), ci) - o) ./ h(2:n-1);
        do_dxip2 = (oi_fun(ai, bi, ci + h(3:n))   - o) ./ h(3:n);
        Jo = spdiags([do_dxi, do_dxip1, do_dxip2], [0, 1, 2], n-2, n);
    else
        Jo = sparse([], [], [], 0, n);
    end
    
    % J totale (includendo f1)
    Jf1 = sparse(1, 1, 1, 1, n);
    J = [Jf1; Je; Jo];

    % --- 2. Costruzione Termine del Secondo Ordine S ---
    % S = sum(f_k * Hess(f_k)). Approssimiamo le diagonali di Hess(f_k)
    S_diag = zeros(n,1);
    
    % Per 'e', solo derivata seconda rispetto a xi è non nulla
    d2e_dxi2 = (10*((xi + h(1:n-1)).^2 - xip1) - 2*e + 10*((xi - h(1:n-1)).^2 - xip1)) ./ (h(1:n-1).^2);
    S_diag(1:n-1) = S_diag(1:n-1) + e .* d2e_dxi2;
    
    % Per 'o', approssimiamo solo la diagonale principale di Hess(oi)
    if n >= 3
        d2o_dxi2 = (oi_fun(ai+h(1:n-2), bi, ci) - 2*o + oi_fun(ai-h(1:n-2), bi, ci)) ./ (h(1:n-2).^2);
        d2o_dxip12 = (oi_fun(ai, bi+h(2:n-1), ci) - 2*o + oi_fun(ai, bi-h(2:n-1), ci)) ./ (h(2:n-1).^2);
        d2o_dxip22 = (oi_fun(ai, bi, ci+h(3:n)) - 2*o + oi_fun(ai, bi, ci-h(3:n))) ./ (h(3:n).^2);
        
        S_diag(1:n-2) = S_diag(1:n-2) + o .* d2o_dxi2;
        S_diag(2:n-1) = S_diag(2:n-1) + o .* d2o_dxip12;
        S_diag(3:n)   = S_diag(3:n)   + o .* d2o_dxip22;
    end
    
    % --- 3. Assemblaggio Finale ---
    H = J' * J + spdiags(S_diag, 0, n, n);
    H = 0.5 * (H + H'); % Simmetria numerica
end