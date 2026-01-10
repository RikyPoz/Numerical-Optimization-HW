function g = fd_grad49_vett(x, k, use_relative)
    n = length(x);
    h0 = 10^(-k);
    if use_relative, h = h0 * max(abs(x), 1e-8); else, h = h0 * ones(n,1); end

    % --- 1. Residui base ---
    xi = x(1:n-1); xip1 = x(2:n);
    e = 10 * (xi.^2 - xip1);
    
    g = zeros(n,1);
    g(1) = (x(1) - 1); % Gradiente di 0.5*(x1-1)^2
    
    % --- 2. Contributo 'e' (Even) ---
    % de/dxi approssimato (analitico sarebbe 20*xi)
    de_dxi = (10*((xi + h(1:n-1)).^2 - xip1) - e) ./ h(1:n-1);
    g(1:n-1) = g(1:n-1) + e .* de_dxi;
    g(2:n)   = g(2:n)   + e .* (-10); % de/dxip1 è costante -10

    % --- 3. Contributo 'o' (Odd) ---
    if n >= 3
        ai = x(1:n-2); bi = x(2:n-1); ci = x(3:n);
        oi_fun = @(a, b, c) 2*exp(-(a-b).^2) + exp(-2*(b-c).^2);
        o = oi_fun(ai, bi, ci);

        % FD locali per le 3 componenti del gradiente di o_i
        do_dxi   = (oi_fun(ai + h(1:n-2), bi, ci) - o) ./ h(1:n-2);
        do_dxip1 = (oi_fun(ai, bi + h(2:n-1), ci) - o) ./ h(2:n-1);
        do_dxip2 = (oi_fun(ai, bi, ci + h(3:n))   - o) ./ h(3:n);

        g(1:n-2) = g(1:n-2) + o .* do_dxi;
        g(2:n-1) = g(2:n-1) + o .* do_dxip1;
        g(3:n)   = g(3:n)   + o .* do_dxip2;
    end
end