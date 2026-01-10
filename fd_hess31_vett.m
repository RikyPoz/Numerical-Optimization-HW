function H = fd_hess31_vett(x, k, use_relative)
    n = length(x);
    h = 10^(-k);
    if use_relative, h = h * max(abs(x), 1e-8); end

    % Conosciamo la struttura: H = J'J + S
    % S è diagonale perché f_k è quadratica solo in x_k
    
    % 1. Calcolo Jacobiana (vettorizzato)
    main_J = ( (3-2*(x+h)).*(x+h) - (3-2*x).*x ) ./ h;
    J = spdiags([[-ones(n-1,1);0], main_J, [0;-2*ones(n-1,1)]], [-1,0,1], n, n);
    
    % 2. Calcolo termine di secondo ordine S_ii = f_k * d^2f_k/dx_k^2
    % d^2f_k/dx_k^2 approssimato con FD
    f = @(x_val) (3-2*x_val).*x_val + 1; % parte quadratica del residuo
    d2f = (f(x+h) - 2*f(x) + f(x-h)) ./ (h.^2);
    
    f_val = (3-2*x).*x + 1 - [0; x(1:end-1)] - [2*x(2:end); 0];
    S = spdiags(f_val .* d2f, 0, n, n);
    
    % 3. Assemblaggio H = J'J + S
    H = J' * J + S;

    
end