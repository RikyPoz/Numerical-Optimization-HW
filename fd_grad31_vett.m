function g = fd_grad31_vett(x, k, use_relative)
    n = length(x);
    h0 = 10^(-k);
    
    if use_relative
        h = h0 * max(abs(x), 1e-8); % Incremento relativo 
    else
        h = h0 * ones(n,1); % Incremento fisso 
    end

    % Funzione residui vettorizzata (BC: x0=0, xn+1=0) 
    get_f = @(x) (3 - 2*x).*x + 1 - [0; x(1:end-1)] - [2*x(2:end); 0];
    
    % Obiettivo F(x) = 0.5 * sum(f_k^2) [cite: 404]
    F0 = 0.5 * sum(get_f(x).^2);
    
    g = zeros(n,1);
    
    % VERSIONE OTTIMIZZATA J' * f 
    f_val = get_f(x);
    % Approssimiamo le 3 diagonali della Jacobiana J
    % J_ii = df_i/dx_i, J_{i,i-1} = df_i/dx_{i-1}, J_{i,i+1} = df_i/dx_{i+1}
    
    main_diag = (( (3-2*(x+h0)).*(x+h0)+1 ) - ( (3-2*x).*x+1 )) / h0;
    sub_diag  = -ones(n-1,1); % df_k/dx_{k-1} è costante -1
    sup_diag  = -2*ones(n-1,1); % df_k/dx_{k+1} è costante -2
    
    J = spdiags([ [sub_diag; 0], main_diag, [0; sup_diag] ], [-1, 0, 1], n, n);
    g = J' * f_val; % g = J^T * f

    %-> sfruttiamo il fatto che una perturbazione incide solo su pochi
    %residui perchè è questo che rende J sparsa e ci rende possibile g = J^T * f
end