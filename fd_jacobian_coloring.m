function J = fd_jacobian_coloring(rfun, x, k, use_relative, bandwidth)
    r0 = rfun(x); % F(x)
    n  = length(x);
    m  = length(r0);
    h0 = 10^(-k);

    if use_relative
        h = h0 * abs(x);
        h(h == 0) = h0;
    else
        h = h0 * ones(n,1); % costant perturbation
    end

    bw = bandwidth;          % dependency radius in variables
    p  = 2*bw + 1;           % safe coloring
    color = mod((1:n)-1, p) + 1; %cyclic colors assignment

    % Store J sparse banded (optional). For simplicity, build sparse via triplets:
    nnz_est = n*(2*bw+1);
    I = zeros(nnz_est,1); Jc = zeros(nnz_est,1); V = zeros(nnz_est,1);
    t = 0;

    for c = 1:p %we iterates only on p perturbations (colors) and not for variables 
        idx = find(color == c); %indexes of variables with this colour
        d = zeros(n,1);
        d(idx) = h(idx); %perturbation vector

        r1 = rfun(x + d); % F(x+d)
        dr = r1 - r0;  % dr ≈ J*d deltaR

        % start the distribution phase
    
        for s = 1:numel(idx)
            j = idx(s); % we iterate over the perturbed variables
            
            if m == n
                % --- PROBLEM 31 ---
                % every x_j impacts f_{j-1}, f_j, f_{j+1}
                row_start = max(1, j - 1);
                row_end   = min(m, j + 1);
            else
                % --- PROBLEM 49  ---
                % m = 2(n-1). The residuals are coupled (e_i, o_i).
                % x_j influences the residuals approximately between 2j-3 and 2j+1
                row_start = max(1, 2*j - 3);
                row_end   = min(m, 2*j + 1);
            end
            
            rows = row_start:row_end; % residuals affected by x_j
            
            % Extraction and calculation of the difference quotient % deltaR(fk)/h 
            vals = dr(rows) / h(j);
            
            % Instead of creating a 100,000x100,000 Jacobian matrix,
            % the code only saves the non-zero values.
            % Remember, the Jacobian is very sparse.
            
            % Saving into triplets
            L = numel(rows);
            I(t+1:t+L)  = rows;
            Jc(t+1:t+L) = j;
            V(t+1:t+L)  = vals;
            t = t + L;
        end
    end
    % we create the Jacobian based on those 3 vectors
    J = sparse(I(1:t), Jc(1:t), V(1:t), m, n);
end
