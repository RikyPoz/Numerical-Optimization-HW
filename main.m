%% Script di Test: Derivative-based Optimization - Problema 31
clear; clc;

% --- CONFIGURAZIONE TEAM ---
student_ids = [123456, 234567, 345678]; 
min_id = min(student_ids);
rng(min_id); % Impostazione del seed casuale 

% --- PARAMETRI METODI ---
kmax = 1000;
tolgrad = 1e-6; % Criterio di arresto basato sulla norma del gradiente 
c1 = 1e-4; 
rho = 0.5; 
btmax = 50;

% Per Newton Troncato: forcing term per convergenza superlineare/quadratica 
fterms = @(k, gradfk) min(0.5, sqrt(norm(gradfk))); 
cg_maxit = 500;

% --- PROBLEM HANDLES ---
% You can change them to @prob49_obj, @prob49_grad, @prob49_hess to see the results for problem 49
f_handle = @prob31_obj;
g_handle = @prob31_grad;
h_handle = @prob31_hess;

% --- TEST SETTINGS ---
dimensions = [2, 1e3, 1e4, 1e5]; 
n_rand_points = 5;
results = [];

for n = dimensions
    fprintf('\n==========================================\n');
    fprintf('PROVA DIMENSIONE n = %d\n', n);
    fprintf('==========================================\n');
    
    x_suggested = -ones(n, 1); 
    x_starts = [x_suggested, x_suggested + (2*rand(n, n_rand_points) - 1)];
    % The starting point would be different for problem 49: 
    % x_suggested = zeros(n,1);
    % x_suggested(1:2:end) = -1.2;
    % x_suggested(2:2:end) =  1.0;
    
    for s = 1:size(x_starts, 2)
        x0 = x_starts(:, s);
        fprintf('\n[Punto Iniziale %d/%d]\n', s, size(x_starts, 2));
        
        % --- TEST NEWTON MODIFICATO ---
        fprintf('  > Esecuzione Modified Newton... ');
        tic;
        [~, ~, gnorm_m, k_m, ~, ~] = modified_newton_bcktrck(...
            x0, f_handle, g_handle, h_handle, ...
            kmax, tolgrad, c1, rho, btmax);
        time_m = toc;
        fprintf('Finito. Iterazioni: %d, Tempo: %.2fs, Successo: %d\n', k_m, time_m, gnorm_m < tolgrad);
        
        results = [results; struct('n', n, 'pt', s, 'method', 'Modified', ...
            'gnorm', gnorm_m, 'iters', k_m, 'time', time_m, 'success', gnorm_m < tolgrad)];

        % --- TEST NEWTON TRONCATO ---
        fprintf('  > Esecuzione Truncated Newton... ');
        tic;
        [~, ~, gnorm_t, k_t, ~, ~, ~] = truncated_newton_bcktrck(...
            x0, f_handle, g_handle, h_handle, ...
            kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
        time_t = toc;
        fprintf('Finito. Iterazioni: %d, Tempo: %.2fs, Successo: %d\n', k_t, time_t, gnorm_t < tolgrad);
        
        results = [results; struct('n', n, 'pt', s, 'method', 'Truncated', ...
            'gnorm', gnorm_t, 'iters', k_t, 'time', time_t, 'success', gnorm_t < tolgrad)];
    end
end

% Visualizzazione tabella risultati 
disp(struct2table(results));

%% --- FUNZIONI LOCALI (Problema 31) ---
% Queste funzioni devono stare in fondo al file per essere riconosciute.

function f = prob31_obj(x)
    n = length(x);
    fk = (3 - 2*x).*x + 1;
    fk(2:n) = fk(2:n) - x(1:n-1);
    fk(1:n-1) = fk(1:n-1) - 2*x(2:n);
    f = 0.5 * sum(fk.^2); % Funzione definita come somma di quadrati 
end

function g = prob31_grad(x)
    n = length(x);
    fk = (3 - 2*x).*x + 1;
    fk(2:n) = fk(2:n) - x(1:n-1);
    fk(1:n-1) = fk(1:n-1) - 2*x(2:n);
    
    main_diag = 3 - 4*x;
    % Jacobiana tridiagonale sparsa per n=10^5 [cite: 966]
    J = spdiags(main_diag, 0, n, n) + ...
        spdiags([-ones(n,1), [0; -2*ones(n-1,1)]], [-1, 1], n, n);
    g = J' * fk;
end

function H = prob31_hess(x)
    n = length(x);
    fk = (3 - 2*x).*x + 1;
    fk(2:n) = fk(2:n) - x(1:n-1);
    fk(1:n-1) = fk(1:n-1) - 2*x(2:n);
    
    main_diag = 3 - 4*x;
    J = spdiags(main_diag, 0, n, n) + ...
        spdiags([-ones(n,1), [0; -2*ones(n-1,1)]], [-1, 1], n, n);
    
    % S = sum( f_k * Hess(f_k) ) -> Hess(f_k) è solo -4 sulla diagonale
    S = spdiags(-4 * fk, 0, n, n);
    H = J' * J + S; % Hessiana esatta per problemi ai minimi quadrati
end
