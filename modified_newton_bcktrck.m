function [xk, fk, gradfk_norm, k, xseq, btseq] = ...
    modified_newton_bcktrck(x0, f, gradf, Hessf, ...
    kmax, tolgrad, c1, rho, btmax)
% MODIFIED_NEWTON_BCKTRCK Metodo di Newton Modificato con Line Search.
%
% Questo metodo arricchisce i metodi basati sul gradiente aggiungendo informazioni
% sulle derivate di secondo ordine (Hessiana) per migliorare il tasso di 
% convergenza.

% --- PARAMETRI EURISTICI PER LA MODIFICA DELL'HESSIANA ---
% Se l'Hessiana non è definita positiva, il modello quadratico m_k(p) è 
% illimitato inferiormente. Si corregge l'Hessiana con una matrice 
% E_k tale che B_k = Hessf + E_k sia definita positiva.
beta = 1e-3; % Parametro per la scelta di tau_k iniziale.

% Condizione di Armijo per garantire la convergenza globale[cite: 93, 163].
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

% Inizializzazioni
xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);
xk = x0;
k = 0;

while k < kmax
    fk = f(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % Criterio di arresto basato sulla stazionarietà: grad f(x*) = 0.
    if gradfk_norm < tolgrad
        break;
    end
    
    % --- FASE 1: COSTRUZIONE DELLA MATRICE B_k (Hessiana Modificata) ---
    % Il metodo di Newton richiede che l'Hessiana sia definita positiva affinché 
    % la direzione calcolata sia di discesa.
    Hk = Hessf(xk);
    n = size(Hk, 1);
    
    % Strategia per computare E_k = tau*I[cite: 200, 202].
    % Una condizione necessaria (ma non sufficiente) affinché Hk sia definita 
    % positiva è che tutti gli elementi sulla diagonale siano positivi.
    min_diag = min(diag(Hk));
    if min_diag > 0
        tau_j = 0; % Se la diagonale è positiva, "ci fidiamo" e partiamo da tau=0.
    else
        % Altrimenti, impostiamo un valore iniziale per tau.
        tau_j = beta - min_diag;
    end
    
    % Ciclo di verifica della positività tramite decomposizione di Cholesky.
    while true
        Bk = Hk + tau_j * speye(n); % Usa speye invece di eye per risparmiare RAM
        
        % La decomposizione R^T*R = B_k funziona solo se B_k è definita positiva.
        [R, p_chol] = chol(Bk);
        
        if p_chol == 0
            % B_k è sufficientemente definita positiva.
            break;
        else
            % Se Cholesky fallisce, Bk non è definita positiva; aumentiamo tau.
            % Incremento euristico: tau = max(2*tau, beta)[cite: 254].
            tau_j = max(2 * tau_j, beta);
        end
    end
    
    % --- FASE 2: COMPUTAZIONE DELLA DIREZIONE DI NEWTON MODIFICATA ---
    % Risolviamo il sistema lineare B_k * p = -grad f(x).
    % Poiché B_k è definita positiva, p_MN è una direzione di discesa.
    % Sfruttiamo i fattori di Cholesky R già calcolati per efficienza.
    pk = - (R \ (R' \ gradfk));
    
    % --- FASE 3: LINE SEARCH (BACKTRACKING + ARMIJO) ---
    % Per default si prova il passo alpha = 1 per mantenere, quando possibile, 
    % il tasso di convergenza quadratico locale del metodo di Newton.
    alpha = 1; 
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;
    
    bt = 0;
    % Se la condizione di Armijo non è soddisfatta, riduciamo alpha.
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha; % Riduzione del passo.
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end
    
    % Aggiornamento dello stato dell'algoritmo
    xk = xnew;
    k = k + 1;
    xseq(:, k) = xk;
    btseq(k) = bt;
end

% Preparazione dei risultati finali
xseq = [x0, xseq(:, 1:k)];
btseq = btseq(1:k);
fk = f(xk);
gradfk_norm = norm(gradf(xk));

end