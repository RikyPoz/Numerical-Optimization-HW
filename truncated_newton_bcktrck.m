function [xk, fk, gradfk_norm, k, xseq, btseq, cgiterseq] = ...
    truncated_newton_bcktrck(x0, f, gradf, Hessf, ...
    kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit)
% TRUNCATED_NEWTON_BCKTRCK Metodo di Newton Troncato (Newton-CG).

% Inizializzazioni standard
xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);
cgiterseq = zeros(1, kmax); % Per memorizzare le iterazioni interne del CG

xk = x0;
k = 0;

% Funzione per la condizione di Armijo 
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

while k < kmax
    fk = f(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % Criterio di arresto esterno sulla norma del gradiente 
    if gradfk_norm < tolgrad
        break;
    end
    
    % --- FASE 2.1 e 2.2: SOLUTORE CG PER IL SISTEMA LINEARE Bk*z = ck ---
    % Obiettivo: Risolvere Hessf(xk) * p = -gradf(xk) 
    Hk = Hessf(xk);
    ck = -gradfk; 
    
    % Inizializzazione CG interno [cite: 450-452]
    zk = zeros(size(xk));     % z(0) = 0
    rk = ck;                  % r(0) = ck - Bk*z(0) = ck
    dk = rk;                  % d(0) = r(0)
    etak = fterms(k, gradfk); % Forcing term per Newton Inesatto 
    
    j = 0;
    while j < cg_maxit
        % Calcolo del prodotto Hessiana-direzione per evitare ricalcoli
        Hk_dk = Hk * dk;
        curvatura = dk' * Hk_dk; % d(j)' * Bk * d(j) 
        
        % --- GESTIONE HESSIANA NON DEFINITA POSITIVA --- 
        if curvatura <= 0
            if j == 0
                % Caso ii): Curvatura non positiva subito.
                % Si usa la direzione di Steepest Descent.
                pk = ck;
            else
                % Caso iii): Curvatura non positiva dopo qualche passo.
                % Si usa l'ultimo zk calcolato con successo.
                pk = zk;
            end
            break;
        end
        
        % --- AGGIORNAMENTO CG (Bk è definita positiva lungo dk) --- 
        alpha_cg = (rk' * rk) / curvatura;
        zk_new = zk + alpha_cg * dk;
        rk_new = rk - alpha_cg * Hk_dk; % Residuo aggiornato
        
        % --- CRITERIO DI ARRESTO CG (Newton Inesatto) --- 
        % Si verifica se il residuo relativo è sotto la forcing sequence etak.
        if norm(rk_new) <= etak * norm(ck)
            pk = zk_new;
            j = j + 1;
            break;
        end
        
        % Aggiornamento per l'iterazione CG successiva
        beta_cg = (rk_new' * rk_new) / (rk' * rk);
        dk = rk_new + beta_cg * dk;
        rk = rk_new;
        zk = zk_new;
        j = j + 1;
        
        % Se raggiungiamo cg_maxit senza soddisfare etak, prendiamo l'ultimo zk
        pk = zk;
    end
    
    % --- FASE 2.3: LINE SEARCH (BACKTRACKING + ARMIJO) ---
    % La direzione pk così ottenuta è garantita essere di discesa.
    alpha = 1; % Si prova sempre alpha_0 = 1 per la convergenza quadratica.
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;
    
    bt = 0;
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end
    
    % Se il backtracking fallisce nel trovare un punto accettabile
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        break;
    end
    
    % Aggiornamento stato
    xk = xnew;
    k = k + 1;
    xseq(:, k) = xk;
    btseq(k) = bt;
    cgiterseq(k) = j;
end

% "Pulizia" sequenze e aggiunta x0
xseq = [x0, xseq(:, 1:k)];
btseq = btseq(1:k);
cgiterseq = cgiterseq(1:k);
fk = f(xk);
gradfk_norm = norm(gradf(xk));

end