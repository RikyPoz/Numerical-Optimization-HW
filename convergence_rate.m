function p = convergence_rate(gradseq)
    % estimate_rate: Stima l'ordine di convergenza p (es. 1 per lineare, 2 per quadratica)
    % Basato sul rapporto tra le ultime norme dei gradienti della sequenza.
    
    if length(gradseq) < 4
        % Non ci sono abbastanza iterazioni per una stima affidabile
        p = NaN; 
        return;
    end
    
    % Prendiamo gli ultimi 3 valori significativi della sequenza
    % p \approx log(||g_{k}|| / ||g_{k-1}||) / log(||g_{k-1}|| / ||g_{k-2}||)
    g_k   = gradseq(end);
    g_km1 = gradseq(end-1);
    g_km2 = gradseq(end-2);
    
    % Evitiamo divisioni per zero o logaritmi di numeri non positivi
    if g_k > 0 && g_km1 > 0 && g_km2 > 0 && g_km1 ~= g_km2
        p = log(g_k / g_km1) / log(g_km1 / g_km2);
    else
        p = NaN;
    end
end