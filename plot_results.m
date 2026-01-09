function plot_results(results, problem_ids, dimensions, tolgrad)
    % plot_results: Genera i grafici per l'Assignment 1.
    % Ogni punto di partenza ha un colore unico e i metodi sono distinti per stile.
    
    if isempty(results)
        warning('Nessun risultato da plottare.');
        return;
    end

    all_fd_cases = string({results.FDcase}); 
    fd_cases = unique(all_fd_cases); 
    
    for f = 1:length(fd_cases)
        current_fd = fd_cases(f);
        
        for prob_id = problem_ids
            %% --- 1. SEQUENCE PATHS (n=2) ---
            res_n2 = results(all_fd_cases == current_fd & [results.Prob] == prob_id & [results.n] == 2);
            
            if ~isempty(res_n2)
                figure('Name', sprintf('Prob %d - Paths (%s)', prob_id, current_fd), 'Color', 'w');
                hold on;
                
                % Setup area e curve di livello
                all_pts = [res_n2.xseq];
                margin = 0.5;
                x_range = [min(all_pts(1,:))-margin, max(all_pts(1,:))+margin];
                y_range = [min(all_pts(2,:))-margin, max(all_pts(2,:))+margin];
                [X, Y] = meshgrid(linspace(x_range(1), x_range(2), 100), linspace(y_range(1), y_range(2), 100));
                
                % Calcolo della Z per le curve di livello
                Z = zeros(size(X));
                for i = 1:size(X,1)
                    for j = 1:size(X,2)
                        Z(i,j) = get_f_val(prob_id, [X(i,j); Y(i,j)]);
                    end
                end
                contour(X, Y, Z, 50, 'LineColor', [0.8 0.8 0.8]);
                
                % Generazione colori distinti
                num_plots = length(res_n2);
                line_colors = lines(num_plots); 
                
                for i = 1:num_plots
                    if strcmpi(res_n2(i).Method, "Modified")
                        line_style = '-'; 
                    else
                        line_style = '--'; 
                    end
                    
                    plot(res_n2(i).xseq(1,:), res_n2(i).xseq(2,:), line_style, ...
                        'Color', line_colors(i,:), 'Marker', 'o', 'MarkerSize', 3, ...
                        'LineWidth', 1.2, 'DisplayName', sprintf('%s (Pt %d)', res_n2(i).Method, res_n2(i).pt));
                end
                
                title(sprintf('Problem %d: Paths (n=2, %s)', prob_id, current_fd));
                xlabel('x_1'); ylabel('x_2'); grid on;
                legend('Location', 'northeastoutside', 'FontSize', 8);
            end

            %% --- 2. CONVERGENCE RATES (Tutte le n) ---
            unique_dims = unique([results([results.Prob] == prob_id).n]);
            for dim = unique_dims
                res_conv = results(all_fd_cases == current_fd & [results.Prob] == prob_id & [results.n] == dim);
                
                if ~isempty(res_conv)
                    figure('Name', sprintf('Prob %d - Conv (n=%d, %s)', prob_id, dim, current_fd), 'Color', 'w');
                    hold on;
                    
                    num_conv = length(res_conv);
                    conv_colors = lines(num_conv);
                    
                    for i = 1:num_conv
                        if strcmpi(res_conv(i).Method, "Modified")
                            line_style = '-';
                        else
                            line_style = '--';
                        end
                        
                        g_seq = res_conv(i).gradseq;
                        semilogy(0:length(g_seq)-1, g_seq, line_style, 'Color', conv_colors(i,:), ...
                            'LineWidth', 1.3, 'DisplayName', sprintf('%s (Pt %d)', res_conv(i).Method, res_conv(i).pt));
                    end
                    
                    yline(tolgrad, ':k', 'Tolerance', 'LabelHorizontalAlignment','left', 'LineWidth', 1.5);
                    title(sprintf('Problem %d: Conv Rate (n=%d, %s)', prob_id, dim, current_fd));
                    xlabel('Iteration (k)'); ylabel('||\nabla f(x_k)||');
                    legend('Location', 'northeastoutside', 'FontSize', 8);
                    grid on;
                end
            end
        end
    end
end

% --- FUNZIONE LOCALE PER IL CALCOLO DELLA F (DA NON DIMENTICARE!) ---
function f = get_f_val(prob_id, x)
    if prob_id == 31
        n = length(x);
        x_prev = [0; x(1:n-1)];
        x_next = [x(2:n); 0];
        % f_k(x) = (3-2x_k)x_k - x_{k-1} - 2x_{k+1} + 1
        r = (3 - 2*x).*x - x_prev - 2*x_next + 1;
        f = 0.5 * sum(r.^2); 
    else
        % Problema 49
        f1 = x(1) - 1;
        xi = x(1:end-1); xip1 = x(2:end);
        e = 10 * (xi.^2 - xip1);
        if length(x) > 2
            a = x(1:end-2) - x(2:end-1); b = x(2:end-1) - x(3:end);
            o = 2*exp(-(a.^2)) + exp(-2*(b.^2));
            f = f1^2 + sum(e.^2) + sum(o.^2);
        else
            f = f1^2 + sum(e.^2);
        end
    end
end