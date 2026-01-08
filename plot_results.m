function plot_results(results, problem_ids, dimensions, tolgrad)
    % plot_results: Genera i grafici mandatori per l'Assignment 1
    
    % Forziamo FDcase a essere un array di stringhe per evitare errori di tipo
    all_fd_cases = string({results.FDcase}); 
    fd_cases = unique(all_fd_cases); 

    for f = 1:length(fd_cases)
        current_fd = fd_cases(f);
        
        for prob_id = problem_ids
            %% --- 1. SEQUENCE PATHS (n=2) ---
            % Filtriamo usando il confronto tra stringhe
            res_n2 = results(all_fd_cases == current_fd & [results.Prob] == prob_id & [results.n] == 2);
            
            if ~isempty(res_n2)
                figure('Name', sprintf('Prob %d - Paths (%s)', prob_id, current_fd));
                hold on;
                
                % Setup area del plot basato sui punti xseq
                all_pts = [];
                for i = 1:length(res_n2)
                    all_pts = [all_pts, res_n2(i).xseq];
                end
                
                margin = 0.5;
                x_range = [min(all_pts(1,:))-margin, max(all_pts(1,:))+margin];
                y_range = [min(all_pts(2,:))-margin, max(all_pts(2,:))+margin];
                
                % Curve di livello
                [X, Y] = meshgrid(linspace(x_range(1), x_range(2), 100), ...
                                  linspace(y_range(1), y_range(2), 100));
                Z = zeros(size(X));
                for i = 1:size(X,1)
                    for j = 1:size(X,2)
                        Z(i,j) = get_f_val(prob_id, [X(i,j); Y(i,j)]);
                    end
                end
                contour(X, Y, Z, 50, 'LineColor', [0.7 0.7 0.7]);
                
                % Percorsi (Paths) [cite: 89, 114]
                colors = lines(length(res_n2));
                for i = 1:length(res_n2)
                    plot(res_n2(i).xseq(1,:), res_n2(i).xseq(2,:), '-o', ...
                        'Color', colors(i,:), 'MarkerSize', 3, 'LineWidth', 1, ...
                        'DisplayName', sprintf('%s (Pt %d)', res_n2(i).Method, res_n2(i).pt));
                end
                
                title(sprintf('Problem %d: Paths (n=2, %s)', prob_id, current_fd));
                xlabel('x_1'); ylabel('x_2'); legend('Location', 'northeastoutside'); grid on;
            end

            %% --- 2. CONVERGENCE RATES ---
            for dim = dimensions
                res_conv = results(all_fd_cases == current_fd & [results.Prob] == prob_id & [results.n] == dim);
                
                if ~isempty(res_conv)
                    figure('Name', sprintf('Prob %d - Conv (n=%d, %s)', prob_id, dim, current_fd));
                    hold on;
                    for i = 1:length(res_conv)
                        g_seq = res_conv(i).gradseq;
                        semilogy(0:length(g_seq)-1, g_seq, '-s', 'LineWidth', 1.2, ...
                            'DisplayName', sprintf('%s (Pt %d)', res_conv(i).Method, res_conv(i).pt));
                    end
                    yline(tolgrad, '--r', 'Tolerance');
                    title(sprintf('Problem %d: Conv Rate (n=%d, %s)', prob_id, dim, current_fd));
                    xlabel('Iteration (k)'); ylabel('||\nabla f(x_k)||');
                    legend('Location', 'best'); grid on;
                end
            end
        end
    end
end

function f = get_f_val(prob_id, x)
    if prob_id == 31
        % f_k(x) = (3-2x_k)x_k - x_{k-1} - 2x_{k+1} + 1
        n = length(x);
        x_prev = [0; x(1:n-1)];
        x_next = [x(2:n); 0];
        r = (3 - 2*x).*x - x_prev - 2*x_next + 1;
        f = sum(r.^2);
    else
        % Calcolo per Problema 49
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