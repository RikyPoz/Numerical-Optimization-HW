%% Test Script: Derivative-based Optimization - Problem 31 & 49
clear; clc;

% --- TEAM CONFIGURATION ---
student_ids = [360765, 361352, 359620]; 
min_id = min(student_ids);
rng(min_id); 

% --- METHOD PARAMETERS ---
kmax = 200;
tolgrad = 1e-5; 
c1 = 1e-4; 
rho = 0.5; 
btmax = 20;
fterms = @(k, gradfk) min(0.5, sqrt(norm(gradfk))); % Forcing term for superlinear conv in truncated
cg_maxit = 50;

% --- FINITE DIFFERENCES SETTINGS --- 
k_list = [4, 8, 12]; 
step_types = ["h", "hi"]; 
use_relative_list = [false, true]; % false -> h costante, true -> hi adattivo 

% --- CHOOSING WHAT TO RUN ---
RUN_EXACT    = true; % Point 2: Exact derivatives 
RUN_POINT3_1 = true; % Point 3.1: Exact Gradient, Hessian FD 
RUN_POINT3_2 = true; % Point 3.2: Grandient and Hessian FD 

% --- TEST SETTINGS ---
dimensions = [2,1000,1e4,1e5]; 
n_rand_points = 5; 
problem_ids = [31,49]; 
nStarts = 1 + n_rand_points;
nFD = numel(k_list) * numel(use_relative_list);

% --- PRE-ALLOCATION ---
totalRows = 0;
if RUN_EXACT
    totalRows = totalRows + numel(problem_ids) * numel(dimensions) * nStarts * 2;
end
if RUN_POINT3_1
    totalRows = totalRows + numel(problem_ids) * numel(dimensions) * nStarts * nFD * 2;
end
if RUN_POINT3_2
    totalRows = totalRows + numel(problem_ids) * numel(dimensions) * nStarts * nFD * 2;
end
results(totalRows, 1) = struct('Prob', [], 'n', [], 'pt', [], 'FDcase', "", 'Method', "", ...
    'k', [], 'step', "", 'gnorm', [], 'iters', [], 'time', [], 'success', false, ...
    'xseq', [], 'gradseq', [], 'exp_rate', []); 

r = 0;
for prob_id = problem_ids
    fprintf('\n###########################################################\n');
    fprintf(' OPTIMIZATION ANALYSIS: PROBLEM %d\n', prob_id);
    fprintf('###########################################################\n');
    
    if prob_id == 31
        f_h = @prob31_obj; g_h = @prob31_grad; h_h = @prob31_hess;
        bw_res = 1;
        bw_hess = 2;
    else
        f_h = @prob49_obj; g_h = @prob49_grad; h_h = @prob49_hess;
        bw_res = 2;
        bw_hess = 4;
    end

    for n = dimensions
        fprintf('\n--- Testing Dimension n = %d ---\n', n);
        
        % --- Suggested start  ---
        if prob_id == 31
            x_suggested = -ones(n, 1);
        else
            x_suggested = ones(n, 1);
            x_suggested(1:2:end) = -1.2;
            x_suggested(2:2:end) =  1.0;
        end

        % --- 6 Starting points ---
        x_starts = [x_suggested, x_suggested + (2*rand(n, n_rand_points) - 1)];

        for s = 1:nStarts
            x0 = x_starts(:, s);
            fprintf(' [Point %d/%d]\n', s, nStarts);
            
            % --- 1. EXACT PART (Point 2) ---
            if RUN_EXACT
                fprintf('  > MN (Exact)... ');
                tic; 
                [~, ~, gnorm_m, k_m, xseq_m, gradseq_m, ~] = modified_newton_bcktrck(x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax);
                t_m = toc; 
                fprintf('Done. Iters: %d, Time: %.2fs\n', k_m, t_m);
                r = r + 1; 
                results(r) = fill_struct(prob_id, n, s, "Exact", "Modified", NaN, "", gnorm_m, k_m, t_m, xseq_m, gradseq_m, tolgrad);
                
                fprintf('  > TN (Exact)... ');
                tic; 
                [~, ~, gnorm_t, k_t, xseq_t, gradseq_t, ~, ~] = truncated_newton_bcktrck(x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
                t_t = toc; 
                fprintf('Done. Iters: %d, Time: %.2fs\n', k_t, t_t);
                r = r + 1; 
                results(r) = fill_struct(prob_id, n, s, "Exact", "Truncated", NaN, "", gnorm_t, k_t, t_t, xseq_t, gradseq_t, tolgrad);
            end

            % --- 2. POINT 3.1 & 3.2 (Finite Differences) ---
            if RUN_POINT3_1 || RUN_POINT3_2
                for kk = 1:numel(k_list)
                    for st = 1:numel(use_relative_list)
                        k_fd = k_list(kk); 
                        use_rel = use_relative_list(st); 
                        step_name = step_types(st);
                        fd_label = sprintf("k=%d, %s", k_fd, step_name);
                        
                        
                        if RUN_POINT3_1

                            if prob_id == 31
                                H_fd = @(x) fd_hess31_vett(x, k_fd, use_rel);
                            elseif prob_id == 49
                                H_fd = @(x) fd_hess49_vett(x, k_fd, use_rel);
                            end
                             

                            fprintf('  > MN (3.1, %s)... ', fd_label);
                            tic; 
                            [~, ~, gnorm_m, it_m, xseq_m, gradseq_m, ~] = modified_newton_bcktrck(x0, f_h, g_h, H_fd, kmax, tolgrad, c1, rho, btmax);
                            t_m = toc; 
                            fprintf('Done. Iters: %d, Time: %.2fs\n', it_m, t_m);
                            r = r + 1; 
                            results(r) = fill_struct(prob_id, n, s, "Point3.1", "Modified", k_fd, step_name, gnorm_m, it_m, t_m, xseq_m, gradseq_m, tolgrad);

                            fprintf('  > TN (3.1, %s)... ', fd_label);
                            tic; 
                            [~, ~, gnorm_t, it_t, xseq_t, gradseq_t, ~, ~] = truncated_newton_bcktrck(x0, f_h, g_h, H_fd, kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
                            t_t = toc; 
                            fprintf('Done. Iters: %d, Time: %.2fs\n', it_t, t_t);
                            r = r + 1; 
                            results(r) = fill_struct(prob_id, n, s, "Point3.1", "Truncated", k_fd, step_name, gnorm_t, it_t, t_t, xseq_t, gradseq_t, tolgrad);
                        end
                        
                        if RUN_POINT3_2
                            if prob_id == 31
                                g_fd = @(x) fd_grad31_vett(x, k_fd, use_rel);
                                H_fd = @(x) fd_hess31_vett(x, k_fd, use_rel);
                            elseif prob_id == 49
                                g_fd = @(x) fd_grad49_vett(x, k_fd, use_rel);
                                H_fd = @(x) fd_hess49_vett(x, k_fd, use_rel);
                            end
                           
                            
                            fprintf('  > MN (3.2, %s)... ', fd_label);
                            tic; 
                            [~, ~, gnorm_m, it_m, xseq_m, gradseq_m, ~] = modified_newton_bcktrck(x0, f_h, g_fd, H_fd, kmax, tolgrad, c1, rho, btmax);
                            t_m = toc; 
                            fprintf('Done. Iters: %d, Time: %.2fs\n', it_m, t_m);
                            r = r + 1; 
                            results(r) = fill_struct(prob_id, n, s, "Point3.2", "Modified", k_fd, step_name, gnorm_m, it_m, t_m, xseq_m, gradseq_m, tolgrad);

                            fprintf('  > TN (3.2, %s)... ', fd_label);
                            tic; 
                            [~, ~, gnorm_t, it_t, xseq_t, gradseq_t, ~, ~] = truncated_newton_bcktrck(x0, f_h, g_fd, H_fd, kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
                            t_t = toc; 
                            fprintf('Done. Iters: %d, Time: %.2fs\n', it_t, t_t);
                            r = r + 1; 
                            results(r) = fill_struct(prob_id, n, s, "Point3.2", "Truncated", k_fd, step_name, gnorm_t, it_t, t_t, xseq_t, gradseq_t, tolgrad);
                        end
                    end
                end
            end
        end
    end
end


%% Helper for filling the results struct
function s = fill_struct(prob, n, pt, case_name, method, k, step, gnorm, iters, time, xseq, gradseq, tol)
    s.Prob = prob; s.n = n; s.pt = pt; s.FDcase = case_name; s.Method = method;
    s.k = k; s.step = step; s.gnorm = gnorm; s.iters = iters; s.time = time;
    s.success = gnorm < tol; s.xseq = xseq; s.gradseq = gradseq;
    s.exp_rate = convergence_rate(gradseq);
end


% Converts results in a table
T_full = struct2table(results);

cols_to_show = {'Prob', 'n', 'pt', 'FDcase', 'Method', 'k', 'gnorm', 'iters', 'time', 'success', 'exp_rate'};
T_report = T_full(:, cols_to_show);

% --- EXACT AVG table ---
T_exact = T_report(T_report.FDcase == "Exact" & T_report.success == true, :);
avg_exact = groupsummary(T_exact, {'Prob', 'n', 'Method'}, 'mean', {'gnorm', 'iters', 'time', 'exp_rate'});
disp('Avg Scores For Exact Methods:');
disp(avg_exact);

% --- FD AVG table ---
T_fd = T_report(T_report.FDcase ~= "Exact" & T_report.success == true, :);
avg_fd = groupsummary(T_fd, {'Prob', 'n', 'Method', 'k'}, 'mean', {'gnorm', 'iters', 'time', 'exp_rate'});
disp('Avg Scores For FD:');
disp(avg_fd);

disp('Full Table');
disp(T_report);

plot_results(results, problem_ids, dimensions, tolgrad);





%% --- LOCAL FUNCTIONS ---

% (Problem 31)
function f = prob31_obj(x)
    [f, ~, ~] = prob31_analitico(x);
end
function g = prob31_grad(x)
    [~, g, ~] = prob31_analitico(x);
end
function H = prob31_hess(x)
    [~, ~, H] = prob31_analitico(x);
end

% (Problem 49)
function f = prob49_obj(x)
    [f, ~, ~] = prob49_analytical(x);
end
function g = prob49_grad(x)
    [~, g, ~] = prob49_analytical(x);
end
function H = prob49_hess(x)
    [~, ~, H] = prob49_analytical(x);
end



function p = estimate_rate(gradseq)
   p = convergence_rate(gradseq);
end



