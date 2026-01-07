%% Test Script: Derivative-based Optimization - Problem 31 & 49
clear; clc;

% --- TEAM CONFIGURATION ---
student_ids = [360765, 234567, 345678]; 
min_id = min(student_ids);
rng(min_id); % Set random seed for reproducibility

% --- METHOD PARAMETERS ---
kmax = 1000;
tolgrad = 1e-6; % Stopping criterion based on the gradient norm
c1 = 1e-4; 
rho = 0.5; 
btmax = 50;

% For Truncated Newton: forcing term for superlinear/quadratic convergence
fterms = @(k, gradfk) min(0.5, sqrt(norm(gradfk))); 
cg_maxit = 500;

% --- USING FINITE DIFFERENCES FOR THE DERIVATIVES --- 
bandwidth = 2;
k_list = [4,8,12]; % you should check 4 and 12 also
step_types = ["h", "hi"];
use_relative_list = [false true];  % false -> h, true -> hi; % you can set it to true if you want hi = 10^(-k)|xi|
% g_fd = @(x) fd_grad_from_obj_banded (f_handle, x, k, use_relative, bandwidth);
% g_handle = g_fd in case you want to approximate also the gradient 
% H_fd = @(x) fd_hessian_from_grad(g_handle, x, k, use_relative, bandwidth);
% Note that you should use H_fd only for the Modified Newton method because Truncated Newton 
% does not need the Hessian explicitly

% Choosing what to run:
RUN_POINT3_1 = false; % exact gradient with FD H (only for modified newton)
RUN_POINT3_2 = false; % FD g with FD H (MN and optionaly Truncated Newton)

% --- TEST SETTINGS ---
dimensions = [2, 1e3, 1e4, 1e5]; 
n_rand_points = 5;

% --- PROBLEMS LIST ---
problem_ids = [31, 49]; 
methods_point3_1 = "MN_FD_H"; % only MN uses Hessian explicitly
methods_point3_2 = ["MN_FD_gH", "TN_FD_gH"]; 

nStarts = 1 + n_rand_points;
nFD = numel(k_list) * numel(step_types);
nMethods = 0;

% checking the results for point 2: (Exact computations)
nRows = numel(problem_ids) * numel(dimensions) * (1 + n_rand_points) * 2;

if RUN_POINT3_1
    nMethods = nMethods + numel(methods_point3_1); 
    nRows = numel(problem_ids) * numel(dimensions) * nStarts * nFD * nMethods;
end

if RUN_POINT3_2
    nMethods = nMethods + numel(methods_point3_2); 
    nRows = numel(problem_ids) * numel(dimensions) * nStarts * nFD * nMethods;
end

results(nRows,1) = struct( ...
    'Prob', [], 'n', [], 'pt', [], 'FDcase', "", 'Method', "", ...
    'k', [], 'step', "", ...
    'gnorm', [], 'iters', [], 'time', [], 'success', false);

r = 0;

for prob_id = problem_ids
    fprintf('\n###########################################################\n');
    fprintf(' OPTIMIZATION ANALYSIS: PROBLEM %d\n', prob_id);
    fprintf('###########################################################\n');

    % --- PROBLEM HANDLES ---
    if prob_id == 31
        f_h = @prob31_obj; g_h = @prob31_grad; h_h = @prob31_hess;
    else
        f_h = @prob49_obj; g_h = @prob49_grad; h_h = @prob49_hess;
    end

    for n = dimensions
        fprintf('\n==========================================\n');
        fprintf('TESTING DIMENSION n = %d\n', n);
        fprintf('==========================================\n');

        % --- Suggested start (depends on n!) ---
        if prob_id == 31
            x_suggested = -ones(n, 1);
        else
            x_suggested = ones(n, 1);
            x_suggested(1:2:end) = -1.2;
            x_suggested(2:2:end) =  1.0;
        end

        % --- Starting points ---
        x_starts = [x_suggested, x_suggested + (2*rand(n, n_rand_points) - 1)];

        for s = 1:size(x_starts, 2)
            x0 = x_starts(:, s);
            fprintf('\n[Starting Point %d/%d]\n', s, size(x_starts, 2));

            % ============================================================
            % FD PART (Point 3.1 and/or 3.2)
            % ============================================================
            if (RUN_POINT3_1 || RUN_POINT3_2)

                for kk = 1:numel(k_list)
                    k_fd = k_list(kk);

                    for st = 1:numel(use_relative_list)
                        use_relative = use_relative_list(st);
                        step_name = step_types(st);
                        fd_label = sprintf("k=%d, step=%s", k_fd, step_name);

                        % -----------------------------
                        % POINT 3.1: exact gradient + FD Hessian  (MN only)
                        % -----------------------------
                        if RUN_POINT3_1                           
                            H_fd = @(x) fd_hessian_from_grad(g_h, x, k_fd, use_relative, bandwidth);

                            fprintf('  > MN (FD Hessian, %s)... ', fd_label);
                            tic;
                            [~, ~, gnorm_m, it_m, ~, ~] = modified_newton_bcktrck( ...
                                x0, f_h, g_h, H_fd, ...
                                kmax, tolgrad, c1, rho, btmax);
                            t_m = toc;

                            succ = gnorm_m < tolgrad;
                            fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', it_m, t_m, succ);

                            r = r + 1;
                            results(r) = struct('Prob', prob_id, 'n', n, 'pt', s, ...
                                'FDcase', "Point3.1", 'Method', "Modified", ...
                                'k', k_fd, 'step', step_name, ...
                                'gnorm', gnorm_m, 'iters', it_m, 'time', t_m, 'success', succ);
                        end

                        % -----------------------------
                        % POINT 3.2 : FD gradient + FD Hessian                   
                        % -----------------------------
                        if RUN_POINT3_2
                            g_fd = @(x) fd_grad_from_obj_banded(f_h, x, k_fd, use_relative, bandwidth);
                            H_fd = @(x) fd_hessian_from_grad(g_fd, x, k_fd, use_relative, bandwidth);

                            % MN with FD g and FD H
                            fprintf('  > MN (FD g & H, %s)... ', fd_label);
                            tic;
                            [~, ~, gnorm_m, it_m, ~, ~] = modified_newton_bcktrck( ...
                                x0, f_h, g_fd, H_fd, ...
                                kmax, tolgrad, c1, rho, btmax);
                            t_m = toc;

                            succ = gnorm_m < tolgrad;
                            fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', it_m, t_m, succ);

                            r = r + 1;
                            results(r) = struct('Prob', prob_id, 'n', n, 'pt', s, ...
                                'FDcase', "Point3.2", 'Method', "Modified", ...
                                'k', k_fd, 'step', step_name, ...
                                'gnorm', gnorm_m, 'iters', it_m, 'time', t_m, 'success', succ);

                            % TN with FD g & FD H 
                            fprintf('  > TN (FD g & H, %s)... ', fd_label);
                            tic;
                            [~, ~, gnorm_t, it_t, ~, ~, ~] = truncated_newton_bcktrck( ...
                                x0, f_h, g_fd, H_fd, ...
                                kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
                            t_t = toc;

                            succ = gnorm_t < tolgrad;
                            fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', it_t, t_t, succ);

                            r = r + 1;
                            results(r) = struct('Prob', prob_id, 'n', n, 'pt', s, ...
                                'FDcase', "Point3.2", 'Method', "Truncated", ...
                                'k', k_fd, 'step', step_name, ...
                                'gnorm', gnorm_t, 'iters', it_t, 'time', t_t, 'success', succ);
                        end
                    end
                end

            % ============================================================
            % EXACT PART (Point 2): run MN + TN with exact derivatives
            % ============================================================
            else
                % Exact MN
                fprintf(' > Executing Modified Newton... ');
                tic;
                [~, ~, gnorm_m, k_m, ~, ~] = modified_newton_bcktrck( ...
                    x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax);
                time_m = toc;

                success_m = gnorm_m < tolgrad;
                fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', k_m, time_m, success_m);

                r = r + 1;
                results(r) = struct('Prob', prob_id, 'n', n, 'pt', s, ...
                    'FDcase', "Exact", 'Method', "Modified", ...
                    'k', NaN, 'step', "", ...
                    'gnorm', gnorm_m, 'iters', k_m, 'time', time_m, 'success', success_m);

                % Exact TN
                fprintf('  > Executing Truncated Newton... ');
                tic;
                [~, ~, gnorm_t, k_t, ~, ~, ~] = truncated_newton_bcktrck( ...
                    x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
                time_t = toc;

                success_t = gnorm_t < tolgrad;
                fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', k_t, time_t, success_t);

                r = r + 1;
                results(r) = struct('Prob', prob_id, 'n', n, 'pt', s, ...
                    'FDcase', "Exact", 'Method', "Truncated", ...
                    'k', NaN, 'step', "", ...
                    'gnorm', gnorm_t, 'iters', k_t, 'time', time_t, 'success', success_t);
            end
        end
    end
end


% Final table display
results = results(1:r);
disp(struct2table(results));

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