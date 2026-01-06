%% Test Script: Derivative-based Optimization - Problem 31 & 49
clear; clc;

% --- TEAM CONFIGURATION ---
student_ids = [123456, 234567, 345678]; 
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
k = 8; % you should check 4 and 12 also
use_relative = false; % you can set it to true if you want hi = 10^(-k)|xi|
% g_fd = @(x) fd_grad_from_obj_banded (f_handle, x, k, use_relative, bandwidth);
% g_handle = g_fd in case you want to approximate also the gradient 
% H_fd = @(x) fd_hessian_from_grad(g_handle, x, k, use_relative, bandwidth);
% Note that you should use H_fd only for the Modified Newton method because Truncated Newton 
% does not need the Hessian explicitly

% --- TEST SETTINGS ---
dimensions = [2, 1e3, 1e4, 1e5]; 
n_rand_points = 5;
results = [];

% --- PROBLEMS LIST ---
problem_ids = [31, 49]; 

for prob_id = problem_ids
    fprintf('\n###########################################################\n');
    fprintf(' OPTIMIZATION ANALYSIS: PROBLEM %d\n', prob_id);
    fprintf('###########################################################\n');
    
    % 1. Handle Configuration and specific Parameters
    if prob_id == 31
        f_h = @prob31_obj; g_h = @prob31_grad; h_h = @prob31_hess;
    else
        f_h = @prob49_obj; g_h = @prob49_grad; h_h = @prob49_hess;
    end
    
    for n = dimensions
        fprintf('\n==========================================\n');
        fprintf('TESTING DIMENSION n = %d\n', n);
        fprintf('==========================================\n');
        
        % 2. Suggested Starting Point definition
        if prob_id == 31
            x_suggested = -ones(n, 1);
        else
            x_suggested = ones(n, 1);
            x_suggested(1:2:end) = -1.2; % odd
            x_suggested(2:2:end) = 1.0;  % even
        end
        
        % Generation of 6 points (Suggested + 5 Random in the neighborhood)
        x_starts = [x_suggested, x_suggested + (2*rand(n, n_rand_points) - 1)];
        
        for s = 1:size(x_starts, 2)
            x0 = x_starts(:, s);
            fprintf('\n[Starting Point %d/%d]\n', s, size(x_starts, 2));
            
            % --- MODIFIED NEWTON TEST ---
            fprintf('  > Executing Modified Newton... ');
            tic;
            [~, ~, gnorm_m, k_m, ~, ~] = modified_newton_bcktrck(...
                x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax);
            time_m = toc;
            success_m = gnorm_m < tolgrad;
            fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', k_m, time_m, success_m);
            
            results = [results; struct('Prob', prob_id, 'n', n, 'pt', s, 'Method', 'Modified', ...
                'gnorm', gnorm_m, 'iters', k_m, 'time', time_m, 'success', success_m)];
                
            % --- TRUNCATED NEWTON TEST ---
            fprintf('  > Executing Truncated Newton... ');
            tic;
            [~, ~, gnorm_t, k_t, ~, ~, ~] = truncated_newton_bcktrck(...
                x0, f_h, g_h, h_h, kmax, tolgrad, c1, rho, btmax, fterms, cg_maxit);
            time_t = toc;
            success_t = gnorm_t < tolgrad;
            fprintf('Finished. Iterations: %d, Time: %.2fs, Success: %d\n', k_t, time_t, success_t);
            
            results = [results; struct('Prob', prob_id, 'n', n, 'pt', s, 'Method', 'Truncated', ...
                'gnorm', gnorm_t, 'iters', k_t, 'time', time_t, 'success', success_t)];
        end
    end
end

% Final table display
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