function g = fd_grad_from_obj_banded(rfun, x, k, use_relative, bandwidth)
    r0 = rfun(x);
    J  = fd_jacobian_coloring(rfun, x, k, use_relative, bandwidth);
    g  = J' * r0;
end

