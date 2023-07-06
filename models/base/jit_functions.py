import numba
import numpy as np
import math

@numba.jit(nopython=True)
def _L1_clip(total_grads, reg_alpha):
    """
    L1 regularisation on the gradients, controlled by self.reg_alpha

    :param total_grads:
    :return:
    """
    if total_grads > reg_alpha:
        return total_grads - reg_alpha
    elif total_grads < -1 * reg_alpha:
        return total_grads + reg_alpha
    else:
        return 0


@numba.jit(nopython=True)
def _calculate_weight(total_grads, total_hess, reg_alpha, reg_delta, reg_lambda):
    """
    Calculates weight for leaf nodes

    :param total_grads: Total sum of gradients
    :param total_hess:  Total sum of hessians
    :return: Weight for leaf node
    """
    if total_hess < 0:
        total_hess = 0

    weight = -1 * (_L1_clip(total_grads, reg_alpha) / (total_hess + reg_lambda))
    if reg_delta != 0 and abs(weight) > reg_delta:
        return math.copysign(reg_delta, weight)  # Delta clipping
    else:
        return weight


@numba.jit(nopython=True)
def _calculate_gain(total_grads, total_hess, reg_alpha, reg_delta, reg_lambda):
    """
    Calculates gain from sum of gradients and sum of hessians

    :param total_grads: Sum of gradients
    :param total_hess: Sum of hessians
    :return: Gain score
    """
    con = _L1_clip(total_grads, reg_alpha)
    weight = -1 * (con / (total_hess + reg_lambda))
    if reg_delta != 0 and abs(weight) > reg_delta: # If delta-clipping is enabled the gain calculation is a little more complicated, following the implementation in the original XGBoost: https://github.com/dmlc/xgboost/blob/d7d1b6e3a6e2aa8fcb1857bf5e3188302a03b399/src/tree/param.h
        weight = math.copysign(reg_delta, weight)  # Delta clipping
        return -(2 * total_grads * weight + (total_hess + reg_lambda) * weight ** 2) + reg_alpha * abs(weight)  # This is an L1-regularised clipped gain calculation
    else:
        return -weight * con  # G^2/H + lambda, with possible L1 regularisation and delta clipping on G

# not faster than numpy
# @numba.jit(nopython=True)
# def _calculate_hyper_tune_score(left_sum_grad, left_sum_hessian, right_sum_grad, right_sum_hessian, smoothing_lambda):
#     left_num = left_sum_grad**2
#     right_num = right_sum_grad**2
        
#     left_denum = np.clip(left_sum_hessian + smoothing_lambda, a_min=smoothing_lambda, a_max=np.Inf)
#     right_denum = np.clip(right_sum_hessian + smoothing_lambda, a_min=smoothing_lambda, a_max=np.Inf)
        
#     score = left_num / left_denum + right_num / right_denum
        
#     return score
    