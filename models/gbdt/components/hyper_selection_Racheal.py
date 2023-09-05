from autodp.autodp_core import Mechanism, Transformer
from scipy.optimize import minimize_scalar

import math


def RDP_selection(params, alpha, dist):

    if alpha <= 1:
        alpha = 1.000000000000001
        
    if dist == 'TNB':
        return RDP_selection_TNB(params, alpha)
    elif dist == 'Poisson':
        return RDP_selection_poisson(params, alpha)
    else:
        raise Exception("Distribution type must be one of \
            'TNB' (truncated negative binomial) or 'Poisson'.")


def RDP_selection_TNB(params, alpha, alpha_hat_max=1000):
    
    eta, gamma = params['eta'], params['gamma']
    base_mech = params['base_mech']

    assert(eta > -1 and 0 < gamma < 1)

    if eta == 0:
        exp_K = (1/gamma - 1)/(math.log(1/gamma))
    else:
        exp_K = eta / gamma  * (1 - gamma) / (1 - gamma ** eta)

    log_exp_K = math.log(exp_K)
    
    func = lambda alpha_hat: (1 + eta) * (1 - 1/alpha_hat) * base_mech.get_RDP(alpha_hat) + (1 + eta) * math.log(1/gamma) / alpha_hat
    
    results = minimize_scalar(func, method='bounded', bounds = [1, alpha_hat_max])
    # result.fun is the optimized value
    
    #eps_prime = lambda x: base_mech.get_RDP(x) + results.fun + log_exp_K / (x - 1)
    eps_prime = base_mech.get_RDP(alpha) + results.fun + log_exp_K / (alpha-1)

    #return eps_prime(alpha)
    return eps_prime



def RDP_selection_poisson(params, alpha, alpha_hat_max=10):

    def search_delta_hat(base_mech, alpha, delta_max=1e-5, tol=10e-10):
        """
        Finds the smallest delta_hat for base_mech whose epsilon_hat does not exceed
        log(1 + 1 / (alpha - 1)), within a given tolerance.
        """
        eps_hat = math.log(1 + 1/(alpha - 1))
        eps_search = base_mech.approxDP(delta_max)
        while eps_search < eps_hat:
            delta_max = delta_max / 10
            eps_search = base_mech.approxDP(delta_max)
        delta_min, delta_max = delta_max, delta_max * 10
        while abs(eps_search - eps_hat) > tol or eps_search > eps_hat:
            delta_search = (delta_min + delta_max) / 2
            eps_search = base_mech.approxDP(delta_search)
            if eps_search > eps_hat:
                delta_min = delta_search
            else:
                delta_max = delta_search
        return delta_search

    mu = params['mu']
    base_mech = params['base_mech']
    delta_hat = search_delta_hat(base_mech, alpha)
    eps_prime = lambda x: base_mech.get_RDP(alpha) + mu * delta_hat + math.log(mu) / (alpha - 1)
    return eps_prime(alpha)


class PrivateSelection(Transformer):
    """ Given a base mechanism, transforms it into a mechanism that implements
        private selection as described in Papernot & Steinke's
        'Hyperparameter Tuning with Renyi Differential Privacy'.
    """
    def __init__(self):
        Transformer.__init__(self)
        self.transform = self.repeat
    
    def repeat(self, base_mech, dist, dist_params, alpha_hat_max=10e6):

        new_mech = Mechanism()
        new_mech.name = base_mech.name + dist + str(dist_params) # needed for composition, fine for transforming single mech, should check if holds for multiple
        new_mech.params = base_mech.params
        new_mech.params['base_mech'] = base_mech # pass into RDP_selection
        new_mech.params['dist'] = dist
        new_mech.params['eta'] = dist_params["eta"]
        new_mech.params['gamma'] = dist_params["gamma"]
        new_rdp = lambda x: RDP_selection(new_mech.params, x, dist)
        new_mech.propagate_updates(new_rdp, 'RDP')

        return new_mech