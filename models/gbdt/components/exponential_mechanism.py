import numpy as np

def expo_mech(scores, epsilon, sensitivity, seed=None):
    """
    exponential mechanism implemented as report-noisy-max with gumble noise
    ref: https://arxiv.org/abs/2105.07260
    """
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    beta = 2*sensitivity / epsilon
    rng = np.random.default_rng(seed)
    noise = rng.gumbel(0, beta, len(scores))
    idx = np.argmax(scores + noise)
    return idx


    
    
    
    
    


