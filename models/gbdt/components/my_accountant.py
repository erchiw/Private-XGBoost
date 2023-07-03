import numpy as np

from autodp.mechanism_zoo import ExactGaussianMechanism, ExponentialMechanism, ComposedGaussianMechanism
from autodp.calibrator_zoo import ana_gaussian_calibrator, generalized_eps_delta_calibrator
from autodp.transformer_zoo import Composition
from scipy.optimize import bisect


class MyPrivacyAccountant():
    """
    This class is only used for grad_based
    """
    def __init__(self, loss_name, epsilon, delta, dp_method, # DP params
                 num_trees, num_features, max_depth, split_method, task_type="classification", sketch_type="uniform", sketch_rounds=np.Inf,# Tree params
                 ratio_hist=None, ratio_expo=None, ratio_leaf=None, # budget ratio for different mechanism
                 gau_sigma_leaf=None, gau_sigma_hist=None, gub_beta=None, # allow predefine noise level
                 ):

        # sanity check and bounds for CE loss and sigmoid least square
        if loss_name == "SigmoidBinaryCrossEntropyLoss" and task_type == "classification":
            self.max_gradient = 1
            self.min_gradient = -1
            self.max_hess = 1/4
            self.min_hess = 0
            self.loss_name = loss_name
        elif loss_name == "SigmoidLeastSquareLoss" and task_type == "classification":
            self.max_gradient = 1/4
            self.min_gradient = -1/4
            self.max_hess = 1/12
            self.min_hess = -1/4
            self.loss_name = loss_name
        else:
            raise NotImplementedError("For regression task, Use default privacy accountant instead")
        
        self.num_features = num_features
        self.dp_method = dp_method
        self.epsilon = epsilon
        self.calibrated_epsilon = None
        self.delta = delta
        self.is_calibrated = False

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.split_method = split_method

        if sketch_rounds == float("Inf"):
            self.sketch_rounds = num_trees
        else:
            self.sketch_rounds = sketch_rounds

       
        if (ratio_hist is not None) and (ratio_expo is not None) and (ratio_leaf is not None):
            """
            For adaptive hessian: release a tensor of hess hist each tree, # queries = sketch_round, but sens = sart(#feature) * sens(hess)
            For expo: # query = #tree * #depth
            For leaf: # query = #tree
            """
            self.ratio_per_hist = ratio_hist / self.sketch_rounds
            self.ratio_per_expo = ratio_expo / (self.num_trees * self.max_depth)
            self.ratio_per_leaf = ratio_leaf / self.num_trees
        else:
            raise ValueError("Should enter all ratio!")
        
        self.grad_sensitivity = max(self.max_gradient, abs(self.min_gradient))
        self.hess_sensitivity = max(self.max_hess, abs(self.min_hess))
        self.hess_hist_sensitivity = np.sqrt(self.num_features)*self.hess_sensitivity     
        self.count_sensitivity = 1 if split_method == "grad_based" else None
        self.leaf_sensitivity = np.sqrt(self.grad_sensitivity**2 + self.hess_sensitivity**2)

        self.loss_name = loss_name
        self.sketch_type = sketch_type

        # get noise multiplier
        if (gau_sigma_hist is not None) and (gau_sigma_leaf is not None) and (gub_beta is not None):
            self.sigma_hist = gau_sigma_hist
            self.sigma_leaf = gau_sigma_leaf
            self.beta = gub_beta
        else:
            self.sigma_hist = self.gaussian_var(mech_type="hist")
            self.sigma_leaf = self.gaussian_var(mech_type="leaf")
            self.beta = self.gumbel_beta()


    def create_tree_mechanism(self, gau_sigma_hist, expo_eps, gau_sigma_leaf):
        compose = Composition()

        expo_node = ExponentialMechanism(expo_eps, BR_off=False, name="expo_node")
        gau_leaf = ExactGaussianMechanism(gau_sigma_leaf, name="gau_leaf")
        tree_mech_no_proposal = compose([expo_node, gau_leaf], [self.max_depth, 1]) 

        if self.sketch_type == "adaptive_hessian":
            if gau_sigma_hist != 0:
                gau_hist = ExactGaussianMechanism(gau_sigma_hist, name="gau_hist")
            else:
                raise ValueError('budget on hess hist is 0')
            tree_mech_proposal = compose([gau_hist, expo_node, gau_leaf], [1, self.max_depth, 1]) # tree with adaptive hessian
        
            if self.sketch_rounds == self.num_trees:
                tree_mech = compose([tree_mech_proposal], [self.num_trees])
            else:
                tree_mech = compose([tree_mech_proposal, tree_mech_no_proposal], [self.sketch_rounds, self.num_trees-self.sketch_rounds])
        elif self.sketch_type == "uniform":
            tree_mech = compose([tree_mech_no_proposal], [self.num_trees])
        else:
            raise NotImplementedError("sketch type not implemented")

        return tree_mech
    
    # # TODO: fix re-calibration issue when given beta and sigma as non-None input
    # def update_noise_by_budget(self, new_eps, new_delta):
    #     self.epsilon = new_eps
    #     self.delta = new_delta
    #     self.sigma = self.gaussian_var()
    #     self.beta = self.gumbel_beta()
    
    @staticmethod
    def get_gau_sigma(eps, delta):
        ana_calibrate = ana_gaussian_calibrator()
        mech = ana_calibrate(ExactGaussianMechanism, eps, delta, name='Ana_GM')
        return mech.params['sigma']
    

    def get_eps_tree_mech(self, total_design_eps, delta):   
        eps_per_expo = total_design_eps * self.ratio_per_expo
        
        eps_per_leaf = total_design_eps * self.ratio_per_leaf
        sigma_leaf = self.get_gau_sigma(eps_per_leaf, delta)
        
        if self.sketch_type == "adaptive_hessian":
            eps_per_hist = total_design_eps * self.ratio_per_hist
            sigma_hist = self.get_gau_sigma(eps_per_hist, delta)

            tree_mech = self.create_tree_mechanism(sigma_hist, eps_per_expo, sigma_leaf)    
            actual_eps = tree_mech.get_approxDP(self.delta)
        elif self.sketch_type == "uniform":
            tree_mech = self.create_tree_mechanism(0, eps_per_expo, sigma_leaf)    
            actual_eps = tree_mech.get_approxDP(self.delta)

        return actual_eps
    

    def get_calibrated_epsilon(self, maxiter=1000):
       
        def f(x, delta, target_eps):
            # calculate the difference between x and target_eps
            return self.get_eps_tree_mech(x, delta) - target_eps

        start_pt = 0.8 * self.epsilon
        end_pt = 3 * self.epsilon

        # check if f(end_pt) has different sign compared to f(start_pt)
        while(True):
            if f(end_pt, self.delta, self.epsilon) <= 0:
                end_pt = end_pt + 1
            else:
                break
    
        # |eps - target_eps| < xtol + rtol*target_eps
        eps = bisect(f, 
                    a=start_pt, 
                    b=end_pt, 
                    args=(self.delta, self.epsilon),
                    xtol=1e-6, 
                    rtol=1e-10, 
                    maxiter=maxiter)
        
        self.calibrated_epsilon = eps


    def gumbel_beta(self):
        beta = None

        if self.calibrated_epsilon is None:
            self.get_calibrated_epsilon()
        
        if self.count_sensitivity is not None:
            beta = 2 / (self.calibrated_epsilon * self.ratio_per_expo)
        
        return beta


    def gaussian_var(self, mech_type):
        """
        return noise multiplier for gaussian mechanism
        """
        if self.calibrated_epsilon is None:
            self.get_calibrated_epsilon()
        
        if mech_type == "leaf":
            sigma = self.get_gau_sigma(self.calibrated_epsilon * self.ratio_per_leaf, self.delta)
        elif mech_type == "hist":
            sigma = self.get_gau_sigma(self.calibrated_epsilon * self.ratio_per_hist, self.delta)
            
        return sigma
    

        
    def _add_dp_noise(self, grad_sum, hess_sum, depth, feature=None, histogram_row=False, noise_size=None, num_obs=None, adaptive_hessian=False):        
        """
        Only For histogram release:
        Called at every node in the tree, returns perturbed (if using DP) sums of gradients/hessians

        :param grad_sum: List of gradients (first-derivative of the loss)
        :param hess_sum: List of hessians (second-derivative of the loss)
        :param depth: Current level in the tree
        :return: Perturbed sum of gradients and hessians
        """
        hist_gaussian_std = self.sigma_hist * self.hess_hist_sensitivity
        leaf_gaussian_std = self.sigma_leaf * np.sqrt(self.grad_sensitivity**2 + self.hess_sensitivity**2)

        perturbed_gradients = grad_sum
        perturbed_hessians = hess_sum

        if histogram_row and self.dp_method == "gaussian_cdp":
            #perturbed_gradients = grad_sum + np.random.normal(0, gaussian_std, size=noise_size)
            perturbed_gradients = None
            perturbed_hessians = hess_sum + np.random.normal(0, hist_gaussian_std, size=noise_size)
        elif self.dp_method == "gaussian_cdp":
            perturbed_gradients += np.random.normal(0, leaf_gaussian_std, size=1)[0]
            perturbed_hessians += np.random.normal(0, leaf_gaussian_std, size=1)[0]

        return perturbed_gradients, perturbed_hessians
    


    def _add_dp_noise_counts(self, grad_sum, hess_sum, depth, feature=None, histogram_row=False, noise_size=None, num_obs=None, adaptive_hessian=False):
        """
        Only For histogram release:
        Called at every node in the tree, returns perturbed (if using DP) sums of gradients/hessians

        :param grad_sum: List of gradients (first-derivative of the loss)
        :param hess_sum: List of hessians (second-derivative of the loss)
        :param depth: Current level in the tree
        :return: Perturbed sum of gradients and hessians
        """

        gaussian_std = self.sigma * np.sqrt(self.grad_sensitivity**2 + 1**2)

        perturbed_gradients = grad_sum
        perturbed_hessians = np.ones_like(hess_sum)
        
        if histogram_row and self.dp_method == "gaussian_cdp":
            perturbed_gradients = grad_sum + np.random.normal(0, gaussian_std, size=noise_size)
            perturbed_hessians = hess_sum + np.random.normal(0, gaussian_std, size=noise_size)
        elif self.dp_method == "gaussian_cdp":
            perturbed_gradients += np.random.normal(0, gaussian_std, size=1)[0]
            perturbed_hessians += np.random.normal(0, gaussian_std, size=1)[0]

        return perturbed_gradients, perturbed_hessians
    