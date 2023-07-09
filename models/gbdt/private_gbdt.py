import math
import random

import pandas as pd
import numpy as np
from scipy.optimize import bisect

from copy import copy

from fast_histogram import histogram1d

from federated_gbdt.models.base.tree_base import TreeBase
from federated_gbdt.models.base.tree_node import DecisionNode
from federated_gbdt.models.base.jit_functions import _calculate_gain, _calculate_weight, _L1_clip

from federated_gbdt.models.gbdt.components.split_candidate_manager import SplitCandidateManager
from federated_gbdt.models.gbdt.components.index_sampler import IndexSampler
from federated_gbdt.models.gbdt.components.privacy_accountant import PrivacyAccountant
from federated_gbdt.models.gbdt.components.my_accountant import MyPrivacyAccountant
from federated_gbdt.models.gbdt.components.train_monitor import TrainMonitor

from federated_gbdt.core.pure_ldp.frequency_oracles.hybrid_mechanism.hybrid_mech_client import HMClient
from federated_gbdt.core.loss_functions import SigmoidBinaryCrossEntropyLoss, BinaryRFLoss, SoftmaxCrossEntropyLoss, SigmoidLeastSquareLoss

from sklearn.preprocessing import LabelBinarizer


class PrivateGBDT(TreeBase):

    def __init__(self, num_trees=2, max_depth=6, # Default tree params
                 num_features=None,
                 task_type="classification", loss=SigmoidBinaryCrossEntropyLoss(),
                 reg_lambda=1, reg_alpha=0, reg_gamma=1e-7, reg_eta=0.3, reg_delta=2, reg_delta_grad=0, smoothing_lambda=1,# Regularisation params
                 min_samples_split=2, min_child_weight=0,  # Regularisation params
                 subsample=1, row_sample_method=None, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,  # Sampling params
                 sketch_type="uniform", sketch_eps=0.001, sketch_rounds=float("inf"), bin_type="all", range_multiplier=1, hist_bin=32, categorical_map=None,  # Sketch params
                 dp_method="", accounting_method="rdp_scaled_improved", epsilon=1, quantile_epsilon=0, gradient_clipping=False, # DP params
                 ratio_hist=None, ratio_selection=None, ratio_leaf=None, # DP params
                 tree_budgets=None, level_budgets=None, gradient_budgets=None, # DP params
                 ignore_split_constraints=False, grad_clip_const=None, # DP params
                 split_method="hist_based", weight_update_method="xgboost", training_method="boosting", batched_update_size=1, # training method params
                 feature_interaction_method="", feature_interaction_k=None, full_ebm=False, # feature interaction params
                 early_stopping=None, es_metric=None, es_threshold=-5, es_window=3, # early stopping
                 track_budget=True, split_method_per_level=None, hist_estimator_method=None, sigma=None, verbose=False, output_train_monitor=False, # macro parms
                 seed_random=None, seed_numpy=None, # reproducility
                 selection_mechanism = "exponential_mech", hyper_tune_coverage=0.9,
                 ):

        super(PrivateGBDT, self).__init__(min_samples_split=min_samples_split, max_depth=max_depth, task_type=task_type)
        self.output_train_monitor = output_train_monitor

        # Training type
        self.training_method = training_method
        self.loss = loss
        if self.training_method == "rf":
            self.loss = BinaryRFLoss()

        self.batched_update_size = batched_update_size
        self.weight_update_method = weight_update_method # xgboost vs gbm updates
        self.split_method = split_method # Determines how splits are chosen - hist_based, partially_random, totally_random, hybrid_random
        self.split_method_per_level = split_method_per_level
        self.selection_mechanism = selection_mechanism

        self.feature_interaction_method = feature_interaction_method
        self.feature_interaction_k = feature_interaction_k

        self.full_ebm = full_ebm

        if self.split_method in ["hist_based", "partially_random", "totally_random", "node_based", "grad_based", "hyper_tune"]:
            self.split_method_per_level = [self.split_method]*self.max_depth
        else: 
            raise ValueError("split_method invalid")

        if self.split_method == "hybrid_random" and self.split_method_per_level is None:
                self.split_method_per_level = ["totally_random"] * self.max_depth # By default just do totally random
        
        if self.split_method == "grad_based" and selection_mechanism not in ["exponential_mech", "permutate_flip", "public_cheat"]:
            raise ValueError("selection mech is invalid")
        
        if self.split_method == "hyper_tune":
            if self.feature_interaction_method == "cyclical" and self.feature_interaction_k == 1:
                expected_run_num = np.floor(hist_bin * hyper_tune_coverage)
            else:
                expected_run_num = np.floor(num_features * hist_bin * hyper_tune_coverage)
            
            p = self._calibrate_hyper_p(expected_run_num)
            if p[1].converged:
                self.hyper_p = p[1].root # p for numpy.random.logseries
            else:
                raise ValueError("p is not converged")
        else:
            self.hyper_p = None
        
        self.hist_estimator_method = hist_estimator_method # one_sided, two_sided, two_sided_averaging

        self.epsilon = epsilon        
        self.sketch_type = sketch_type
        
        # Base Parameters
        self.num_trees = num_trees
        self.feature_list = None
        if num_features is None:
            raise ValueError('Should input a valid number of features')
        else:
            self.num_features = num_features
        
        self.X = None
        self.ignore_split_constraints = ignore_split_constraints
        self.feature_bin = []
        self.gradient_histogram, self.hessian_histogram, self.root_hessian_histogram = [], [], []


        # Tracking vars
        self.train_monitor = TrainMonitor(0)

        # Regularisation Parameters
        self.reg_lambda = reg_lambda  # L2 regularisation on weights
        self.reg_alpha = reg_alpha  # L1 regularisation on gradients
        self.reg_gamma = reg_gamma  # Equivalent to the min impurity score needed to split a node further (or just leave it as a leaf)
        self.reg_delta_grad = reg_delta_grad
        self.smoothing_lambda = smoothing_lambda # for denominator smoothing of renyi hyper score releasing

        self.min_child_weight = min_child_weight  # Minimum sum of instance weight (hessian) needed in a child, if the sum of hessians less than this then the node is not split further
        self.min_samples_split = min_samples_split

        self.reg_eta = reg_eta  # Learning rate - multiplicative factor on weights
        self.reg_delta = reg_delta  # Clipping on the weights -> Useful in imablanced scenarios where it's possible for the hess to be 0 and thus the weights arbitraily large


        # Random Sampling Parameters
        self.index_sampler = IndexSampler(subsample, row_sample_method, colsample_bytree, colsample_bylevel, colsample_bynode)

        # Binning / Sketching Parameters for Feature Splits
        self.split_candidate_manager = SplitCandidateManager(hist_bin, self.num_trees, quantile_epsilon,
                                                             sketch_type, sketch_rounds, categorical_map,
                                                             sketch_eps, bin_type, range_multiplier)

        # Privacy (DP) Parameters
        self.dp_method = dp_method
        self.track_budget = track_budget
        self.verbose = verbose

        # budget ratio / noise level, only for my account
        self.ratio_hist = ratio_hist
        self.ratio_selection = ratio_selection
        self.ratio_leaf = ratio_leaf
        
        # print(self.ratio_hist, self.ratio_selection, self.ratio_leaf)
        # The delta value of 1e-5 is a placeholder that is updated to 1/n when the dataset is being trained
        if split_method in ["grad_based", "hyper_tune"]:
            if isinstance(self.loss, SigmoidBinaryCrossEntropyLoss):
                loss_name = "SigmoidBinaryCrossEntropyLoss"
            elif isinstance(self.loss, SigmoidLeastSquareLoss):
                loss_name = "SigmoidLeastSquareLoss"
            else: 
                raise NotImplementedError("loss not implemented for grad_based")
     
            self.loss_name = loss_name
            self.privacy_accountant = MyPrivacyAccountant(loss_name=loss_name, epsilon=epsilon, delta=1e-5, dp_method=dp_method, 
                                                          num_trees=self.num_trees, num_features=self.num_features, max_depth=self.max_depth,
                                                          split_method=self.split_method, task_type=self.task_type, 
                                                          sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                          ratio_hist=self.ratio_hist, ratio_leaf=self.ratio_leaf, ratio_selection=self.ratio_selection, 
                                                          selection_mechanism=self.selection_mechanism,
                                                          feature_interaction_method=self.feature_interaction_method, 
                                                          feature_interaction_k=self.feature_interaction_k)
            
        else:
            self.privacy_accountant = PrivacyAccountant(accounting_method, epsilon, 1e-5, quantile_epsilon, dp_method,
                                                        self.num_trees, self.max_depth, self.split_method, self.training_method, self.weight_update_method,
                                                        split_method_per_level=self.split_method_per_level,
                                                        tree_budgets=tree_budgets, gradient_budgets=gradient_budgets, level_budgets=level_budgets,
                                                        feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                        sample_method=self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                        sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                        task_type=self.task_type, sigma=sigma, grad_clip_const=grad_clip_const, gradient_clipping=gradient_clipping,
                                                        verbose=self.verbose)

        # Early stopping (not used)
        self.early_stopping = early_stopping
        self.es_metric = "root_hess" if es_metric is None else es_metric
        self.es_window = es_window
        self.es_threshold = es_threshold

        # for seed
        self.seed_random = seed_random
        self.seed_numpy = seed_numpy
    

    @staticmethod
    def _calibrate_hyper_p(target):
        def f(p):
            return -p/(1-p)/np.log(1-p) - target
        
        root = bisect(f, 1e-16, 1-1e-16, full_output=True)
        return root
        

    def _reset_accountant(self):
        self.privacy_accountant = PrivacyAccountant(self.privacy_accountant.accounting_method, self.privacy_accountant.epsilon, 1e-5,
                                                    self.privacy_accountant.quantile_epsilon, self.dp_method,
                                                    self.num_trees, self.max_depth, self.split_method, self.training_method, self.weight_update_method,
                                                    split_method_per_level=self.split_method_per_level,
                                                    tree_budgets=self.privacy_accountant.tree_budgets, gradient_budgets=self.privacy_accountant.gradient_budgets, level_budgets=self.privacy_accountant.level_budgets,
                                                    feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                    sample_method = self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                    sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                    task_type=self.task_type, sigma=self.privacy_accountant.sigma,
                                                    grad_clip_const=self.privacy_accountant.grad_clip_const, gradient_clipping=self.privacy_accountant.gradient_clipping,
                                                    verbose=self.verbose,)

    def _reset_tracking_attributes(self, checkpoint):
        self.X, self.y = None, None

        # These dont need to be removed but save space...
        if not checkpoint:
            self.train_monitor.current_tree_weights = []
            self.train_monitor.previous_tree_weights = []
            self.train_monitor.y_weights = []
            self.train_monitor.leaf_gradient_tracker = [[], []]
            self.train_monitor.root_gradient_tracker = [[], []]
            self.train_monitor.gradient_info = []

            self.privacy_accountant = None
            self.gradient_histogram = []
            self.feature_bin = []

    # Gradient/Hessian Calculations
    # ---------------------------------------------------------------------------------------------------

    def _compute_grad_hessian_with_samples(self, y, y_pred):
        """
        Called at the start of every tree, computes gradients and hessians for every observation from the previous predictions of the ensemble

        If using a LDP method, the perturbation is done here and the tree is formed as a post-processing step on the LDP perturbed gradients

        Otherwise, the raw gradients are passed to the model from fit() to  _build_tree() and they are perturbed later on in _add_dp_noise()

        :param y: True labels
        :param y_pred: Predicted labels
        :return: List of gradients and hessians
        """
        if self.task_type == 'classification' or self.task_type == "regression":
            grads = self.loss.compute_grad(y, y_pred)

            if self.task_type == "regression":
                grads = np.clip(grads, self.privacy_accountant.min_gradient, self.privacy_accountant.max_gradient)

            if self.weight_update_method == "xgboost":
                hess = self.loss.compute_hess(y, y_pred)
            else:
                hess = np.ones(len(y))

            if self.dp_method == "mean_mech_ldp":
                # Use mean mechanism perturbation
                hess_hm_client = HMClient(self.privacy_accountant.tree_budgets[len(self.trees)]*self.epsilon, self.privacy_accountant.max_hess, self.privacy_accountant.min_hess)  # Hess perturber
                grad_hm_client = HMClient(self.privacy_accountant.tree_budgets[len(self.trees)]*self.epsilon, self.privacy_accountant.max_gradient, self.privacy_accountant.min_gradient)  # Grad perturber
                grads = np.array([grad_hm_client.privatise(g) for g in grads])
                hess = np.array([hess_hm_client.privatise(h) for h in hess])
            elif self.dp_method == "gaussian_ldp":
                # Gaussian LDP
                grad_sigma = self.privacy_accountant.gaussian_var(gradient_type="gradient", depth=self.max_depth-1)
                hess_sigma = self.privacy_accountant.gaussian_var(gradient_type="hessian", depth=self.max_depth-1)
                if self.split_method == "hist_based":
                    grad_sigma /= math.sqrt(self.num_features * self.max_depth)
                    hess_sigma /= math.sqrt(self.num_features * self.max_depth)
                gradient_noise = np.random.normal(0, grad_sigma, size=(len(grads)))
                hess_noise = np.random.normal(0, grad_sigma, size=(len(hess)))

                grads = grads + gradient_noise
                hess = hess + hess_noise
        else:
            raise TypeError('%s task is not included in our XGboost algorithm !' % self.task_type)
        return grads, hess

    # Following methods assume that the total grads/hess that are passed have already been perturbed under some DP scheme
    def _L1_clip(self, total_grads):
        """
        L1 regularisation on the gradients, controlled by self.reg_alpha

        :param total_grads:
        :return:
        """
        return _L1_clip(total_grads, self.reg_alpha)
        
    def _calculate_weight(self, total_grads, total_hess):
        """
        Calculates weight for leaf nodes

        :param total_grads: Total sum of gradients
        :param total_hess:  Total sum of hessians
        :return: Weight for leaf node
        """
        # if total_hess < self.min_hess:
        #     total_hess = 0
        return _calculate_weight(total_grads, total_hess, self.reg_alpha, self.reg_delta, self.reg_lambda)
    

    def _calculate_gain(self, total_grads, total_hess):

        """
        Calculates gain from sum of gradients and sum of hessians

        :param total_grads: Sum of gradients
        :param total_hess: Sum of hessians
        :return: Gain score
        """
        return _calculate_gain(total_grads, total_hess, self.reg_alpha, self.reg_delta, self.reg_lambda)
    
    
    def _calculate_gain_grad_based(self, total_grads, total_hess):
        """
        Calculate gain for releasing splitting score for grad_based 
        """
        score = total_grads**2 / (total_hess + self.smoothing_lambda) 
        return score

      
      
    def _calculate_split_score(self, left_gain, right_gain, total_gain):
        return 0.5 * (left_gain + right_gain - total_gain)
    
    
    def _calculate_hyper_tune_score(self, left_sum_grad, left_sum_hessian, right_sum_grad, right_sum_hessian, smoothing_lambda):  
        left_num = left_sum_grad**2
        right_num = right_sum_grad**2    
        left_denum = np.clip(left_sum_hessian + smoothing_lambda, a_min=smoothing_lambda, a_max=np.Inf)
        right_denum = np.clip(right_sum_hessian + smoothing_lambda, a_min=smoothing_lambda, a_max=np.Inf)
        
        score = left_num / left_denum + right_num / right_denum
        return score
    


    def _calculate_leaf_weight(self, total_grads, total_hess):
        """
        Calculates weight for leaf nodes, with optional learning rate specified by self.reg_eta

        :param total_grads: Sum of gradients
        :param total_hess: Sum of hessians
        :return: Leaf weight
        """
        if self.reg_alpha == 0:
            reg_alpha = float("inf")
        else:
            reg_alpha = self.reg_alpha

        if self.num_classes > 2:
            total_hess = np.clip(total_hess, 0, float("inf"))
            weight = -1 * (np.clip(total_grads, -reg_alpha, reg_alpha) / (total_hess + self.reg_lambda))
            if self.reg_delta != 0:
                clip_idx = np.abs(weight) > self.reg_delta
                weight[clip_idx] = np.copysign(self.reg_delta, weight[clip_idx])
            return weight
        else:
            return self._calculate_weight(total_grads, total_hess) * self.reg_eta  # Multiply the weight by the learning rate for leaf values

    # Main training logic
    # ---------------------------------------------------------------------------------------------------

    # Public method to train the model
    def fit(self, X, y):
        """
        Main training loop

        :param X: Training data as a pandas dataframe/ numpy array
        :param y: Training labels
        :return: self (trained GBDT model)
        """

        if self.seed_numpy is not None:
            np.random.seed(self.seed_numpy)
        
        if self.seed_random is not None:
            random.seed(self.seed_random)

        X = self._convert_df(X)
        self.num_features = X.shape[1]
        self.feature_list = range(0, self.num_features)  

        # Calculate split candidates
        self.split_candidate_manager.find_split_candidates(X, 0, None, features_considering=self.feature_list)  

        if self.split_method not in ["grad_based", "hyper_tune"]:
            self.privacy_accountant.update_feature_candidate_size(self.split_candidate_manager.feature_split_candidates)

        self.X = X
        self.train_monitor.batched_weights = np.zeros(self.X.shape[0])

        if "cyclical" in self.feature_interaction_method and (self.split_candidate_manager.sketch_type == "adaptive_hessian" or self.full_ebm):
            if self.split_method not in ["grad_based", "hyper_tune"]:

                if self.full_ebm:
                    self.num_trees = self.num_trees * X.shape[1]
                self.split_candidate_manager.sketch_rounds = min(self.num_trees, self.split_candidate_manager.sketch_rounds*self.num_features)

                # recompute budget allocation
                self.privacy_accountant.__init__(self.privacy_accountant.accounting_method, epsilon=self.privacy_accountant.epsilon, delta=self.privacy_accountant.delta,
                                                        quantile_epsilon=self.privacy_accountant.quantile_epsilon, dp_method=self.dp_method,
                                                        num_trees=self.num_trees, max_depth=self.max_depth, split_method=self.split_method, training_method=self.training_method, weight_update_method=self.weight_update_method,
                                                        split_method_per_level=self.split_method_per_level,
                                                        feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                        sample_method = self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                        sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                        task_type=self.task_type, sigma=self.privacy_accountant.sigma,
                                                        grad_clip_const=self.privacy_accountant.grad_clip_const, gradient_clipping=self.privacy_accountant.gradient_clipping,
                                                        verbose=self.verbose,)

        if self.batched_update_size < 1:
            self.batched_update_size = int(self.batched_update_size * self.num_trees)

        y = y if not isinstance(y, pd.Series) else y.values
        self.num_classes = len(np.unique(y))
        self.train_monitor.set_num_classes(self.num_classes)

        if self.num_classes > 2:
            self.loss = SoftmaxCrossEntropyLoss()
            y = LabelBinarizer().fit_transform(y)
            self.train_monitor.y_weights = np.full((X.shape[0], self.num_classes), 1/self.num_classes,)
            self.train_monitor.current_tree_weights = np.zeros((X.shape[0], self.num_classes))
        else:
            self.train_monitor.y_weights = np.zeros(X.shape[0]) # Initialise training weights to zero which is sigmoid(0) = 0.5 prob to either class
            self.train_monitor.current_tree_weights = np.zeros(X.shape[0])

        # Initialise Gaussian DP parameters
        if "gaussian" in self.dp_method and self.split_method not in ["grad_based", "hyper_tune"]:
            self.privacy_accountant.assign_budget(self.privacy_accountant.epsilon, 1 / X.shape[0], num_rows=X.shape[0], num_features=X.shape[1]) # Update delta to 1/n
        if self.split_method in ["grad_based", "hyper_tune"]:
            # re-init to set new delta and thus update beta, sigma
            self.privacy_accountant.__init__(loss_name=self.loss_name, epsilon=self.epsilon, delta=1 / X.shape[0], dp_method=self.dp_method,
                                             num_trees=self.num_trees, num_features=self.num_features, max_depth=self.max_depth,
                                             split_method=self.split_method, task_type=self.task_type, sketch_type=self.sketch_type,
                                             ratio_hist=self.ratio_hist, ratio_leaf=self.ratio_leaf, ratio_selection=self.ratio_selection,
                                             selection_mechanism=self.selection_mechanism,
                                             feature_interaction_method=self.feature_interaction_method,
                                             feature_interaction_k=self.feature_interaction_k
                                             )
            # print("reinit-finished")
            #print(self.privacy_accountant.sigma_hist)

        # Form histogram bin assignments for each feature - this caching saves a lot of time for histogram based gradient aggregation later on
        for i in range(0, self.num_features):
            self.feature_bin.append(
                np.digitize(self.X[:, i], 
                            bins=[-np.inf] + list(np.array(self.split_candidate_manager.feature_split_candidates[i]) + 1e-11) + [np.inf])
            )

        self.feature_bin = np.array(self.feature_bin)
        features = np.array(range(0, self.num_features))
        previous_rounds_features = None

        for i in range(0, self.num_trees):
            
            self.train_monitor.node_count = -1 # Reset node count for new trees

            if self.split_candidate_manager.sketch_each_tree:
                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) >= self.split_candidate_manager.sketch_rounds:
                    pass
                else:
                    features_updated = previous_rounds_features if previous_rounds_features is not None else list(range(0, self.num_features))
                    self.split_candidate_manager.find_split_candidates(X, len(self.trees), self.root_hessian_histogram, features_considering=features_updated)

                    for j in features_updated:
                        self.feature_bin[j] = np.digitize(self.X[:, j], bins=[-np.inf] + list(np.array(self.split_candidate_manager.feature_split_candidates[j]) + 1e-11) + [np.inf])

            # Row and Feature Sampling if enabled
            row_sample, col_tree_sample, col_level_sample = self.index_sampler.sample(i, X.shape[0], X.shape[1], self.max_depth, feature_interaction_k=self.feature_interaction_k, feature_interaction_method=self.feature_interaction_method)
            previous_rounds_features = col_tree_sample
            split_constraints = {i : [0, len(self.split_candidate_manager.feature_split_candidates[i])+1] for i in range(0,self.num_features)}

            if i != 0 and self.split_method not in ["grad_based", "hyper_tune"]:
                self.privacy_accountant.update_tree() # Increment tree count in privacy_accountant, used to index tree_budgets

            if i==0 or self.training_method == "boosting" or (self.training_method == "batched_boosting" and (i % self.batched_update_size == 0)):
                if self.training_method == "batched_boosting":
                    self.train_monitor.y_weights += self.train_monitor.batched_weights / self.batched_update_size
                    self.train_monitor.batched_weights = np.zeros(self.X.shape[0])
                    
                # compute public grad and hessian
                ## once per tree, but use left/right_split_index to select points at each node
                grads, hess = self._compute_grad_hessian_with_samples(y, self.loss.predict(self.train_monitor.y_weights)) # Compute raw grads,hess
                self.train_monitor.gradient_info = [(grads, hess)] # Append to gradient_info, at each node this is retrieved and privatised with DP to calculate feature scores etc
                #print(grads)

            tree = self._build_tree(features, row_sample, None, None,
                                    split_constraints=split_constraints, col_tree_sample=col_tree_sample, col_level_sample=col_level_sample, row_ids=np.arange(0,X.shape[0]))
            self.trees.append(tree) # Build and add tree to ensemble


            if self.training_method == "batched_boosting":
                self.train_monitor.batched_weights += self.train_monitor.current_tree_weights

                if i==self.num_trees-1 and (i+1) % self.batched_update_size != 0:
                    self.train_monitor.y_weights += self.train_monitor.batched_weights / ((i+1) % self.batched_update_size)
                elif i==self.num_trees-1:
                    pass
            else:
                self.train_monitor.y_weights += self.train_monitor.current_tree_weights # Update weights

            self.train_monitor.leaf_gradient_tracker[0].append(self.train_monitor.gradient_total[0])
            self.train_monitor.leaf_gradient_tracker[1].append(self.train_monitor.gradient_total[1])

            
            # Reset tracking vars
            self.train_monitor._update_comm_stats(self.split_method, self.training_method)
            self.train_monitor.reset()
        
        self.root = self.trees[0]
        
        self.y_weights = self.train_monitor.y_weights
        
        return self
    

    def _calculate_feature_split(self, features_considering, split_index, current_depth, total_gain, total_grads, total_hess, grads, hess, feature_split_constraints):
        """

        Calculates split scores for a specific features and all their split candidate values

        :param feature_i: Feature index
        :param feature_values: Feature values
        :param current_depth: Current depth in the tree
        :param total_gain: Total gain
        :param total_grads: Total grads
        :param total_hess: Total hess
        :param grads: list of grads
        :param hess: list of hess
        :return:
        """
        # Iterate through all unique values of feature column i and calculate the impurity
        values = []
        current_max_score = -np.Inf
        split_method = self.split_method_per_level[current_depth]
        self.current_tree_depth = max(self.train_monitor.current_tree_depth, current_depth)

        valid_features = []
        for i in features_considering:
            if feature_split_constraints[i][0] < min(feature_split_constraints[i][1], len(self.split_candidate_manager.feature_split_candidates[i])): # Ignore features that cannot be split on any further
                valid_features.append(i)

        if len(valid_features) == 0:
            return []

        if split_method == "totally_random":
            new_features = valid_features
            weights = None
            chosen_feature = np.random.choice(new_features, p=weights)
            chosen_split = np.random.choice(
                range(feature_split_constraints[chosen_feature][0], min(feature_split_constraints[chosen_feature][1], len(self.split_candidate_manager.feature_split_candidates[chosen_feature])))
                ) # Upper split constraint may be len(feature_splits)+1 so needs to be truncated back down; This is due to how the hist_based splitting logic works because of the splicing              
        else: # when split_method != "totally_random":
            for feature_i in valid_features:
                constraints = feature_split_constraints[feature_i]
                self.train_monitor.bin_tracker[current_depth] += constraints[1] - constraints[0]

        if split_method == "hyper_tune":
            threshold_indicator = 0 # moving index
            candidate_set_size = 0
            for i in valid_features:
                candidate_set_size += len(range(feature_split_constraints[i][0], min(feature_split_constraints[i][1], len(self.split_candidate_manager.feature_split_candidates[i]))))
            
            if self.k_rounds <= 1: # same as totally random 
                chosen_thresholds = np.random.choice(range(candidate_set_size), size=1)
            else:
                chosen_thresholds = np.random.choice(range(candidate_set_size), replace=True, size=self.k_rounds)   
            
        for feature_i in valid_features:
            split_constraints = feature_split_constraints[feature_i]

            if split_method == "partially_random":
                chosen_split = random.randint(split_constraints[0], min(split_constraints[1], len(self.split_candidate_manager.feature_split_candidates[feature_i])))
            elif split_method == "totally_random" and feature_i != chosen_feature:
                continue

            if split_method == "hist_based":
                cumulative_grads = np.cumsum(self.private_gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                cumulative_hess = np.cumsum(self.private_hessian_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                total_grads_cu = cumulative_grads[-1]
                total_hess_cu = cumulative_hess[-1]
            
            if split_method == "grad_based":
                cumulative_grads = np.cumsum(self.gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                cumulative_hess = np.cumsum(self.counts_histogram[feature_i][split_constraints[0]: split_constraints[1]+1]) # use counts as hessian when calculate splitting score
                total_grads_cu = cumulative_grads[-1]
                total_hess_cu = cumulative_hess[-1]
            
            if split_method == "hyper_tune":
                cumulative_grads = np.cumsum(self.gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                cumulative_hess = np.cumsum(self.hessian_histogram[feature_i][split_constraints[0]: split_constraints[1]+1]) # use hessian
                total_grads_cu = cumulative_grads[-1]
                total_hess_cu = cumulative_hess[-1]
                
                
            # vectorization for grad_based
            
            if split_method == "grad_based":            
                
                j_array= []
                for j, threshold in enumerate(self.split_candidate_manager.feature_split_candidates[feature_i]):
                    if split_constraints[0] <= j <= split_constraints[1] or self.ignore_split_constraints:
                        j_array.append(j)
                j_array = np.array(j_array)
                
                left_grads_sum = cumulative_grads[j_array - (split_constraints[0])]
                left_hess_sum = cumulative_hess[j_array - (split_constraints[0])]
                right_grads_sum = total_grads_cu - left_grads_sum
                right_hess_sum = total_hess_cu - left_hess_sum

                score_array = self._calculate_gain_grad_based(left_grads_sum, left_hess_sum) + self._calculate_gain_grad_based(right_grads_sum, right_hess_sum)
                
                if self.selection_mechanism == "exponential_mech":
                    score_array += np.random.gumbel(loc=0, scale = 3 * self.privacy_accountant.grad_sensitivity * self.privacy_accountant.beta, size=len(score_array))
                elif self.selection_mechanism == "permutate_flip":
                    score_array += np.random.exponential(scale = 3 * self.privacy_accountant.grad_sensitivity * self.privacy_accountant.beta, size=len(score_array))
                
                v = max(score_array)
                if len(score_array) != len(j_array):
                    raise ValueError("score_array doesn't has same length as j_array")
                
                if v > current_max_score:
                    j_star = j_array[np.argmax(score_array)]
                    threshold = self.split_candidate_manager.feature_split_candidates[feature_i][j_star]
                    values = [feature_i, j_star, threshold, v, None, (None, None, None, None)] 
                            # though pri_left_sum is sum over noised hist, it doesn't effect grad_based
                    current_max_score = v
            else:
                for j, threshold in enumerate(self.split_candidate_manager.feature_split_candidates[feature_i]):

                    if (split_method == "partially_random" or split_method == "totally_random") and j != chosen_split:
                        continue

                    if split_constraints[0] <= j <= split_constraints[1] or self.ignore_split_constraints: # Only add split if it isn't one-sided (based on public knowledge of previous splits)
                        # Calculate impurity score of proposed split
                        if split_method == "hist_based":
                            left_grads_sum = cumulative_grads[j-(split_constraints[0])]
                            left_hess_sum = cumulative_hess[j-(split_constraints[0])]

                            if self.hist_estimator_method == "one-sided":
                                right_grads_sum = total_grads - left_grads_sum
                                right_hess_sum = total_hess - left_hess_sum
                            else:
                                right_grads_sum = total_grads_cu - left_grads_sum
                                right_hess_sum = total_hess_cu - left_hess_sum

                            if self.hist_estimator_method == "two_sided":
                                total_grads = self.private_gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]].sum()
                                total_hess = self.private_hessian_histogram[feature_i][split_constraints[0]: split_constraints[1]].sum()
                                total_gain = self._calculate_gain(total_grads, total_hess)

                            split_score = self._calculate_split_score(self._calculate_gain(left_grads_sum, left_hess_sum), self._calculate_gain(right_grads_sum, right_hess_sum), total_gain)
                        elif split_method == "hyper_tune":
                            if threshold_indicator in chosen_thresholds:
                                num_runs = sum(chosen_thresholds == threshold_indicator)
                                
                                # vectorization to speed up
                                left_grads_sum = np.repeat(cumulative_grads[j-(split_constraints[0])], num_runs)
                                left_hess_sum = np.repeat(cumulative_hess[j-(split_constraints[0])], num_runs)
                                right_grads_sum = total_grads_cu - left_grads_sum
                                right_hess_sum = total_hess_cu - left_hess_sum
                                
                                noise = np.random.normal(loc=0, scale=self.privacy_accountant.hyper_sensitivity*self.privacy_accountant.sigma_score, size=((4, num_runs)))         
                                
                                noised_left_grads_sum = left_grads_sum + noise[0,:]
                                noised_left_hess_sum = left_hess_sum + noise[1,:]
                                noised_right_grads_sum = right_grads_sum + noise[2,:]
                                noised_right_hess_sum = right_hess_sum + noise[3,:]
                                    
                                split_score = max(self._calculate_hyper_tune_score(noised_left_grads_sum, noised_left_hess_sum, 
                                                                                noised_right_grads_sum, noised_right_hess_sum, 
                                                                                self.smoothing_lambda))
                                
                                indicator_run = True
                            else:
                                indicator_run = False
                            threshold_indicator += 1
                        elif split_method == "node_based" or split_method == "partially_random":
                            new_split_index = self.X[split_index, feature_i] <= threshold
                            left_grads_sum, left_hess_sum = self.privacy_accountant._add_dp_noise(grads[new_split_index].sum(), 
                                                                                                hess[new_split_index].sum(), 
                                                                                                current_depth, 
                                                                                                feature=feature_i, 
                                                                                                num_obs=len(new_split_index))
                            right_grads_sum = total_grads - left_grads_sum
                            right_hess_sum = total_hess - left_hess_sum
                            split_score = self._calculate_split_score(self._calculate_gain(left_grads_sum, left_hess_sum), self._calculate_gain(right_grads_sum, right_hess_sum), total_gain)
                        else: # In the case of totally random no score are computed
                            split_score = float("inf")
                            left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum = [float("inf")]*4


                        # update best split
                        if split_method not in ["grad_based", "hyper_tune"] and split_score > current_max_score:
                            # Divide X and y depending on if the feature value of X at index feature_i meets the threshold
                            values = [feature_i, j, threshold, split_score, None, (left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum)]
                            current_max_score = split_score
                        elif split_method == "hyper_tune" and indicator_run:
                            if split_score > current_max_score:
                                values = [feature_i, j, threshold, split_score, None, (None, None, None, None)]
                                current_max_score = split_score
        return values
    
    
    def _form_private_gradient_histogram(self, grads, hess, features_considering, split_index, current_depth, adaptive_hessian=False):
        """
        generate self.private_gradient_histogram and self.gradient_histogram
        """
        
        if current_depth == 0:
            self.gradient_histogram = {}
            self.hessian_histogram = {}
            self.private_gradient_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering} # place holder
            self.private_hessian_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering}
            if self.split_method == "grad_based":
                self.counts_histogram = {}

        if current_depth == 0 and len(self.trees) == 0:
            self.root_hessian_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering}

        for i in features_considering:
            num_bins = len(self.split_candidate_manager.feature_split_candidates[i])+1
            digitized = self.feature_bin[i][split_index]
            self.gradient_histogram[i] = np.array(histogram1d(digitized, bins=num_bins, range=[0, num_bins+0.1], weights=grads)) # Fast C histogram implementation
            self.hessian_histogram[i] = np.array(histogram1d(digitized, bins=num_bins, range=[0, num_bins+0.1], weights=hess))
            if self.split_method == "grad_based":
                self.counts_histogram[i] = np.array(histogram1d(digitized, bins=num_bins, range=[0, num_bins+0.1], weights=np.ones_like(hess)))


            if self.dp_method != "":
                if adaptive_hessian:
                    _, self.root_hessian_histogram[i] = self.privacy_accountant._add_dp_noise(self.gradient_histogram[i], 
                                                                                              self.hessian_histogram[i],
                                                                                              depth=self.max_depth-1,
                                                                                              feature=i, 
                                                                                              histogram_row=True, 
                                                                                              noise_size=num_bins, 
                                                                                              adaptive_hessian=True)
                    #print("success")
                else:
                    self.private_gradient_histogram[i], self.private_hessian_histogram[i] = self.privacy_accountant._add_dp_noise(self.gradient_histogram[i], 
                                                                                                                                  self.hessian_histogram[i],
                                                                                                                                  depth=current_depth,
                                                                                                                                  feature=i, 
                                                                                                                                  histogram_row=True, 
                                                                                                                                  noise_size=num_bins,
                                                                                                                                  adaptive_hessian=False)
            
        if self.dp_method == "":
            self.private_gradient_histogram, self.private_hessian_histogram = self.gradient_histogram, self.hessian_histogram
    
    
    def _get_node_id(self, depth, node_num):
        return str(depth) + "_" + str(node_num)

    def _build_tree(self, features, split_index, node_total_grads, node_total_hess,
                    current_depth=0, col_tree_sample=None, col_level_sample=None, row_ids=None, split_constraints=None, previous_node_num=1):
        """
        Main method for building a tree of the ensemble

        :param split_index: Boolean index of current observations in the node
        :param node_total_grads: Total gradients of the node
        :param node_total_hess: Total hessians of the node
        :param current_depth: Current depth of the node
        :param col_tree_sample: Boolean index of features to sample if self.colsample_bynode is not 1
        :param col_level_sample: Boolean index of features to sample if self.colsample_bylevel is not 1
        :return:
        """
        
        features_considering = features
        # Perform column (feature) sampling if needed
        if col_tree_sample is not None:
            features_considering = features_considering[col_tree_sample]
        if col_level_sample is not None:
            features_considering = features_considering[col_level_sample[current_depth]]
        if self.index_sampler.colsample_bynode < 1:
            features_considering = features_considering[np.random.choice(range(0, len(features_considering)), size=math.ceil(len(features_considering) * self.index_sampler.colsample_bynode), replace=False)]

        self.train_monitor.node_count += 1

        if self.split_method not in ["grad_based", "hpyer_tune"]:
            self.privacy_accountant.current_node = self.train_monitor.node_count
            self.privacy_accountant.current_tree = len(self.trees)
        split_method = self.split_method_per_level[min(current_depth, self.max_depth-1)]
        
        # RAW! gradients/hessians for the observations in the current node
        grads, hess = self.train_monitor.gradient_info[-1][0][split_index], self.train_monitor.gradient_info[-1][1][split_index]

        # RAW! gradients/hessians sum
        if self.num_classes > 2:
            raw_grads_sum, raw_hess_sum = np.sum(grads, axis=0), np.sum(hess, axis=0)
        else:
            raw_grads_sum, raw_hess_sum = grads.sum(), hess.sum()

        if current_depth == 0: # Update private grads at root node
            if split_method == "node_based" or split_method == "partially_random":
                node_total_grads, node_total_hess = self.privacy_accountant._add_dp_noise(raw_grads_sum, raw_hess_sum, -1, num_obs=len(split_index)) # Depth zero
            
            elif split_method == "totally_random":
                node_total_hess = float("inf")
                node_total_grads = float("inf")

                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) < self.split_candidate_manager.sketch_rounds:
                    # Adaptive hess histogram
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=True) 
            
            elif split_method == "hist_based":
                self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth) # Form privatised grads,hess
                
                # average estimator for noised sum of gradient and hessian
                node_total_grads = sum([i.sum() for i in self.private_gradient_histogram.values()])/len(self.private_gradient_histogram.values())
                node_total_hess = sum([i.sum() for i in self.private_hessian_histogram.values()])/len(self.private_hessian_histogram.values())
            
            elif split_method == "grad_based":
                # generate (grad_hist, hess_hist, counts_hist)

                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) < self.split_candidate_manager.sketch_rounds:
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=True) # Adaptive hess test
                else:
                    # Form privatised grads,hess
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=False)
                
                # this two are PUBLIC
                node_total_grads = sum([i.sum() for i in self.gradient_histogram.values()])/len(self.gradient_histogram.values())
                node_total_hess = sum([i.sum() for i in self.counts_histogram.values()])/len(self.counts_histogram.values())
            
            elif split_method == "hyper_tune":
                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) < self.split_candidate_manager.sketch_rounds:
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=True) # Adaptive hess test
                else:
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=False) # Adaptive hess test
                
                self.k_rounds = np.random.logseries(self.hyper_p)

            # for early stopping
            self.train_monitor.root_gradient_tracker[0].append(node_total_grads)
            self.train_monitor.root_gradient_tracker[1].append(node_total_hess)
        
        # For nodes with depth > 0!
        # If the spliting conditions are satisfied then split the current node otherwise stop and make it a leaf
        # For "totally_random", "grad_based", always split until reach max depth.
        if (split_method in ["totally_random", "grad_based", "hyper_tune"] or node_total_hess >= self.min_child_weight) and (current_depth < self.max_depth):
            if split_method == "hist_based" and current_depth > 0:
                self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth) # Form privatised grads,hess
                if self.hist_estimator_method == "two_sided_averaging" or node_total_grads == float("inf"):
                    node_total_grads = sum([i.sum() for i in self.private_gradient_histogram.values()])/len(self.private_gradient_histogram.values())
                    node_total_hess = sum([i.sum() for i in self.private_hessian_histogram.values()])/len(self.private_hessian_histogram.values())
            
            if split_method in ["grad_based", "hyper_tune"] and current_depth > 0:
                self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth) # Form non-privatised grads,hess

            if split_method not in ["grad_based", "hyper_tune"]:
                # Calculate current nodes total gain
                node_gain = self._calculate_gain(node_total_grads, node_total_hess)
                # Find best (feature, split) candidates for each feature
                split_data = self._calculate_feature_split(features_considering, split_index, current_depth, node_gain, node_total_grads, node_total_hess, grads, hess, split_constraints)
            elif split_method == "grad_based":
                raw_node_total_grads = sum([i.sum() for i in self.gradient_histogram.values()])/len(self.gradient_histogram.values())
                raw_node_total_hess = sum([i.sum() for i in self.counts_histogram.values()])/len(self.counts_histogram.values())
                node_gain = 0 # since we use eqn 3 in https://arxiv.org/pdf/1911.04209.pdf
                
                # the output gradient infos in split_data has been noised
                split_data = self._calculate_feature_split(features_considering, split_index, current_depth, node_gain, raw_node_total_grads, raw_node_total_hess, grads, hess, split_constraints)        
            elif split_method == "hyper_tune":
                node_gain = None
                total_grads = None
                total_hess = None
                split_data = self._calculate_feature_split(features_considering, split_index, current_depth, node_gain, total_grads, total_hess, grads, hess, split_constraints)
            

            # Commit budget spent by participants for computing split scores
            
            # Note: node_total_grads/hess privitaized or not doesn't influence "grad_based"
            if split_data:
                chosen_feature, bucket_index, chosen_threshold, largest_score, left_split_index, split_gradient_info = split_data
                if left_split_index is None:
                    left_split_index = self.X[split_index, chosen_feature] <= chosen_threshold
                left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum = split_gradient_info # Gradient information to pass to child nodes
                right_split_index = split_index[~left_split_index]
                left_split_index = split_index[left_split_index]

                if largest_score > self.reg_gamma or self.split_method in ["grad_based", "hyper_tune"]:
                    # TODO: check if ignore reg_gamma for grad_based is valid -> Doesn't matter, always let it grow
                    # Update feature split constraints with valid feature split candidate index bounds - this stops the algo from picking one-sided splits later on
                    left_split_constraints = copy(split_constraints)
                    left_split_constraints[chosen_feature] = [left_split_constraints[chosen_feature][0], bucket_index-1]
                    right_split_constraints = copy(split_constraints)
                    right_split_constraints[chosen_feature] = [bucket_index+1, right_split_constraints[chosen_feature][1]]

                    # Build subtrees recursively for the right and left branches
                    self.train_monitor.last_feature = chosen_feature

                    left_num = 2*(previous_node_num)-1
                    right_num = 2*(previous_node_num)

                    left_branch = self._build_tree(features, left_split_index, left_grads_sum, left_hess_sum,
                                                   current_depth + 1, col_tree_sample, col_level_sample, split_constraints=left_split_constraints, previous_node_num=left_num)
                    right_branch = self._build_tree(features, right_split_index, right_grads_sum, right_hess_sum,
                                                    current_depth + 1, col_tree_sample, col_level_sample, split_constraints=right_split_constraints, previous_node_num=right_num)

                    self.train_monitor.internal_node_count[current_depth] += 1
                    return DecisionNode(node_id=str(current_depth) + "_" + str(previous_node_num), feature_i=chosen_feature, threshold=chosen_threshold, true_branch=left_branch, false_branch=right_branch, split_gain=largest_score, gradient_sum=node_total_grads, hessian_sum=node_total_hess, num_observations=len(split_index), depth=current_depth)
                
        # We're at leaf => determine weight ???????????????
        if split_method == "totally_random":
            if self.dp_method != "" and self.dp_method != "gaussian_ldp":
                size = self.num_classes if self.num_classes > 2 else None
                raw_grads_sum = raw_grads_sum.sum()
                raw_hess_sum = raw_hess_sum.sum()

                node_total_grads = raw_grads_sum + np.random.normal(0, self.privacy_accountant.gaussian_var(gradient_type="gradient", depth=self.max_depth-1), size=size)
                node_total_hess = raw_hess_sum + np.random.normal(0, self.privacy_accountant.gaussian_var(gradient_type="hessian", depth=self.max_depth-1), size=size)
            else:
                node_total_grads, node_total_hess = raw_grads_sum, raw_hess_sum
        elif self.split_method in ["grad_based", "hyper_tune"]:
            size = self.num_classes if self.num_classes > 2 else None
            raw_grads_sum = raw_grads_sum.sum()
            raw_hess_sum = raw_hess_sum.sum()
            node_total_grads = raw_grads_sum + np.random.normal(0, self.privacy_accountant.leaf_sensitivity * self.privacy_accountant.sigma_leaf, size=size)
            node_total_hess = raw_hess_sum + np.random.normal(0, self.privacy_accountant.leaf_sensitivity * self.privacy_accountant.sigma_leaf, size=size)
        else:
            pass
        
        leaf_weight = self._calculate_model_update(node_total_grads, node_total_hess, None, None) # pass grads/hess?

        
        # Update training information...
        # gradient_total is for early stopping 
        self.train_monitor.gradient_total[0] += node_total_grads
        self.train_monitor.gradient_total[1] += node_total_hess
        self.train_monitor.leaf_count += 1
        self.train_monitor.current_tree_weights[split_index] += leaf_weight

        if self.num_classes == 2:
            leaf_weight = np.array([leaf_weight])
        return DecisionNode(node_id=str(current_depth) + "_" + str(previous_node_num), 
                            value=leaf_weight, 
                            num_observations=len(split_index), 
                            gradient_sum=node_total_grads,
                            hessian_sum=node_total_hess,
                            split_gain=0,
                            feature_i=self.train_monitor.last_feature)

    def _calculate_model_update(self, node_total_grads, node_total_hess,  grads=None, hess=None):
        if self.training_method == "rf":  # RF update
            if node_total_hess <= 0:
                leaf_weight = 0.5
            elif node_total_grads <= 0:
                leaf_weight = 0
            else:
                leaf_weight = node_total_grads/node_total_hess
                if leaf_weight > 1:
                    leaf_weight = 1
        else: # Grad or Newton update
            if self.num_classes > 2:
                leaf_weight = self._calculate_leaf_weight(node_total_grads, node_total_hess)
            else:
                if node_total_hess <= self.privacy_accountant.min_hess:
                    leaf_weight = 0
                else:
                    node_total_grads = np.clip(node_total_grads, 
                                               self.privacy_accountant.min_gradient*self.X.shape[0], 
                                               self.privacy_accountant.max_gradient*self.X.shape[0])
                    node_total_hess = np.clip(node_total_hess, 
                                              self.privacy_accountant.min_hess*self.X.shape[0], 
                                              self.privacy_accountant.max_hess*self.X.shape[0])
                    leaf_weight = self._calculate_leaf_weight(node_total_grads, node_total_hess)

        # Signed or individual updates...
        if "signed" in self.weight_update_method or "per_sample" in self.weight_update_method:
            leaf_weight = (-grads/(hess+self.reg_lambda)) if "newton" in self.weight_update_method else -grads
            # leaf_weight = np.sign(leaf_weight) # local sign - doesnt work...

            leaf_weight = leaf_weight.sum()

            if "signed" in self.weight_update_method:
                if leaf_weight < 0:
                    leaf_weight = -self.reg_delta
                elif leaf_weight > 0:
                    leaf_weight = self.reg_delta
                else:
                    leaf_weight = 0

            leaf_weight = np.clip(leaf_weight, -self.reg_delta, self.reg_delta)
            leaf_weight *= self.reg_eta
            # print(leaf_weight)

        return leaf_weight
