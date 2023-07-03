from federated_gbdt.core import baseline_constants
import copy


class TransformParam:
    """
    Define how to transfer the cols

    Parameters
    ----------
    transform_cols : list of column index, default: -1
        Specify which columns need to be transform. If column index is None, None of columns will be transformed.
        If it is -1, it will use same columns as cols in binning module.

    transform_names: list of string, default: []
        Specify which columns need to calculated. Each element in the list represent for a column name in header.


    transform_type: str, 'bin_num'or 'woe' or None default: 'bin_num'
        Specify which value these columns going to replace.
         1. bin_num: Transfer original feature value to bin index in which this value belongs to.
         2. woe: This is valid for guest party only. It will replace original value to its woe value
         3. None: nothing will be replaced.
    """

    def __init__(self, transform_cols=-1, transform_names=None, transform_type="bin_num"):
        super(TransformParam, self).__init__()
        self.transform_cols = transform_cols
        self.transform_names = transform_names
        self.transform_type = transform_type


class OptimalBinningParam:
    """
    Indicate optimal binning params

    Parameters
    ----------
    metric_method: str, default: "iv"
        The algorithm metric method. Support iv, gini, ks, chi-square


    min_bin_pct: float, default: 0.05
        The minimum percentage of each bucket

    max_bin_pct: float, default: 1.0
        The maximum percentage of each bucket

    init_bin_nums: int, default 100
        Number of bins when initialize

    mixture: bool, default: True
        Whether each bucket need event and non-event records

    init_bucket_method: str default: quantile
        Init bucket methods. Accept quantile and bucket.

    """

    def __init__(self, metric_method='iv', min_bin_pct=0.05, max_bin_pct=1.0,
                 init_bin_nums=1000, mixture=True, init_bucket_method='quantile'):
        super().__init__()
        self.init_bucket_method = init_bucket_method
        self.metric_method = metric_method
        self.max_bin = None
        self.mixture = mixture
        self.max_bin_pct = max_bin_pct
        self.min_bin_pct = min_bin_pct
        self.init_bin_nums = init_bin_nums
        self.adjustment_factor = None


class FeatureBinningParam:
    """
    Define the feature binning method

    Parameters
    ----------
    method : str, 'quantile'， 'bucket' or 'optimal', default: 'quantile'
        Binning method.

    compress_thres: int, default: 10000
        When the number of saved summaries exceed this threshold, it will call its compress function

    head_size: int, default: 10000
        The buffer size to store inserted observations. When head list reach this buffer size, the
        QuantileSummaries object start to generate summary(or stats) and insert into its sampled list.

    error: float, 0 <= error < 1 default: 0.001
        The error of tolerance of binning. The final split point comes from original data, and the rank
        of this value is close to the exact rank. More precisely,
        floor((p - 2 * error) * N) <= rank(x) <= ceil((p + 2 * error) * N)
        where p is the quantile in float, and N is total number of data.

    bin_num: int, bin_num > 0, default: 10
        The max bin number for binning

    bin_indexes : list of int or int, default: -1
        Specify which columns need to be binned. -1 represent for all columns. If you need to indicate specific
        cols, provide a list of header index instead of -1.

    bin_names : list of string, default: []
        Specify which columns need to calculated. Each element in the list represent for a column name in header.

    adjustment_factor : float, default: 0.5
        the adjustment factor when calculating WOE. This is useful when there is no event or non-event in
        a bin. Please note that this parameter will NOT take effect for setting in host.

    category_indexes : list of int or int, default: []
        Specify which columns are category features. -1 represent for all columns. List of int indicate a set of
        such features. For category features, bin_obj will take its original values as split_points and treat them
        as have been binned. If this is not what you expect, please do NOT put it into this parameters.

        The number of categories should not exceed bin_num set above.

    category_names : list of string, default: []
        Use column names to specify category features. Each element in the list represent for a column name in header.

    local_only : bool, default: False
        Whether just provide binning method to guest party. If true, host party will do nothing.

    transform_param: TransformParam
        Define how to transfer the binned data.

    need_run: bool, default True
        Indicate if this module needed to be run

    skip_static: bool, default False
        If true, binning will not calculate iv, woe etc. In this case, optimal-binning
        will not be supported.

    """

    def __init__(self, method=baseline_constants.QUANTILE,
                 compress_thres=baseline_constants.DEFAULT_COMPRESS_THRESHOLD,
                 head_size=baseline_constants.DEFAULT_HEAD_SIZE,
                 error=baseline_constants.DEFAULT_RELATIVE_ERROR,
                 bin_num=baseline_constants.G_BIN_NUM, bin_indexes=-1, bin_names=None, adjustment_factor=0.5,
                 transform_param=TransformParam(), optimal_binning_param=OptimalBinningParam(),
                 local_only=False, category_indexes=None, category_names=None,
                 need_run=True, skip_static=False):
        self.method = method
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.adjustment_factor = adjustment_factor
        self.bin_num = bin_num
        self.bin_indexes = bin_indexes
        self.bin_names = bin_names
        self.category_indexes = category_indexes
        self.category_names = category_names
        self.local_only = local_only
        self.transform_param = copy.deepcopy(transform_param)
        self.optimal_binning_param = copy.deepcopy(optimal_binning_param)
        self.need_run = need_run
        self.skip_static = skip_static
