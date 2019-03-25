import numpy as np

def find_optimal_threshold(fpr, tpr, thresholds, optimal_performance=(0, 1)):
    """
    Find the optimal threshold below which the classifier assigns points to 
    the '0' class and above which it assigns points to the '1' class.
    
    This is the threshold for which the fpr (false positive rate) and tpr (true
    positive rate) are as close as possible (in Euclidean distance) to
    an optimal classifier for which fpr=0 and tpr=1. (This corresponds to the
    tuple of `optimal_performance` which is an optional argument to this function.)
    
    Parameters
    ----------
    fpr : array, shape = [>2]
        Increasing false positive rates for a binary classifier such that 
        element i is the false positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates for a binary classifier such that 
        element i is the true positive rate of predictions with score >= thresholds[i].
    thresholds : array, shape = [>2]
        Decreasing thresholds on the decision function used to compute fpr and tpr. 
        thresholds[0] represents no instances being predicted and is arbitrarily set.
    optimal_performance : tuple(float, float), optional
        Tuple representing the optimal fpr and tpr. (The default is (0, 1), 
        which corresponds to fpr=0 and tpr=1).

    Note
    ----
    fpr, tpr and thresholds are parameters returned by the `sklearn.metrics.roc_curve()`
    method and hence the specific way in which fpr, tpr and thresholds get passed as 
    arguments here. This function should be passed the return values of `sklearn.metrics.roc_curve()`
    method. 
    
    Returns
    ------
     opt_prob_threshold : float (between 0 and 1)
                        The probability threshold below which the classifier
                        assigns points to the '0' class and above which it
                        assigns points to the '1' class. For this threshold,
                        fpr and tpr are as close as possible to 
                        (`optimal performance`)
    """
    

    assert 0 <= optimal_performance[0] <= 1 and 0 <= optimal_performance[1] <= 1, \
        'Each element of this tuple must be between 0 and 1'    
        
    optimal_idx = np.array([np.sqrt((x - optimal_performance[0])**2 +  \
                          (y - optimal_performance[1])**2) for \
                          (x, y) in zip(fpr[1:], tpr[1:])]).argmin()
    
    opt_prob_threshold = thresholds[1:][optimal_idx]
    return opt_prob_threshold
