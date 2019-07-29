from collections import namedtuple
from functools import partial
import scipy.stats as st


def calculate_pairwise_sample_size(alpha, beta, metric_baseline, metric_test, 
                                   type_of_design, r=1, delta=0):
    """Compute sample sizes for different A/B experiment designs for binary
    metrics. 
    
    Compares sample sizes for simple A/B experiments (standard, non-inferiority,
    superiority, equivalence). Uses references from section 4.2 in 
    `SAMPLE SIZE CALCULATIONS IN CLINICAL RESEARCH` by Shein-Chung Chow, 
    Jun Shao and Hansheng Wang. 
    
    Parameters
    ----------
    alpha : float
        experiment wide type 1 error bound
    beta : float
        experiment wide type 2 error bound
    metric_baseline : float
        baseline metric
    metric_test : float
        expected test metric
    type_of_design : str
        One of 'standard', 'non-inferiority', 'superiority', 'equivalence'
    r : int, optional
        Ratio of n_test/ n_baseline, by default 1
    delta : int, optional
        non-inferiority/ superiority margin, by default 0 (which corresponds to
        standard AB test.) When delta < 0, the rejection of the null hypothesis 
        indicates the non-inferiority of the test against the control (baseline).

        When delta > 0, the rejection of the null hypothesis 
        indicates the superiority of the test against the control (baseline).

    Returns
    -------
    sample_size_numbers : namedtuple
        (n_baseline, n_test)
    """
    sample_size = namedtuple('sample_size', 'n_baseline n_test')
    assert type_of_design in ('standard', 'non-inferiority', 'superiority', 'equivalence')
    
    effect_size = metric_test - metric_baseline
    
    if type_of_design == 'standard':
        modified_alpha = alpha/ 2  # 2 sided test
    else:
        modified_alpha = alpha
    
    if type_of_design == 'equivalence':
        modified_beta = beta/2
    else:
        modified_beta = beta
        
    Z_a = st.norm.ppf(1 - modified_alpha)
    Z_b = st.norm.ppf(1 - modified_beta)
    
    pooled_var = ((metric_test*(1 - metric_test)/ r) + metric_baseline*(1 - metric_baseline))
    
    num = (Z_a + Z_b)**2
    
    if type_of_design == 'equivalence':
        den = (delta - abs(effect_size))**2
    else:
        den = (effect_size - delta)**2
    
    n_baseline = (num * pooled_var)/ den
    n_test = r*n_baseline
    
    sample_size_numbers = sample_size(n_baseline=n_baseline, n_test=n_test)
    
    return sample_size_numbers

if __name__ == "__main__":
    ab = partial(calculate_pairwise_sample_size, alpha=0.05, beta=0.2, 
                 metric_baseline = 0.1, metric_test=0.11, r=1) # 10% relative lift
    
    # Let's look at 5% wiggle room for non-inferiority/ superiority/ equivalence
    # designs

    print("Sample sizes for non-inferiority "
    "design are `{}`".format(ab(type_of_design='non-inferiority', delta=-0.005)))
    print("Sample sizes for standard "
    "design are `{}`".format(ab(type_of_design='standard', delta=0)))
    print("Sample sizes for superiority "
    "design are `{}`".format(ab(type_of_design='superiority', delta=0.005)))
    print("Sample sizes for equivalence"
    "design are `{}".format(ab(type_of_design='equivalence', delta=0.005)))
    