# -*- coding: utf-8 -*-
def partial_dependence_plot_custom(gbrt, 
                                   target_variable_name, 
                                   label_name,
                                   X, 
                                   grid=None, 
                                   percentiles=(0.05, 0.95), 
                                   grid_resolution=100):   
    
    """Partial dependence plot for ``target_variable_name``
    Plots 1D Partial Dependence Plots for Gradient Goosted Classifiers.
    
    [Current functionality is not yet extended to Gradient Boosted Regressors]
    
    Parameters
    ----------
    gbrt : BaseGradientBoosting
        A fitted gradient boosting classification (binary or multilabel) model.  
        
    target_variable_name : str
        Name of the variable for which partial dependence plot is to be drawn.
        
    label_name : str
        The class label name for which the PDPs should be drawn.    
        
    X : array-like, shape=(n_samples, n_features)
        The data on which ``gbrt`` was trained.
        
    grid_resolution : int, default=100
        The number of equally spaced points on the axes.
        
    percentiles : (low, high), default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values
        for the PDP axes..
        
    Returns
    -------
    fig : figure
        The Matplotlib Figure object.
    ax : Axis object
        An Axis object, for the plot.
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    from sklearn.ensemble.partial_dependence import partial_dependence

    if label_name is None:
        raise ValueError('label is not given for PDP')
        
    if target_variable_name is None:
        raise ValueError('target variable is not given for PDP')
        
    if label_name not in gbrt.classes_:
        raise ValueError('label `%s` not in ``gbrt.classes_``' % str(label_name))
    
    if target_variable_name not in X.columns:
        raise ValueError('target variable `%s` is not a feature' % str(target_variable_name))
        
    target_variable_idx = list(X.columns).index(target_variable_name)
    label_idx = list(gbrt.classes_).index(label_name)
    
    log_odds, x_vals = partial_dependence(gbrt=gbrt, 
                                          target_variables=target_variable_idx, 
                                          grid=grid, 
                                          X=X, 
                                          percentiles=percentiles, 
                                          grid_resolution=grid_resolution)
    
    x_vals = x_vals.pop() #Get `target_variables` values. list.pop() is necessary
                          #because of how partial_dependence() returns values.
    
    #Compute the probabilities for each class. Referenc:  Eqns (29) and (30) 
    #in https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451
    odds = np.exp(log_odds)
    prob_array = odds/ odds.sum(axis=0)
    
    #lot one dimensional PDP for the label corresponding to `label_name`
    plt.figure(figsize=(12, 8)) #This hardcoded value provides good aspect-ratio
    plt.plot(x_vals, prob_array[label_idx], lw=1.5)
    fig = plt.gcf()
    return fig, fig.axes.pop()