# -*- coding: utf-8 -*-
# -*- author: Karthik Iyer -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('fivethirtyeight')


class PartialDependence(object):
    """
    Custom class for calculating and plotting Partial Dependence plots giving
    more flexibility compared to sklearn's implementation.

    Partial dependence plots show the dependence between the target function
    and a set of ‘target’ features, marginalizing over the values of
    all other features (the complement features)

    Algorithm for partial dependence taken from section 8.2 in
    https://projecteuclid.org/euclid.aos/1013203451

    Parameters
    ----------
    model : scikit-learn model object
        A scikit learn classifier or a regressor object.
        If classifier, then it must implement predict_proba() method

    feature_name: str
        The name of the target feature(s) for which partial dependence plots are
        to be created

    training_df : pandas dataframe, shape=(n_samples, n_features)
        The data on which model was trained. Currently there is no support for
        numpy ndarray

    n_grid : int, default=100
        The number of equally spaced points for target feature `feature_name`
        for which partial dependence function is to be computed.

    percentile : tuple: (low, high), default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme value for
        target feature `feature_name`.

    """

    def __init__(self, model, feature_name, training_df,
                 percentile=(0.05, 0.95), n_grid=100):

        self.model = model
        self.feature_name = feature_name
        self.training_df = training_df
        self.percentile = percentile
        self.n_grid = n_grid
        self._check_data()

    def _check_data(self):
        """
        Helper function to check that the partial dependence object is
        correctly instantiated.

        Raises:
            ValueError -- if target variable is not provided
            ValueError -- if target variable is not a valid feature
            ValueError -- if target feature is not one of ['float64', \
                          'int64', 'category', 'bool']
            AssertionError -- if `training_df` is not a pandas dataframe
        """

        assert(isinstance(self.training_df, pd.core.frame.DataFrame)), \
            '`training_df` must be a pandas dataframe'

        if self.feature_name is None:
            raise ValueError('target variable must be provided')

        if self.feature_name not in self.training_df.columns:
            raise ValueError('target variable `%s` is not a feature'
                             % str(self.feature_name))

        valid_dtypes = [
            'float64',
            'int64',
            'float32',
            'int32',
            'category',
            'bool']
        if self.feature_name not in \
                self.training_df.select_dtypes(include=valid_dtypes).columns:
            raise ValueError('Type of target feature must be one of float64, \
                              int64, category or bool')

    def plot_partial_dependence(self, label=0):
        """
        Plots partial dependence function for the given label

        Parameters
        ----------
        label_name: str or int
            Name of the label or label index for which partial dependence \
            function is to be plotted. Defaults to integer 0, which corresponds \
            to the first label for classifier models or the predicted target
            for regressor models.

        Returns
        -------
        fig : figure
            The Matplotlib Figure object.
        ax : Axis object
            An Axis object, for the plot.
        """

        self._check_label(label)
        print("Calculating partial dependence function")
        pd_vals, x_vals = self.calculate()
        print("Finished")

        # Plot one dimensional PDP
        # This hardcoded value provides good aspect-ratio
        plt.figure(figsize=(12, 8))
        print("Plotting partial dependence function for the given label")

        if hasattr(self.model, 'predict_proba'):
            label_idx = list(self.model.classes_).index(label) if \
                isinstance(label, str) else label
            plt.plot(x_vals, pd_vals[label_idx], lw=1.5)
            fig = plt.gcf()
        else:
            plt.plot(x_vals, pd_vals, lw=1.5)
            fig = plt.gcf()
        return fig, fig.axes[0]

    def _check_label(self, label):
        """
        Helper function to check that label to partial dependence \
            plotter is passed correcty.

        Arguments:
            label {[int or str]} -- label name of label index

        Raises:
            ValueError -- if label name doesn't exist or if label index is \
                incorrect
            AssertionError -- for regression models, label must be 0
        """

        if hasattr(self.model, 'predict_proba'):  # Classification model;
                                                 # check correctness of label
            if isinstance(label, str) and label not in self.model.classes_:
                raise ValueError('label `%s` not a valid label' % str(label))
            elif isinstance(label, int) and label not in \
                    range(len(self.model.classes_)):
                raise ValueError(
                    'label `%s` not a valid label index' %
                    str(label))
        else:
            assert(label == 0), 'For regression model label=0'

    def calculate(self):
        """
        Calculates partial dependence function for the feature `feature_name`
        Uses sklearn's partial dependence function if model is a gradient
        boosted tree, otherwise implements algorithm from scratch

        Returns
        -------
        pd_vals : numpy ndarray, shape=(no_of_classes, `n_grid`);
                  partial dependence function values for the feature `feature_name`,
                  evaluated on a grid of `n_grid` equally spaced points

        For regression models, shape = (1, `n_grid`);
        For classification models, shape = (`n_classes`, `n_grid`).

        x_vals : numpy ndarray, shape=(`n_grid`, )
                 The values of the target feature `feature_name` for which partial
                 dependence function is computed
        """
        from sklearn.ensemble.gradient_boosting import BaseGradientBoosting
        if isinstance(
                self.model, BaseGradientBoosting):  # Use sklearn implementation
            pd_vals, x_vals = self._calc_partial_dependence_gbt()
        else:  # Implement algorithm from scratch
            pd_vals, x_vals = self._calc_partial_dependence_custom()
        return pd_vals, x_vals

    def _calc_partial_dependence_gbt(self):
        """
        Calculate partial dependence function if model instance is a gradient
        boosted tree. Uses sklearn's implementation.
        """

        from sklearn.ensemble.partial_dependence import partial_dependence
        feature_name_idx = list(
            self.training_df.columns).index(
            self.feature_name)
        pd_vals, x_vals = partial_dependence(gbrt=self.model,
                                             target_variables=feature_name_idx,
                                             grid=None,
                                             X=self.training_df,
                                             percentiles=self.percentile,
                                             grid_resolution=self.n_grid)

        x_vals = x_vals[0]  # Get grid of x-axis values

        if hasattr(self.model, 'predict_proba'):  # Classfication model
            # Compute the probabilities for each class. Reference: Eqns (29) and (30)
            # in https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451
            odds = np.exp(pd_vals)
            prob_vals = odds / odds.sum(axis=0)
            return prob_vals, x_vals
        return pd_vals.T, x_vals  # Regression model

    def _calc_partial_dependence_custom(self):
        """
        Calculate partial dependence function if model instance is not a gradient
        boosted tree. Custom implementation.
        """
        _df = self.training_df.copy()

        # Get a prediction object from the trained model
        predict = self.model.predict_proba if hasattr(
            self.model, 'predict_proba') else self.model.predict

        # Build a grid of feature values. If feature is non-categorical use equally
        # spaced points b/w given percentile values. Otherwise use all possible
        # values.)
        if self.feature_name in _df.select_dtypes(
                include=['float64', 'int64', 'float32', 'int32']).columns:
            lower_limit = np.percentile(_df[self.feature_name],
                                        self.percentile[0] * 100)
            upper_limit = np.percentile(_df[self.feature_name],
                                        self.percentile[1] * 100)
            x_vals = np.linspace(start=lower_limit, stop=upper_limit,
                                 num=self.n_grid)
        else:
            x_vals = _df[self.feature_name].unique()

        # Get the mean prediction
        pd_vals = np.asarray([self._get_mean_prediction(_df, predict, val)
                              for val in x_vals]).T
        return pd_vals, x_vals

    def _get_mean_prediction(self, _df, predict, val):
        """
        Calculate expected partial dependence prediction for `val
        """

        _df[self.feature_name] = val
        return np.mean(predict(_df), axis=0).tolist()
