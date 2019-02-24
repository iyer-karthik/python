In Machine learning there is a recurrent dilemma between performance and interpretation. More complex models usually result in better performance but can be black-boxy and hard to interpret. Simpler models like Linear Regression have a straight forward interpretation but tend to suffer from inflexibility. Partial dependence plots (PDPs) help visualize and quanitify the relationship between features and predicted outcomes. They are are particularly helpful with black-box models like Random Forests, Gradient Boosted Trees, Neural Networks where interpretability is an issue.

PDPs provide a way to look at the directionality of feature(s) with respect to the predicted outcome. Though ensemble methods like random forests and gradient boosted trees give us a way to measure importance of features that give a sense of which features tend to have the most effect on the variability of the outcome, they don't give a way to measure the directionality. Partial dependence plots, introduced by Friedman in 2001, can help in interpreting complex machine learning algorithms by quantifying the directionaity of relationship between the outcome and features.

PDPs help visualize the relationship between a subset of features (typically 1-2) and the response, while accounting for the average effect of the other predictors in the model.

`scikit-learn` currently (as of Feb 2019) implements partial dependence plots only for gradient boosted trees. The goal of this project is to extend sklearn's partial dependence functionality to any classifer and any regressor.

***References***

[Original paper](https://projecteuclid.org/euclid.aos/1013203451)

[scikit-learn implementation](https://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html)

[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
