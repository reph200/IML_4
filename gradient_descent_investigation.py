import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import BaseLR
from cross_validate import cross_validate
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from loss_functions import misclassification_error

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from sklearn.metrics import roc_curve, auc, accuracy_score
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(val, weight, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for name, module in {"L1": L1, "L2": L2}.items():
        fig = go.Figure()
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()

            gd = GradientDescent(learning_rate=FixedLR(eta), max_iter=1000, tol=1e-5, callback=callback)
            gd.fit(module(weights=init.copy()), None, None)
            fig.add_trace(go.Scatter(y=values, mode="lines", name=rf"$\eta={eta}$"))
            if eta == 0.01:
                # Plot algorithm's descent path
                # plot_descent_path(module, np.array([init] + weights), f"{name} - Learning Rate: {eta} ") \
                #     .write_image(f"../figures/gd_{name}_eta_{eta}.png")
                plot_descent_path(module, np.array([init] + weights), f"{name} - Learning Rate: {eta} ").show()
            print(f"for learning rate: {eta} - Lowest loss achieved for {name} module: {min(values)}")

        fig.update_layout(title=f"{name} GD Convergence For Different Learning Rates",
                          xaxis_title="GD Iteration", yaxis_title="Norm")
        fig.show()





def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data

    callback, losses, weights = get_gd_state_recorder_callback()
    learninig_rate = 0.0001
    max_iterations = 20000
    gd = GradientDescent(learning_rate=FixedLR(learninig_rate), max_iter=max_iterations, callback=callback)
    model = LogisticRegression(solver=gd).fit(X_train.values, y_train.values)

    # Predict probabilities on the test set
    y_prob = model.predict_proba(X_test.values)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))

    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='FPR',
        yaxis_title='TPR',
        showlegend=True
    )
    fig.show()

    # Determine the optimal alpha
    optimal_idx = np.argmax(tpr - fpr)
    optimal_alpha = thresholds[optimal_idx]
    print(f"Optimal alpha: {optimal_alpha}")

    # Evaluate model's test error with optimal alpha
    model.alpha_ = optimal_alpha
    test_error = model._loss(X_test.values, y_test.values)
    print(f"Test error with optimal alpha: {test_error}")

    # Fitting l1-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas= [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    penalty = "l1"
    # Running cross validation
    # Initialize array to store scores
    scores = np.zeros((len(lambdas), 2))

    # Perform cross-validation for each lambda value
    for idx, lam in enumerate(lambdas):
        # Initialize gradient descent and logistic regression model
        gd = GradientDescent(learning_rate=FixedLR(learninig_rate), max_iter=max_iterations)
        logistic_model = LogisticRegression(solver=gd, penalty=penalty, lam=lam, alpha=0.5)

        # Perform cross-validation and store scores
        train_error, val_error = cross_validate(estimator=logistic_model, X=X_train.values, y=y_train.values,
                                                scoring=misclassification_error)
        scores[idx] = [train_error, val_error]
        print(f'Lambda: {lam}, Train Error: {train_error:.4f}, Validation Error: {val_error:.4f}')

    # Plot training and validation errors
    fig = go.Figure([
        go.Scatter(x=lambdas, y=scores[:, 0], mode='lines+markers', name="Train Error"),
        go.Scatter(x=lambdas, y=scores[:, 1], mode='lines+markers', name="Validation Error")
    ], layout=go.Layout(
        title="Train and Validation Errors (averaged over k-folds)",
        xaxis=dict(
            title="Lambda",
            type="log",
            tickmode='array',
            tickvals=lambdas,
            ticktext=[str(lam) for lam in lambdas]
        ),
        yaxis=dict(
            title="Error"
        )
    ))
    fig.show()

    # Select the best lambda based on validation error
    optimal_lambda = lambdas[np.argmin(scores[:, 1])]


    # Train final model with optimal lambda on the entire training set
    final_gd = GradientDescent(learning_rate=FixedLR(learninig_rate), max_iter=max_iterations)
    final_model = LogisticRegression(solver=final_gd, penalty=penalty, lam=optimal_lambda, alpha=0.5)
    final_model.fit(X_train.values, y_train.values)

    # Evaluate the final model on the test set
    test_error = final_model._loss(X_test.values, y_test.values)

    print(f"\tOptimal Regularization Parameter: {optimal_lambda}")
    print(f"\tModel achieved test error of {round(test_error, 4)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
