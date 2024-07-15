from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from base_module import BaseModule
from base_learning_rate import BaseLR
from learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        # w = np.zeros_like(f.weights_)
        # best_w = w.copy()
        # best_val = float('inf')
        # sum_w = np.zeros_like(w)
        # w_prev = w.copy()
        # delta = self.tol_
        # t = 0
        #
        # while t < self.max_iter_ and delta >= self.tol_:
        #     eta = self.learning_rate_.lr_step(t=t)
        #     grad = f.compute_jacobian(X=X, y=y)
        #     w_prev = w.copy()
        #     w -= eta * grad
        #     delta = np.linalg.norm(w - w_prev)
        #
        #     current_val = f.compute_output(X=X, y=y)
        #     if current_val < best_val:
        #         best_val = current_val
        #         best_w = w.copy()
        #
        #     sum_w += w
        #
        #     self.callback_(solver=self, weight=w, val=current_val, grad=grad, t=t, eta=eta, delta=delta)
        #     t += 1
        #
        # if self.out_type_ == "last":
        #     return w
        # elif self.out_type_ == "best":
        #     return best_w
        # elif self.out_type_ == "average":
        #     return sum_w / t
        # else:
        #     raise ValueError("Invalid output type")

        t, delta, prev_step = 0, self.tol_, np.copy(f.weights)
        best_val, best_weights, cweights = np.Inf, None, np.zeros_like(f.weights)

        while t < self.max_iter_ and delta >= self.tol_:
            # forward pass - computing the loss and saving all intermediate calculations
            val = f.compute_output(X=X, y=y)
            # backward pass - In neural networks, it is dependant on the existing forward pass
            grad = f.compute_jacobian(X=X, y=y)
            # deciding on the step size
            eta = self.learning_rate_.lr_step(f=f, x=X, dx=-grad, t=t)

            # Performing descent step - taking a step in the negative direction
            f.weights -= eta * grad
            cweights += f.weights

            # Update variables for next iteration
            t, prev_step, delta = t + 1, np.copy(f.weights), np.linalg.norm(f.weights - prev_step)
            if val < best_val:
                best_val, best_weights = val, f.weights

            # Call the callback function
            self.callback_(solver=self, weight=np.copy(f.weights), val=val, grad=grad, t=t, eta=eta, delta=delta)

        if self.out_type_ == "last":
            return f.weights
        if self.out_type_ == "best":
            return best_weights
        return cweights / t
