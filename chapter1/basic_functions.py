from typing import Callable
from typing import List
from typing import Tuple

from numpy import ndarray

import numpy as np
import matplotlib.pyplot as pyplot

ArrayFunction = Callable[[ndarray], ndarray]

Chain = List[ArrayFunction]

ChainDerivFunction = Callable[[Chain, ndarray], ndarray]

ChainFunction = Callable[[Chain, ndarray], ndarray]


def square(x: ndarray) -> ndarray:
    """
    Square each element in the input ndarray
    """
    return np.power(x, 2)


def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element in the input ndarray
    """
    return 1 / (1 + np.exp(-x))


def leaky_relu(x: ndarray) -> ndarray:
    """
    Apply "Leaky ReLU" function to each element in ndarray
    """
    return np.maximum(0.2 * x, x)


def deriv(func: ArrayFunction,
          input_: ndarray,
          delta: float = 0.0001) -> ndarray:
    """
    Evaluates the derivative of a function "func" at every element
    """
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    """
    Evaluate two functions in a row, in a "Chain"
    """
    assert len(chain) is 2, \
        "Length of input 'chain' should be 2"

    f1, f2 = chain[0], chain[1]

    return f2(f1(x))


def chain_length_3(chain: Chain,
                   x: ndarray) -> ndarray:
    """
       Evaluate two functions in a row, in a "Chain"
       """
    assert len(chain) is 3, \
        "Length of input 'chain' should be 3"

    f1, f2, f3 = chain[0], chain[1], chain[2]

    return f3(f2(f1(x)))


def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x)))' = f2'(f1(x))*f1'(x)
    """
    assert len(chain) is 2, \
        "This function requires 'Chain objects of length 2'"

    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"

    f1, f2 = chain[0], chain[1]

    # f1(x)
    f1x = f1(input_range)

    # df1/dx
    df1dx = deriv(f1, input_range)

    # df2/du u=f1(x)
    df2du = deriv(f2, f1x)

    return df2du * df1dx


def chain_deriv_3(chain: Chain,
                  input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivation of three nested functions:
    (f3(f2(f1(x))))'=f3'(f2(f1(x)))*f2'(f1(x))*f1'(x)
    """
    assert len(chain) is 3, \
        "This function requires 'Chain objects of length 3'"

    f1, f2, f3 = chain[0], chain[1], chain[2]

    # w(x)=f3(v(x)), v(x)=f2(u(x)), u(x)=f1(x)

    # u(x)
    ux = f1(input_range)
    # du/dx
    dudx = deriv(f1, input_range)
    # v(x)
    vx = f2(ux)
    # dv/du
    dvdu = deriv(f2, ux)
    # dw/dv
    dwdv = deriv(f3, vx)

    return dwdv * dvdu * dudx


def plot_chain_deriv(chain_deriv_func: ChainDerivFunction,
                     chain: Chain,
                     input_range: ndarray,
                     call_show: bool = True) -> None:
    pyplot.plot(chain_deriv_func(chain, input_range))
    if call_show:
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()


def plot_chain(chain_func: ChainFunction,
               chain: Chain,
               input_range: ndarray,
               call_show: bool = True) -> None:
    pyplot.plot(chain_func(chain, input_range))
    if call_show:
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()


def multiple_input_add(x: ndarray,
                       y: ndarray,
                       sigma: ArrayFunction) -> ndarray:
    """
    Function with multiple inputs and addition, forward pass.
    """
    assert x.shape is y.shape

    a = x + y

    return sigma(a)


def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: ArrayFunction) -> Tuple[ndarray, ndarray]:
    """
    Computes the derivative of this simple function with respect to
    both inputs.
    """
    # Compute "forward pass"
    a = x + y

    # Compute derivatives
    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


def matmul_forward(x: ndarray,
                   w: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication.
    """

    assert x.shape[1] is w.shape[0], \
        """
    For matrix multiplication, the number of columns in the first array should
    match the number of rows in the second; instead the number of columns in the
    first array is {} and the number of rows in the second array is {}.
    """.format(x.shape[1], w.shape[0])

    # matrix multiplication
    n = np.dot(x, w)

    return n


def matmul_backward_first(x: ndarray, w: ndarray) -> ndarray:
    """
    Compute the backward pass of a matrix multiplication with respect to the first argument
    """
    # backward pass
    dndx = np.transpose(w, (1, 0))
    return dndx


def matrix_forward_extra(x: ndarray, w: ndarray, sigma: ArrayFunction) -> ndarray:
    """
    Computes the forward pass of a function involving matrix multiplication, one extra function
    """
    assert x.shape[1] is w.shape[1]

    # matrix multiplication
    n = np.dot(x, w)
    # feeding the output of the matrix multiplication through sigma
    s = sigma(n)
    return s


def matrix_function_backward_1(x: ndarray, w: ndarray, sigma: ArrayFunction) -> ndarray:
    """
    Computes the derivative of matrix function with respect to the first element
    """
    assert x.shape[1] is w.shape[1]

    # matrix multiplication
    n = np.dot(x, w)

    # deeding the output of the matrix multiplication through sigma
    s = sigma(n)

    # backward calculation
    dsdn = deriv(sigma, n)

    # dndx
    dndx = np.transpose(w, (1, 0))

    # multiply them together; since dndx is 1x1 here, order doesn't matter
    return np.dot(dsdn, dndx)


def test() -> None:
    chain_1 = [square, sigmoid]
    chain_2 = [sigmoid, square]
    chain_3 = [leaky_relu, sigmoid, square]

    plot_range = np.arange(-3, 3, 0.01)

    plot_chain(chain_length_2, chain_1, plot_range, False)
    plot_chain_deriv(chain_deriv_2, chain_1, plot_range)

    plot_chain(chain_length_2, chain_2, plot_range, False)
    plot_chain_deriv(chain_deriv_2, chain_2, plot_range)

    plot_chain(chain_length_3, chain_3, plot_range, False)
    plot_chain_deriv(chain_deriv_3, chain_3, plot_range)


test()
