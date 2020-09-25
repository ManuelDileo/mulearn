# AUTOGENERATED! DO NOT EDIT! File to edit: 00_kernel.ipynb (unless otherwise specified).

__all__ = ['Kernel', 'LinearKernel', 'PolynomialKernel', 'HomogeneousPolynomialKernel', 'GaussianKernel',
           'HyperbolicKernel', 'PrecomputedKernel']

# Cell

import numpy as np
import pytest

# Cell

class Kernel:

    def __init__(self):

        self.precomputed = False

    def compute(self, arg_1, arg_2):

        raise NotImplementedError(
            'this class does not implement the compute method')

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __nonzero__(self):
        return True

    def __hash__(self):
        return hash(self.__repr__())

    @classmethod
    def get_default(cls):
        r'''Return the default kernel.
        '''

        return LinearKernel()


# Cell

class LinearKernel(Kernel):

    def compute(self, arg_1, arg_2):
        r'''
        Compute the dot product between `arg_1` and `arg_2`, where the
        dot product $x \cdot y$ is intended as the quantity
        $\sum_{i=1}^n x_i y_i$, $n$ being the dimension of both
        $x$ and $y$.

        - `arg_1`: first dot product argument (iterable).

        - `arg_2`: second dot product argument (iterable).

        Returns: kernel value (float).'''

        return float(np.dot(arg_1, arg_2))

    def __repr__(self):
        return 'LinearKernel()'


# Cell

class PolynomialKernel(Kernel):

    def __init__(self, degree):
        r'''Creates an instance of `PolynomialKernel`

        - `degree`: degree of the polynomial kernel (positive integer).
        '''

        Kernel.__init__(self)
        if degree > 0 and isinstance(degree, int):
            self.degree = degree
        else:
            raise ValueError(str(degree) +
                ' is not usable as a polynomial degree')

    def compute(self, arg_1, arg_2):
        r'''
        Compute the polynomial kernel between `arg_1` and `arg_2`,
        where the kernel value $k(x_1, x_2)$ is intended as the quantity
        $(x_1 \cdot x_2 + 1)^d$, $d$ being the polynomial degree of
        the kernel.

        - `arg_1` first argument to the polynomial kernel (iterable).

        - `arg_2` second argument to the polynomial kernel (iterable).

        Returns: kernel value (float)
        '''

        return float((np.dot(arg_1, arg_2) + 1) ** self.degree)

    def __repr__(self):
        return 'PolynomialKernel(' + repr(self.degree) + ')'


# Cell

class HomogeneousPolynomialKernel(Kernel):

    def __init__(self, degree):
        r'''Creates an instance of `HomogeneousPolynomialKernel`.

        - `degree`: polynomial degree (positive integer).
        '''

        Kernel.__init__(self)
        if degree > 0 and isinstance(degree, int):
            self.degree = degree
        else:
            raise ValueError(str(degree) +
                ' is not usable as a polynomial degree')

    def compute(self, arg_1, arg_2):
        r'''
        Compute the homogeneous polynomial kernel between `arg_1` and
        `arg_2`, where the kernel value $k(x_1, x_2)$ is intended as
        the quantity $(x_1 \cdot x_2)^d$, $d$ being the polynomial
        degree of the kernel.

        - `arg_1`: first argument to the homogeneous polynomial kernel
          (iterable).

        - `arg_2`: second argument to the homogeneous polynomial kernel
          (iterable).

        Returns: kernel value (float).
        '''

        return float(np.dot(arg_1, arg_2) ** self.degree)

    def __repr__(self):
        return 'HomogeneousPolynomialKernel(' + repr(self.degree) + ')'


# Cell

class GaussianKernel(Kernel):

    def __init__(self, sigma=1):
        r'''
        Creates an instance of `GaussianKernel`.

        - `sigma`: gaussian standard deviation (positive float).
        '''

        Kernel.__init__(self)
        if sigma > 0:
            self.sigma = sigma
        else:
            raise ValueError(f'{sigma} is not usable '
                             'as a gaussian standard deviation')

    def compute(self, arg_1, arg_2):
        r'''
        Compute the gaussian kernel between `arg_1` and `arg_2`,
        where the kernel value $k(x_1, x_2)$ is intended as the quantity
        $\mathrm e^{-\frac{||x_1 - x_2||^2}{2 \sigma^2}}$, $\sigma$
        being the kernel standard deviation.

        - `arg_1`: first argument to the gaussian kernel (iterable).

        - `arg_2`: second argument to the gaussian kernel (iterable).

        Returns: kernel value (float).
        '''

        diff = np.linalg.norm(np.array(arg_1) - np.array(arg_2)) ** 2
        return float(np.exp(-1. * diff / (2 * self.sigma ** 2)))

    def __repr__(self):
        return 'GaussianKernel(' + repr(self.sigma) + ')'


# Cell

class HyperbolicKernel(Kernel):

    def __init__(self, scale=1, offset=0):
        r'''Creates an instance of `HyperbolicKernel`.

        - `scale`: scale constant (float).

        - `offset`: offset constant (float).
        '''

        Kernel.__init__(self)
        self.scale = scale
        self.offset = offset

    def compute(self, arg_1, arg_2):
        r'''Compute the hyperbolic kernel between `arg_1` and `arg_2`,
        where the kernel value $k(x_1, x_2)$ is intended as the quantity
        $\tanh(\alpha x_1 \cdot x_2 + \beta)$, $\alpha$ and $\beta$ being the
        scale and offset values, respectively.

        - `arg_1`: first argument to the hyperbolic kernel (iterable).

        - `arg_2`: second argument to the hyperbolic kernel (iterable).

        Returns: kernel value (float).
        '''

        #return float(tanh(self.scale * dot(arg_1, arg_2) +  self.offset))
        dot_orig = np.dot(np.array(arg_1), np.array(arg_2))
        return float(np.tanh(self.scale * dot_orig +  self.offset))

    def __repr__(self):
        return 'HyperbolicKernel(' + repr(self.scale) + ', ' + repr(self.offset) + ')'


# Cell

class PrecomputedKernel(Kernel):

    def __init__(self, kernel_computations):
        r'''
        Creates an instance of ``PrecomputedKernel``.

        - `kernel_computations`: kernel computations (square matrix of float
          elements).
        '''

        Kernel.__init__(self)
        self.precomputed = True
        try:
            (rows, columns) = np.array(kernel_computations).shape
        except ValueError:
            raise ValueError('The supplied matrix is not array-like ')

        if rows != columns:
            raise ValueError('The supplied matrix is not square')

        self.kernel_computations = kernel_computations

    def compute(self, arg_1, arg_2):
        r'''Compute a value of the kernel, given the indices of the
        corresponding objects. Note that each index should be enclosed
        within an iterable in order to be compatible with sklearn.

        - ``arg_1``: first kernel argument (iterable contining one int).

        - ``arg_2``: second kernel argument (iterable contining one int).

        Returns: kernel value (float).
        '''

        return float(self.kernel_computations[arg_1[0]][arg_2[0]])

    def __repr__(self):
        return 'PrecomputedKernel(' + repr(self.kernel_computations) + ')'
