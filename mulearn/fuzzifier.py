# AUTOGENERATED! DO NOT EDIT! File to edit: 01_fuzzifiers.ipynb (unless otherwise specified).

__all__ = ['Fuzzifier', 'CrispFuzzifier']

# Cell

import numpy as np
import pytest
from scipy.optimize import curve_fit

# Cell

class Fuzzifier:
    def __init__(self, xs=None, mus=None):
        self.xs = xs
        self.mus = mus

    def get_r_to_mu(self,
                    sq_radius, # was SV_square_distance
                    sample,
                    x_to_sq_dist): # was estimated_square_distance_from_center
        '''Transforms the square distance between center of the learnt sphere
        and the image of a point in original space into the membership degree
        of the latter to the induced fuzzy set.

        Not implemented in the base fuzzifier class.

        - `sq_radius`: squared radius of the learnt sphere (float).

        - `sample`: sample of points in original space (iterable).

        - `x_to_sq_dist`: mapping of a point in original space into the
          square distance of its image from the center of the learnt sphere
          (function).
        '''

        raise NotImplementedError(
        'the base class does not implement get_r_to_mu method')

    def get_fuzzified_membership(self,
                                 sq_radius, # was SV_square_distance
                                 sample,
                                 x_to_sq_dist, # was estimated_square_distance_from_center
                                 return_profile=False):
        '''Return the induced membership function.

        - `sq_radius`: squared radius of the learnt sphere (float).

        - `sample`: sample of points in original space (iterable).

        - `x_to_sq_dist`: mapping of a point in original space into the
          square distance of its image from the center of the learnt sphere
          (function).

        - `return_profile` flag triggering the generation of the graph
          of the fuzzifier to be returned alongside the fuzzifier itself
          (bool, default=False).

        Returns:

        - if `return_profile` is `False`: membership function (function)
        - if `return_profile` is `True`: list containing the membership
          function (function) and the salient coordinates of the graph of
          the fuzzifier (list), respectively in first and
          second position.
        '''
        r_to_mu = self.get_r_to_mu(sq_radius, sample,x_to_sq_dist)

        def estimated_membership(x):
            r = x_to_sq_dist(np.array(x))
            return r_to_mu(r)

        result = [estimated_membership]

        if return_profile:
            rdata = list(map(x_to_sq_dist, self.xs))
            rdata_synth = np.linspace(0, max(rdata)*1.1, 200)
            estimate = list(map(r_to_mu, rdata_synth))
            result.append([rdata, rdata_synth, estimate, sq_radius])

        return result

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

# Cell

class CrispFuzzifier(Fuzzifier):
    def __init__(self, xs=None, mus=None):
        super().__init__(xs, mus)

        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

    def get_r_to_mu(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        '''Maps distance from center to membership fitting a function
           having the form r -> 1 if r < r_crisp else 0  '''

        def r_to_mu_prototype(r, r_crisp):
            result = np.ones(len(r))
            result[r > r_crisp] = 0
            return result

        rdata = np.fromiter(map(estimated_square_distance_from_center,
                                self.xs),
                            dtype=float)
        popt, _ = curve_fit(r_to_mu_prototype, rdata, self.mus)
                            # bounds=((0,), (np.inf,)))

        if popt[0] < 0:
            raise ValueError('Profile fitting returned a negative parameter')
        return lambda r: r_to_mu_prototype([r], *popt)[0]

    def __repr__(self):
        return 'CrispFuzzifier({}, {})'.format(self.xs, self.mus)