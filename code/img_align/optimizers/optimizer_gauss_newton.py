
# @brief Optimization algorithm interface.
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import numpy as np
from timeit import default_timer as timer
from img_align.optimizers import Optimizer
from img_align.cost_functions import CostFunL2Images


class OptimizerGaussNewton(Optimizer):

    def __init__(self,
                 cost_function,
                 max_iter=20,
                 tol_gradient=0.001,
                 tol_params=0.00001,
                 show_iter=False,
                 profiling=False):

        if not isinstance(cost_function, CostFunL2Images):
            raise ValueError('Only CostFunSquaredL2NormImages cost functions allowed!')

        self.cost_function = cost_function
        self.max_iter = max_iter
        self.tol_gradient = tol_gradient
        self.tol_params = tol_params
        self.show_iter = show_iter
        self.profiling = profiling
        self.profiling_info = dict()

        # The 0 index is the first iteration costs and
        #  the len(iterations_costs)-1 is the
        #  last iteration cost. Every time
        self.profiling_info['iter_costs'] = []
        self.profiling_info['iter_time'] = []
        self.profiling_info['iter_gradient_norm'] = []

    def getProfilingInfo(self):
        return self.profiling_info

    def solve(self, frame, former_params):
        k = 0
        found = False
        gradient_norm = 0.0
        new_params = np.zeros(former_params.shape)
        current_params = np.zeros(former_params.shape)

        if self.show_iter:
            print "\n"
            print "  Iter        F(x)        Gradient        "
            print "  ----        ----        --------        "

        self.iter_costs = []
        current_params = np.copy(former_params)

        while not found and (k < self.max_iter):
            # Start timer per iteration:
            start = timer()

            k += 1

            # compute residuals to minimize for new motion parameters
            residuals = self.cost_function.computeResiduals(frame, current_params)

            # Compute normal equations
            invJ = self.cost_function.computeJacobianPseudoInverse(current_params)
            delta = np.dot(invJ, residuals)

            # 2nd stopping criterion: The increment in the parameters vector
            # is under a given threshold.
            norm_delta = np.linalg.norm(delta)
            if norm_delta < self.tol_params:
                found = True
                if self.show_iter:
                    print "STOP. Parameters not increasing: norm(delta) = ", norm_delta

            # Compute parameters update
            new_params = self.cost_function.updateMotionParams(current_params, delta)

            # Compute gradient instantiated in current x
            residual = self.cost_function.computeResiduals(frame, new_params)
            J = self.cost_function.computeJacobian(new_params)
            gradient = np.dot(J.T, residual)  # residual.T column vector
            gradient_norm = np.linalg.norm(gradient)

            cost = self.cost_function.computeValue(residuals, new_params, delta, frame)
            self.iter_costs.append(cost)

            if self.show_iter:
                 print "  ", k, "        ", cost, "        ", gradient_norm, "\n"

            # 1st stopping criterion: the norm of the gradient is under a given threshold.
            if gradient_norm < self.tol_gradient:
                 found = True
                 if self.show_iter:
                     print "STOP. Norm of the gradient is bellow threshold."

            current_params = np.copy(new_params)

            # End timer per iteration:
            end = timer()
            iter_time = (end - start)

            if self.profiling:
                self.profiling_info['iter_costs'].append(cost)
                self.profiling_info['iter_time'].append(iter_time)
                self.profiling_info['iter_gradient_norm'].append(gradient_norm)

        if (k > self.max_iter) and self.show_iter:
            print "STOP. Max number of iterations exceeded: ", self.max_iter, "\n"

        return current_params